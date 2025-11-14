# File: src/02_models/Wofost7.1/build_initial_conditions.py
# Description: Generates the master InitialConditions.csv file.
# ENHANCEMENT 3: This script has been enhanced to use satellite data for more dynamic initial conditions.

import datetime
import pandas as pd
import yaml
from pathlib import Path
import sys
import logging
import geopandas as gpd
import xarray as xr
import rioxarray
import re
import zipfile
import tempfile
from tqdm import tqdm
from joblib import Parallel, delayed

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG
ERA5_SOIL_DATA_DIR = Path("data/01_raw/era5_land_monthly_soil")
SATELLITE_FEATURES_PATH = config.PROCESSED_DATA_DIR / f"satellite_features_districts_{config.WOFOST_CONFIG['START_YEAR']}-{config.WOFOST_CONFIG['END_YEAR']}.csv"
NDVI_ANOMALY_SENSITIVITY = 5.0 # Defines how much WAV (in cm) changes per unit of NDVI anomaly

class DynamicSowingManager:
    """
    Determines a dynamic sowing date based on weather conditions.
    """
    def __init__(self, sowing_window_start_month=3, sowing_window_start_day=15,
                 sowing_window_end_month=4, sowing_window_end_day=30,
                 temp_threshold_c=7.0, temp_avg_period_days=7):
        self.start_month = sowing_window_start_month
        self.start_day = sowing_window_start_day
        self.end_month = sowing_window_end_month
        self.end_day = sowing_window_end_day
        self.threshold = temp_threshold_c
        self.period = temp_avg_period_days

    def find_sowing_date(self, weather_df_year: pd.DataFrame) -> datetime.date:
        """
        Finds the first suitable sowing date within the year's weather data.
        """
        df = weather_df_year.copy()
        df['mean_temp'] = (df['tmin'] + df['tmax']) / 2
        df['temp_ma'] = df['mean_temp'].rolling(window=self.period, min_periods=self.period).mean()

        try:
            year = df['date'].dt.year.iloc[0]
            window_start = datetime.date(year, self.start_month, self.start_day)
            window_end = datetime.date(year, self.end_month, self.end_day)
        except IndexError:
            return datetime.date(2000, self.start_month, self.start_day) # Fallback

        sow_mask = (
            (df['date'].dt.date >= window_start) &
            (df['date'].dt.date <= window_end) &
            (df['temp_ma'] >= self.threshold)
        )
        potential_sow_days = df.loc[sow_mask]

        if not potential_sow_days.empty:
            return potential_sow_days['date'].dt.date.iloc[0]
        else:
            return window_end

def _precalculate_wav_from_era5(gdf_districts, static_df):
    """
    Calculates WAV for all districts and years using ERA5-Land data.
    """
    logging.info("--- Pre-calculating WAV from ERA5-Land data ---")
    era5_files = sorted(list(ERA5_SOIL_DATA_DIR.glob("*_FEBRUARY.nc")))
    if not era5_files:
        logging.error(f"FATAL: No ERA5-Land NetCDF files found in '{ERA5_SOIL_DATA_DIR}'.");
        sys.exit(1)

    all_years_results = []
    pbar = tqdm(era5_files, desc="Processing ERA5-Land Files")
    for nc_file in pbar:
        year_match = re.search(r'_(\d{4})_FEBRUARY', nc_file.name)
        if not year_match: continue
        year = int(year_match.group(1))
        pbar.set_postfix_str(f"Year: {year}")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                actual_nc_path = nc_file
                if zipfile.is_zipfile(nc_file):
                    with zipfile.ZipFile(nc_file, 'r') as zf: zf.extractall(temp_dir)
                    extracted = list(Path(temp_dir).glob('*.nc'))
                    if not extracted: continue
                    actual_nc_path = extracted[0]

                with xr.open_dataset(actual_nc_path, engine='netcdf4') as rds:
                    da = rds['swvl1'].squeeze(drop=True).rio.set_spatial_dims(x_dim="longitude", y_dim="latitude").rio.write_crs("EPSG:4326")

                    year_results = []
                    for _, district in gdf_districts.iterrows():
                        try:
                            clipped = da.rio.clip([district.geometry], gdf_districts.crs, drop=True)
                            mean_swvl1 = clipped.mean().item()
                        except rioxarray.exceptions.NoDataInBounds:
                            centroid = district.geometry.centroid
                            mean_swvl1 = da.sel(longitude=centroid.x, latitude=centroid.y, method='nearest').item()

                        if pd.notna(mean_swvl1):
                            year_results.append({'district_no': district['district_no'], 'mean_feb_swvl1': mean_swvl1})

                    if year_results:
                        df = pd.DataFrame(year_results)
                        df['year'] = year
                        all_years_results.append(df)
        except Exception as e:
            logging.error(f"CRITICAL FAILURE processing {nc_file.name}. Error: {e}", exc_info=True)
            continue

    if not all_years_results:
        logging.error("FATAL: No soil moisture data could be processed."); sys.exit(1)

    full_swvl1_df = pd.concat(all_years_results, ignore_index=True)
    final_df = pd.merge(full_swvl1_df, static_df[['district_no', 'SMW', 'RDMSOL']], on='district_no')
    final_df['WAV'] = (final_df['mean_feb_swvl1'] - final_df['SMW']) * final_df['RDMSOL']
    final_df['WAV'] = final_df['WAV'].clip(lower=0)
    return final_df[['year', 'district_no', 'WAV']]

def process_district_year(group_name, group_df, sowing_manager):
    """
    Helper function to find the sowing date for a single district-year group.
    """
    year, district_no = group_name
    sowing_date = sowing_manager.find_sowing_date(group_df)
    return {'year': year, 'district_no': district_no, 'sowing_date': sowing_date}

def main():
    """
    Main function to build the initial conditions file.
    """
    logging.info("--- Building InitialConditions.csv (Enhanced with Satellite Data) ---")

    # Load static inputs
    try:
        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            cp_variety = yaml.safe_load(f)['CropParameters']['Varieties']['Sugarbeet_601']
        static_df = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_SOIL_FEATURES'], dtype={'district_no': str})
        if static_df['RDMSOL'].mean() < 10: static_df['RDMSOL'] *= 100
        gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH).rename(columns={'id': 'district_no'})
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        gdf_districts = gdf_districts.to_crs("EPSG:4326")
        historical_weather_df = pd.read_csv(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'], parse_dates=['date'], dtype={'district_no': str})
        df_satellite = pd.read_csv(SATELLITE_FEATURES_PATH, dtype={'district_no': str})
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required data file was not found. Error: {e}"); sys.exit(1)

    # 1. Pre-calculate all WAV values from ERA5
    df_wav = _precalculate_wav_from_era5(gdf_districts, static_df)

    # 2. Determine dynamic sowing dates in parallel
    sowing_manager = DynamicSowingManager()
    grouped = historical_weather_df.groupby(['year', 'district_no'])

    logging.info("Finding sowing dates in parallel...")
    sowing_dates = Parallel(n_jobs=-1)(delayed(process_district_year)(name, group, sowing_manager) for name, group in tqdm(grouped))

    df_sowing = pd.DataFrame(sowing_dates)

    # 3. Merge WAV and sowing dates
    df_merged = pd.merge(df_sowing, df_wav, on=['year', 'district_no'])

    # 4. ENHANCEMENT: Incorporate satellite data
    df_merged = pd.merge(df_merged, df_satellite[['year', 'district_no', 'winter_cropland_ndvi_anomaly']], on=['year', 'district_no'], how='left')
    df_merged['winter_cropland_ndvi_anomaly'].fillna(0, inplace=True)

    # Adjust WAV based on NDVI anomaly
    wav_adjustment = df_merged['winter_cropland_ndvi_anomaly'] * NDVI_ANOMALY_SENSITIVITY
    df_merged['WAV'] = (df_merged['WAV'] + wav_adjustment).clip(lower=0)
    logging.info("Applied WAV adjustment based on winter NDVI anomaly.")

    # 5. Add other static initial conditions
    df_merged['TDWI'] = cp_variety.get('TDWI', [100.0])[0]
    df_merged['RDI'] = cp_variety.get('RDI', [10.0])[0]
    df_merged['CROP_END_DATE'] = CONFIG['AGROMANAGEMENT']['CROP_END_DATE']
    df_merged['DVSEND'] = cp_variety.get('DVSEND', [2.0])[0]

    # 6. Save the final file
    output_path = config.PROCESSED_DATA_DIR / 'InitialConditions.csv'
    output_df = df_merged[['district_no', 'year', 'sowing_date', 'WAV', 'TDWI', 'RDI', 'CROP_END_DATE', 'DVSEND']]
    output_df.to_csv(output_path, index=False)
    logging.info(f"--- InitialConditions.csv with {len(output_df)} records saved to {output_path} ---")
    logging.info("Data Preview:\n" + output_df.head().to_string())

if __name__ == "__main__":
    main()
