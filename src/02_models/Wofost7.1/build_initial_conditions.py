# File: src/02_models/Wofost7.1/build_initial_conditions.py
# Description: Generates InitialConditions.csv.
# OPTIMIZED (v6.1): Respects 'DISTRICT_LIMIT' for fast testing.

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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG
ERA5_SOIL_DATA_DIR = Path("data/01_raw/era5_land_monthly_soil")
SATELLITE_FEATURES_PATH = "data/03_processed/satellite_features_districts_2001-2024.csv"

NDVI_ANOMALY_SENSITIVITY = 5.0


class DynamicSowingManager:
    def __init__(self,
                 sowing_window_start_month=3, sowing_window_start_day=10,
                 sowing_window_end_month=4, sowing_window_end_day=30,
                 temp_threshold_c=5.0,
                 precip_trafficability_limit_mm=5.0,
                 trafficability_window_days=3):

        self.start_month = sowing_window_start_month
        self.start_day = sowing_window_start_day
        self.end_month = sowing_window_end_month
        self.end_day = sowing_window_end_day
        self.temp_threshold = temp_threshold_c
        self.precip_limit = precip_trafficability_limit_mm
        self.precip_window = trafficability_window_days

    def find_sowing_date(self, weather_df_year: pd.DataFrame) -> datetime.date:
        df = weather_df_year.copy()
        df = df.sort_values('date')
        df['temp_roll'] = df['tmean'].rolling(window=3, min_periods=1).mean()
        df['precip_roll_sum'] = df['prec'].rolling(window=self.precip_window, min_periods=1).sum()

        try:
            target_year = df['date'].dt.year.mode()[0]
            window_start = datetime.date(target_year, self.start_month, self.start_day)
            window_end = datetime.date(target_year, self.end_month, self.end_day)
        except (IndexError, ValueError):
            return datetime.date(2000, 4, 15)

        sow_mask = (
                (df['date'].dt.date >= window_start) &
                (df['date'].dt.date <= window_end) &
                (df['temp_roll'] >= self.temp_threshold) &
                (df['precip_roll_sum'] <= self.precip_limit)
        )
        potential_sow_days = df.loc[sow_mask]

        if not potential_sow_days.empty:
            return potential_sow_days['date'].dt.date.iloc[0]
        else:
            return window_end


def _precalculate_wav_from_era5(gdf_districts, static_df):
    logging.info("--- Pre-calculating WAV from ERA5-Land data ---")
    era5_files = sorted(list(ERA5_SOIL_DATA_DIR.glob("*_FEBRUARY.nc")))

    if not era5_files:
        logging.error(f"FATAL: No ERA5-Land NetCDF files found in '{ERA5_SOIL_DATA_DIR}'.")
        sys.exit(1)

    if gdf_districts.crs != "EPSG:4326":
        gdf_districts = gdf_districts.to_crs("EPSG:4326")

    all_years_results = []

    pbar = tqdm(era5_files, desc="Processing ERA5-Land Files")
    for nc_file in pbar:
        year_match = re.search(r'_(\d{4})_FEBRUARY', nc_file.name)
        if not year_match: continue
        year = int(year_match.group(1))

        # Skip years outside config range to save time
        if CONFIG['START_YEAR'] and year < CONFIG['START_YEAR']: continue
        if CONFIG['END_YEAR'] and year > CONFIG['END_YEAR']: continue

        pbar.set_postfix_str(f"Year: {year}")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                actual_nc_path = nc_file
                if zipfile.is_zipfile(nc_file):
                    with zipfile.ZipFile(nc_file, 'r') as zf:
                        zf.extractall(temp_dir)
                    extracted = list(Path(temp_dir).glob('*.nc'))
                    if not extracted: continue
                    actual_nc_path = extracted[0]

                with xr.open_dataset(actual_nc_path, engine='netcdf4') as rds:
                    da = rds['swvl1'].squeeze(drop=True)
                    da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude").rio.write_crs("EPSG:4326")

                    year_results = []
                    for _, district in gdf_districts.iterrows():
                        try:
                            clipped = da.rio.clip([district.geometry], gdf_districts.crs, drop=True)
                            mean_swvl1 = clipped.mean().item()
                        except (rioxarray.exceptions.NoDataInBounds, ValueError):
                            centroid = district.geometry.centroid
                            mean_swvl1 = da.sel(longitude=centroid.x, latitude=centroid.y, method='nearest').item()

                        if pd.notna(mean_swvl1):
                            year_results.append({'district_no': district['district_no'], 'mean_feb_swvl1': mean_swvl1})

                    if year_results:
                        df = pd.DataFrame(year_results)
                        df['year'] = year
                        all_years_results.append(df)
        except Exception as e:
            logging.error(f"Failure processing {nc_file.name}: {e}")
            continue

    if not all_years_results:
        logging.error("FATAL: No soil moisture data could be processed.")
        sys.exit(1)

    full_swvl1_df = pd.concat(all_years_results, ignore_index=True)
    final_df = pd.merge(full_swvl1_df, static_df[['district_no', 'SMW', 'RDMSOL']], on='district_no')
    final_df['WAV'] = (final_df['mean_feb_swvl1'] - final_df['SMW']) * final_df['RDMSOL']
    final_df['WAV'] = final_df['WAV'].clip(lower=0.0)

    return final_df[['year', 'district_no', 'WAV']]


def _infer_missing_ndvi_anomaly(df_satellite: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    logging.info("--- Inferring missing pre-satellite NDVI anomalies ---")
    df_weather['winter_year'] = df_weather['year']
    df_weather.loc[df_weather['date'].dt.month == 12, 'winter_year'] += 1
    winter_weather = df_weather[df_weather['date'].dt.month.isin([12, 1, 2])].copy()

    winter_agg = winter_weather.groupby(['district_no', 'winter_year']).agg(
        winter_temp_mean=('tmean', 'mean'),
        winter_precip_sum=('prec', 'sum'),
        winter_rad_sum=('rad', 'sum')
    ).reset_index().rename(columns={'winter_year': 'year'})

    for col in ['winter_temp_mean', 'winter_precip_sum', 'winter_rad_sum']:
        mean_map = winter_agg.groupby('district_no')[col].transform('mean')
        winter_agg[f'{col}_anomaly'] = winter_agg[col] - mean_map

    df_full = pd.merge(df_satellite, winter_agg, on=['district_no', 'year'], how='outer')
    df_train = df_full.dropna(subset=['winter_cropland_ndvi_anomaly', 'winter_temp_mean_anomaly'])
    df_predict = df_full[df_full['winter_cropland_ndvi_anomaly'].isnull()].copy()

    if df_predict.empty:
        return df_satellite[['district_no', 'year', 'winter_cropland_ndvi_anomaly']]

    features = ['winter_temp_mean_anomaly', 'winter_precip_sum_anomaly', 'winter_rad_sum_anomaly']
    target = 'winter_cropland_ndvi_anomaly'

    # If we have no training data (e.g. in a test with limited districts/years), skip regression
    if len(df_train) < 10:
        logging.warning("Not enough data to train NDVI inference. Filling with 0.")
        return df_full[['district_no', 'year', 'winter_cropland_ndvi_anomaly']].fillna(0)

    model = Ridge(alpha=1.0)
    model.fit(df_train[features], df_train[target])

    valid_mask = df_predict[features].notna().all(axis=1)
    if valid_mask.any():
        preds = model.predict(df_predict.loc[valid_mask, features])
        df_predict.loc[valid_mask, 'inferred_ndvi'] = preds

    df_full = pd.merge(df_full, df_predict[['district_no', 'year', 'inferred_ndvi']], on=['district_no', 'year'],
                       how='left')
    df_full['winter_cropland_ndvi_anomaly'] = df_full['winter_cropland_ndvi_anomaly'].fillna(df_full['inferred_ndvi'])

    return df_full[['district_no', 'year', 'winter_cropland_ndvi_anomaly']].fillna(0)


def process_district_year(group_name, group_df, sowing_manager):
    year, district_no = group_name
    sowing_date = sowing_manager.find_sowing_date(group_df)
    return {'year': year, 'district_no': district_no, 'sowing_date': sowing_date}


def main():
    logging.info("--- Building InitialConditions.csv (Trafficability + ERA5 + Satellite) ---")

    try:
        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            cp_variety = yaml.safe_load(f)['CropParameters']['Varieties']['Sugarbeet_601']

        static_df = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_SOIL_FEATURES'], dtype={'district_no': str})
        if static_df['RDMSOL'].mean() < 10: static_df['RDMSOL'] *= 100

        gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH).rename(columns={'id': 'district_no'})
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)

        # --- OPTIMIZATION: Filter Districts Early ---
        limit = CONFIG.get('DISTRICT_LIMIT')
        if limit:
            logging.warning(f"!!! TEST MODE: Limiting to top {limit} districts !!!")
            target_districts = sorted(gdf_districts['district_no'].unique())[:limit]
            gdf_districts = gdf_districts[gdf_districts['district_no'].isin(target_districts)]
            static_df = static_df[static_df['district_no'].isin(target_districts)]
        else:
            target_districts = None

        # Load Weather
        weather_path = Path(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'])
        weather_files = list(weather_path.glob("*.csv"))

        hist_weather_df = pd.concat(
            (pd.read_csv(f, parse_dates=['date'], dtype={'district_no': str}) for f in
             tqdm(weather_files, desc="Loading Weather")),
            ignore_index=True
        )

        # --- OPTIMIZATION: Filter Weather Early ---
        if target_districts:
            hist_weather_df = hist_weather_df[hist_weather_df['district_no'].isin(target_districts)]

        hist_weather_df['year'] = hist_weather_df['date'].dt.year
        if 'tmean' not in hist_weather_df.columns:
            hist_weather_df['tmean'] = (hist_weather_df['tmin'] + hist_weather_df['tmax']) / 2
        if 'precip' in hist_weather_df.columns and 'prec' not in hist_weather_df.columns:
            hist_weather_df.rename(columns={'precip': 'prec'}, inplace=True)
        if 'srad' in hist_weather_df.columns and 'rad' not in hist_weather_df.columns:
            hist_weather_df.rename(columns={'srad': 'rad'}, inplace=True)

        # Load Satellite
        df_sat_raw = pd.read_csv(SATELLITE_FEATURES_PATH, dtype={'district_no': str})
        # --- OPTIMIZATION: Filter Satellite Early ---
        if target_districts:
            df_sat_raw = df_sat_raw[df_sat_raw['district_no'].isin(target_districts)]

        df_satellite = _infer_missing_ndvi_anomaly(df_sat_raw, hist_weather_df)

    except Exception as e:
        logging.error(f"Initial Data Load Failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Calculate ERA5 WAV (Passed filtered gdf, so it runs fast)
    df_wav = _precalculate_wav_from_era5(gdf_districts, static_df)

    # 3. Calculate Sowing Dates
    logging.info("Calculating Trafficability-Constrained Sowing Dates...")
    sowing_manager = DynamicSowingManager()
    grouped = hist_weather_df.groupby(['year', 'district_no'])

    sowing_dates = Parallel(n_jobs=-1)(
        delayed(process_district_year)(name, group, sowing_manager)
        for name, group in tqdm(grouped)
    )
    df_sowing = pd.DataFrame(sowing_dates)

    # 4. Merge
    df_final = pd.merge(df_sowing, df_wav, on=['year', 'district_no'])
    df_final = pd.merge(df_final, df_satellite, on=['year', 'district_no'], how='left')

    # 5. Adjust WAV
    df_final['winter_cropland_ndvi_anomaly'] = df_final['winter_cropland_ndvi_anomaly'].fillna(0)
    df_final['WAV'] = df_final['WAV'] + (df_final['winter_cropland_ndvi_anomaly'] * NDVI_ANOMALY_SENSITIVITY)
    df_final['WAV'] = df_final['WAV'].clip(lower=0)

    # 6. Constants
    df_final['TDWI'] = cp_variety.get('TDWI', [100.0])[0]
    df_final['RDI'] = cp_variety.get('RDI', [10.0])[0]
    df_final['CROP_END_DATE'] = CONFIG['AGROMANAGEMENT']['CROP_END_DATE']
    df_final['DVSEND'] = cp_variety.get('DVSEND', [2.0])[0]

    # 7. Save
    output_path = config.PROCESSED_DATA_DIR / 'InitialConditions.csv'
    cols = ['district_no', 'year', 'sowing_date', 'WAV', 'TDWI', 'RDI', 'CROP_END_DATE', 'DVSEND']
    df_final[cols].to_csv(output_path, index=False)

    logging.info(f"✓ InitialConditions.csv generated. {len(df_final)} rows.")


if __name__ == "__main__":
    main()