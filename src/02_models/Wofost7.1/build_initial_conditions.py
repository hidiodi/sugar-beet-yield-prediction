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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG
ERA5_SOIL_DATA_DIR = Path("data/01_raw/era5_land_monthly_soil")
# Ensure this path points to the time-series satellite file (2001-2024)
SATELLITE_FEATURES_PATH = "data/03_processed/satellite_features_districts_2001-2024.csv"
NDVI_ANOMALY_SENSITIVITY = 5.0  # Defines how much WAV (in cm) changes per unit of NDVI anomaly


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
            return datetime.date(2000, self.start_month, self.start_day)  # Fallback

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
                    with zipfile.ZipFile(nc_file, 'r') as zf:
                        zf.extractall(temp_dir)
                    extracted = list(Path(temp_dir).glob('*.nc'))
                    if not extracted: continue
                    actual_nc_path = extracted[0]

                with xr.open_dataset(actual_nc_path, engine='netcdf4') as rds:
                    # Optimization: Ensure spatial dims are set once
                    da = rds['swvl1'].squeeze(drop=True).rio.set_spatial_dims(x_dim="longitude",
                                                                              y_dim="latitude").rio.write_crs(
                        "EPSG:4326")

                    year_results = []
                    for _, district in gdf_districts.iterrows():
                        try:
                            # Note: Loop-and-clip is slow but robust for irregular district shapes.
                            # If speed becomes critical, implement rasterstats or vector-masking.
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
        logging.error("FATAL: No soil moisture data could be processed.");
        sys.exit(1)

    full_swvl1_df = pd.concat(all_years_results, ignore_index=True)
    final_df = pd.merge(full_swvl1_df, static_df[['district_no', 'SMW', 'RDMSOL']], on='district_no')

    # Physics: WAV = (Soil Moisture - Wilting Point) * Root Zone Depth
    final_df['WAV'] = (final_df['mean_feb_swvl1'] - final_df['SMW']) * final_df['RDMSOL']
    final_df['WAV'] = final_df['WAV'].clip(lower=0)
    return final_df[['year', 'district_no', 'WAV']]


def _infer_missing_ndvi_anomaly(df_satellite: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Infers missing NDVI anomalies for years before satellite data is available
    by training a regression model on winter weather data.
    """
    logging.info("--- Inferring missing pre-satellite NDVI anomalies ---")

    # Define winter period: Dec (Y-1), Jan (Y), Feb (Y) -> Assigned to Year Y
    df_weather['winter_year'] = df_weather['year']
    df_weather.loc[df_weather['date'].dt.month == 12, 'winter_year'] += 1

    # Filter for winter months
    winter_weather = df_weather[df_weather['date'].dt.month.isin([12, 1, 2])].copy()

    # Aggregate weather data to winter-level stats per district
    winter_agg = winter_weather.groupby(['district_no', 'winter_year']).agg(
        winter_temp_mean=('tmean', 'mean'),
        winter_precip_sum=('prec', 'sum'),
        winter_rad_sum=('rad', 'sum')
    ).reset_index().rename(columns={'winter_year': 'year'})

    # Calculate anomalies (Deviation from district long-term mean)
    for col in ['winter_temp_mean', 'winter_precip_sum', 'winter_rad_sum']:
        mean_map = winter_agg.groupby('district_no')[col].transform('mean')
        winter_agg[f'{col}_anomaly'] = winter_agg[col] - mean_map

    # Merge with satellite data
    df_full = pd.merge(df_satellite, winter_agg, on=['district_no', 'year'], how='left')

    # Identify training data (where NDVI exists) and prediction data (where it's missing)
    df_train = df_full.dropna(subset=['winter_cropland_ndvi_anomaly', 'winter_temp_mean_anomaly'])
    # FIX: Use .copy() to avoid SettingWithCopyWarning
    df_predict = df_full[df_full['winter_cropland_ndvi_anomaly'].isnull()].copy()

    if df_predict.empty:
        logging.info("No missing NDVI data to infer. Returning original data.")
        return df_satellite[['district_no', 'year', 'winter_cropland_ndvi_anomaly']]

    # Define features and target
    features = ['winter_temp_mean_anomaly', 'winter_precip_sum_anomaly', 'winter_rad_sum_anomaly']
    target = 'winter_cropland_ndvi_anomaly'

    X_train = df_train[features]
    y_train = df_train[target]
    X_predict = df_predict[features].dropna()

    if X_predict.empty:
        logging.warning("No valid weather data for the period requiring NDVI inference. Filling with 0.")
        df_full[target] = df_full[target].fillna(0)
        return df_full[['district_no', 'year', 'winter_cropland_ndvi_anomaly']]

    # Train Model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Log Performance
    y_pred_train = model.predict(X_train)
    r2 = r2_score(y_train, y_pred_train)
    logging.info(f"Weather-NDVI inference model trained. R-squared on training data: {r2:.3f}")

    # Predict
    predicted_anomalies = model.predict(X_predict)
    df_predict.loc[X_predict.index, 'inferred_ndvi_anomaly'] = predicted_anomalies

    # Merge predictions back
    df_full = pd.merge(df_full, df_predict[['district_no', 'year', 'inferred_ndvi_anomaly']],
                       on=['district_no', 'year'], how='left')
    df_full[target] = df_full[target].fillna(df_full['inferred_ndvi_anomaly'])
    df_full[target] = df_full[target].fillna(0)  # Final cleanup

    return df_full[['district_no', 'year', 'winter_cropland_ndvi_anomaly']]


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

        # --- Weather Data Loading ---
        weather_path = Path(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'])
        if not weather_path.is_dir():
            logging.error(f"FATAL: Weather path is not a directory: {weather_path}");
            sys.exit(1)

        logging.info(f"Loading weather data from directory: {weather_path}")
        weather_files = list(weather_path.glob("*.csv"))
        if not weather_files:
            logging.error("FATAL: No CSV files found in weather directory.");
            sys.exit(1)

        historical_weather_df = pd.concat(
            (pd.read_csv(f, parse_dates=['date'], dtype={'district_no': str}) for f in
             tqdm(weather_files, desc="Loading Weather Files")),
            ignore_index=True
        )

        # --- Weather Column Normalization ---
        # Extract Year
        historical_weather_df['year'] = historical_weather_df['date'].dt.year

        # Create Mean Temp
        if 'tmean' not in historical_weather_df.columns:
            historical_weather_df['tmean'] = (historical_weather_df['tmin'] + historical_weather_df['tmax']) / 2

        # Rename columns to standard names if they differ
        if 'prec' not in historical_weather_df.columns and 'precip' in historical_weather_df.columns:
            historical_weather_df.rename(columns={'precip': 'prec'}, inplace=True)
        if 'rad' not in historical_weather_df.columns and 'srad' in historical_weather_df.columns:
            historical_weather_df.rename(columns={'srad': 'rad'}, inplace=True)

        logging.info(f"Weather columns finalized: {historical_weather_df.columns.tolist()}")

        # --- Satellite Data Preparation ---
        df_satellite_raw = pd.read_csv(SATELLITE_FEATURES_PATH, dtype={'district_no': str})

        # Create master grid of all District-Year combinations
        years = historical_weather_df['year'].unique()
        districts = historical_weather_df['district_no'].unique()
        master_grid = pd.DataFrame([(d, y) for d in districts for y in years], columns=['district_no', 'year'])

        # Merge existing satellite data onto master grid
        df_satellite_full_timeseries = pd.merge(master_grid, df_satellite_raw, on=['district_no', 'year'], how='left')

        # Infer missing years
        df_satellite = _infer_missing_ndvi_anomaly(df_satellite_full_timeseries, historical_weather_df)

    except Exception as e:
        logging.error(f"FATAL: Data loading failed. Error: {e}", exc_info=True);
        sys.exit(1)

    # 1. Pre-calculate all WAV values from ERA5
    df_wav = _precalculate_wav_from_era5(gdf_districts, static_df)

    # 2. Determine dynamic sowing dates in parallel
    sowing_manager = DynamicSowingManager()
    grouped = historical_weather_df.groupby(['year', 'district_no'])

    logging.info("Finding sowing dates in parallel...")
    sowing_dates = Parallel(n_jobs=-1)(
        delayed(process_district_year)(name, group, sowing_manager) for name, group in tqdm(grouped))

    df_sowing = pd.DataFrame(sowing_dates)

    # 3. Merge WAV and sowing dates
    df_merged = pd.merge(df_sowing, df_wav, on=['year', 'district_no'])

    # 4. ENHANCEMENT: Incorporate satellite data
    df_merged = pd.merge(df_merged, df_satellite[['year', 'district_no', 'winter_cropland_ndvi_anomaly']],
                         on=['year', 'district_no'], how='left')

    # Fix FutureWarning: Assign result instead of inplace=True
    df_merged['winter_cropland_ndvi_anomaly'] = df_merged['winter_cropland_ndvi_anomaly'].fillna(0)

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