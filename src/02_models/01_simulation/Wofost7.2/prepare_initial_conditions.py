# File: src/02_models/Wofost7.1/build_initial_conditions.py
# REFACTORED (v13.0): Physics-Informed Sowing & Soil Init
# - Implements DWD Soil Temp (5cm) proxy via Thermal Inertia
# - Implements DWD Wind Constraint (>5 m/s forbidden)
# - Fixes WAV calculation to respect V6 Soil Physics (SMW/SMFCF)

import datetime
import pandas as pd
import yaml
from pathlib import Path
import sys
import logging
import geopandas as gpd
import xarray as xr
import re
import numpy as np
import tempfile
import zipfile
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config as global_config
import importlib
models_config = importlib.import_module("src.02_models.config")

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = models_config.WOFOST_CONFIG
ERA5_SOIL_DATA_DIR = Path("data/01_raw/era5_land_monthly_soil")
SATELLITE_FEATURES_PATH = "data/03_processed/satellite_features_districts_2001-2024.csv"
FORECAST_PARTS_DIR = global_config.PROCESSED_DATA_DIR / 'forecast_weather_parts'

# Sensitivity Config
# Note: We reduced Sowing Shift Factor because we now model soil temp directly.
SOWING_SHIFT_FACTOR = 2.0
NDVI_ANOMALY_SENSITIVITY = 5.0  # WAV adjustment scalar


def _estimate_soil_temperature(df_weather):
    """
    Approximates 5cm Soil Temperature using Air Temperature Thermal Inertia.
    Soil warms/cools slower than air.
    Uses a 5-day Exponential Moving Average (EMA) of Tmean.
    """
    if 'tmean' not in df_weather.columns:
        df_weather['tmean'] = (df_weather['tmin'] + df_weather['tmax']) / 2.0

    # 5-day span roughly approximates the thermal lag of topsoil (5-10cm)
    return df_weather['tmean'].ewm(span=5, adjust=False).mean()


class DynamicSowingManager:
    def __init__(self,
                 sowing_window_start_month=3, sowing_window_start_day=10,
                 sowing_window_end_month=4, sowing_window_end_day=30,
                 soil_temp_threshold_c=6.0,  # DWD: Sugarbeets germ at 6-8°C
                 precip_trafficability_limit_mm=5.0,
                 wind_limit_ms=5.0,  # DWD: Legal limit for spraying/sowing
                 trafficability_window_days=3):

        self.start_month = sowing_window_start_month
        self.start_day = sowing_window_start_day
        self.end_month = sowing_window_end_month
        self.end_day = sowing_window_end_day
        self.soil_temp_threshold = soil_temp_threshold_c
        self.precip_limit = precip_trafficability_limit_mm
        self.wind_limit = wind_limit_ms
        self.precip_window = trafficability_window_days

    def find_sowing_date(self, weather_df_year: pd.DataFrame) -> datetime.date:
        df = weather_df_year.copy()

        # Standardize Columns
        if 'prec' not in df.columns and 'precip' in df.columns:
            df.rename(columns={'precip': 'prec'}, inplace=True)
        if 'wind' not in df.columns and 'wind_speed_mean' in df.columns:
            df.rename(columns={'wind_speed_mean': 'wind'}, inplace=True)

        # Fill missing wind with low value (assume calm if unknown)
        if 'wind' not in df.columns:
            df['wind'] = 2.0

        # --- 1. Calculate Physical Variables ---
        df = df.sort_values('date')

        # A. Soil Temperature Proxy (The "Real" Trigger)
        df['soil_temp_5cm'] = _estimate_soil_temperature(df)

        # B. Trafficability (Mud Check)
        df['precip_roll_sum'] = df['prec'].rolling(window=self.precip_window, min_periods=1).sum()

        # C. Hard Frost Check (Soil Inertia - Frozen Ground)
        # If tmin < -2, ground is likely frozen surface, regardless of mean
        df['is_frozen'] = (df['tmin'] < -2.0)

        try:
            target_year = df['date'].dt.year.mode()[0]
            # Window: March 15 to April 30
            window_start = datetime.date(target_year, 3, 15)
            window_end = datetime.date(target_year, 4, 30)
        except (IndexError, ValueError):
            return datetime.date(2000, 4, 15)

        # Filter for the window
        mask_window = (df['date'].dt.date >= window_start) & (df['date'].dt.date <= window_end)
        df_window = df.loc[mask_window].copy()

        if df_window.empty:
            return window_end

        # --- 2. Iterate for Agronomic Suitability ---
        for idx, row in df_window.iterrows():
            current_date = row['date']

            # RULE 1: Soil Temperature (Must be > 6°C)
            # We use the calculated proxy, not air temp
            if row['soil_temp_5cm'] < self.soil_temp_threshold:
                continue

            # RULE 2: Soil Trafficability (Too wet to drive)
            if row['precip_roll_sum'] > self.precip_limit:
                continue

            # RULE 3: Wind Speed (Legal/Safety Constraint)
            # DWD: Avoid operations > 5 m/s
            if row['wind'] > self.wind_limit:
                continue

            # RULE 4: Frozen Ground Lock
            # Check last 3 days for hard frost
            lookback_start = current_date - datetime.timedelta(days=3)
            recent_frost = df[(df['date'] >= lookback_start) & (df['date'] < current_date)]['is_frozen'].any()
            if recent_frost:
                continue

                # If all pass, we sow.
            return current_date.date()

        # Fallback: Late Sowing (End of Window)
        return window_end


def _precalculate_wav_from_era5(gdf_districts, static_df):
    logging.info("--- Pre-calculating WAV (Using ERA5 & V6 Soil Physics) ---")
    era5_files = sorted(list(ERA5_SOIL_DATA_DIR.glob("*_FEBRUARY.nc")))

    # We need SMW (Wilting Point) and SMFCF (Field Capacity) from Static Data
    # to correctly interpret the ERA5 Volumetric Water Content
    static_ref = static_df[['district_no', 'SMW', 'SMFCF', 'RDMSOL']].copy()
    all_years_results = []

    pbar = tqdm(era5_files, desc="Scanning ERA5 Soil Moisture")
    for nc_file in pbar:
        year_match = re.search(r'_(\d{4})_FEBRUARY', nc_file.name)
        if not year_match: continue
        year = int(year_match.group(1))

        if CONFIG['START_YEAR'] and year < CONFIG['START_YEAR']: continue
        if CONFIG['END_YEAR'] and year > CONFIG['END_YEAR']: continue

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
                    # 'swvl1' is usually 0-7cm, 'swvl2' is 7-28cm.
                    # We prefer layer 2 for rooting zone init, or average.
                    # Using swvl2 as it's more representative of the seedbed reservoir.
                    if 'swvl2' in rds:
                        target_var = 'swvl2'
                    elif 'swvl1' in rds:
                        target_var = 'swvl1'
                    else:
                        continue

                    year_data = []
                    for _, row in gdf_districts.iterrows():
                        try:
                            # Extract Volumetric Water Content (m3/m3)
                            val = rds[target_var].sel(longitude=row['longitude'], latitude=row['latitude'],
                                                      method='nearest').values.item()
                        except:
                            val = 0.3  # Fallback
                        year_data.append({'district_no': row['district_no'], 'era5_volumetric': val})

                    df_year = pd.DataFrame(year_data)
                    df_year['year'] = year
                    all_years_results.append(df_year)

        except Exception as e:
            logging.error(f"Failure processing {nc_file.name}: {e}")
            continue

    if not all_years_results:
        logging.warning("No ERA5 Soil Data found. Defaulting to Field Capacity.")
        # Default: Start at Field Capacity (WAV = Max)
        static_ref['WAV'] = (static_ref['SMFCF'] - static_ref['SMW']) * static_ref['RDMSOL']
        years = range(CONFIG['START_YEAR'] or 1980, (CONFIG['END_YEAR'] or 2024) + 1)
        dfs = []
        for y in years:
            d = static_ref[['district_no', 'WAV']].copy()
            d['year'] = y
            dfs.append(d)
        return pd.concat(dfs)

    full_swvl_df = pd.concat(all_years_results, ignore_index=True)
    final_df = pd.merge(full_swvl_df, static_ref, on='district_no')

    def calculate_wav(row):
        # 1. Get raw volumetric moisture from ERA5
        theta_current = row['era5_volumetric']

        # 2. Get Soil Limits (The "Bucket" Size)
        theta_wp = row['SMW']  # Wilting Point
        theta_fc = row['SMFCF']  # Field Capacity

        # 3. Calculate Available Water Content (Volumetric)
        # Cannot be less than 0 (below wilting point)
        # Cannot be more than Field Capacity (technically it can be saturated,
        # but WOFOST init usually caps usable water at FC)

        theta_available = np.clip(theta_current, theta_wp, theta_fc) - theta_wp

        # 4. Convert to Total Water Amount (cm) based on Root Depth
        # WAV = Volumetric_Available * Root_Depth
        wav = theta_available * row['RDMSOL']

        return max(0.0, wav)

    final_df['WAV'] = final_df.apply(calculate_wav, axis=1)
    logging.info(f"Recalculated WAV from ERA5. Mean: {final_df['WAV'].mean():.2f} cm")
    return final_df[['year', 'district_no', 'WAV']]


def _infer_missing_ndvi_anomaly(df_satellite: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    logging.info("--- Inferring missing pre-satellite NDVI anomalies ---")

    # Check for required columns
    req_cols = ['tmean', 'prec', 'rad']
    if not all(col in df_weather.columns for col in req_cols):
        logging.error(f"Missing columns for NDVI inference. Available: {df_weather.columns}")
        return df_satellite[['district_no', 'year', 'winter_cropland_ndvi_anomaly']].fillna(0)

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

    if len(df_train) < 10:
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


def process_forecast_sowing_date(file_path, sowing_manager, winter_anomalies):
    try:
        match = re.search(r'forecast_(\d+)_(\d{4})\.parquet', file_path.name)
        if not match: return None
        district_no = match.group(1)
        year = int(match.group(2))

        df_fc = pd.read_parquet(file_path)
        if df_fc.empty: return None

        member_sowing_dates = []
        for member_id, group in df_fc.groupby('member'):
            sowing_date = sowing_manager.find_sowing_date(group)
            member_sowing_dates.append(sowing_date)

        if not member_sowing_dates: return None

        doys = [d.timetuple().tm_yday for d in member_sowing_dates]
        median_doy = int(np.median(doys))

        # Winter Adjustment (Shifted)
        # If winter was super warm, we might assume earlier sowing capability,
        # BUT our dynamic manager now handles soil temp directly.
        # We keep this as a minor biological "readiness" shift, but reduced factor.
        shift = 0
        if winter_anomalies is not None:
            try:
                row = winter_anomalies.loc[(district_no, year)]
                anomaly = row['temp_anomaly']
                shift = -1 * anomaly * SOWING_SHIFT_FACTOR
            except KeyError:
                pass

        final_doy = median_doy + int(shift)
        final_doy = max(60, min(130, final_doy))  # DOY 60 = March 1, 130 = May 10

        start_of_year = datetime.date(year, 1, 1)
        final_sowing_date = start_of_year + datetime.timedelta(days=final_doy - 1)

        return {'year': year, 'district_no': district_no, 'sowing_date': final_sowing_date}

    except Exception as e:
        logging.error(f"Error processing forecast sowing: {e}")
        return None


def main():
    logging.info("--- Building InitialConditions.csv (Physics-Informed Sowing & WAV) ---")

    try:
        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            cp_variety = yaml.safe_load(f)['CropParameters']['Varieties']['Sugarbeet_601']

        static_df = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_SOIL_FEATURES'], dtype={'district_no': str})
        if static_df['RDMSOL'].mean() < 10: static_df['RDMSOL'] *= 100

        data_config = importlib.import_module("src.01_data.config")
        gdf_districts = gpd.read_file(data_config.DISTRICTS_GEOJSON_PATH).rename(columns={'id': 'district_no'})
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)

        limit = CONFIG.get('DISTRICT_LIMIT')
        target_districts = None
        if limit:
            logging.warning(f"!!! TEST MODE: Limiting to top {limit} districts !!!")
            target_districts = sorted(gdf_districts['district_no'].unique())[:limit]
            gdf_districts = gdf_districts[gdf_districts['district_no'].isin(target_districts)]
            static_df = static_df[static_df['district_no'].isin(target_districts)]

        weather_path = Path(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'])
        weather_files = list(weather_path.glob("*.csv"))

        # Load History for NDVI imputation
        hist_weather_df = pd.concat(
            (pd.read_csv(f, parse_dates=['date'], dtype={'district_no': str}) for f in
             tqdm(weather_files, desc="Loading History")),
            ignore_index=True
        )
        if target_districts:
            hist_weather_df = hist_weather_df[hist_weather_df['district_no'].isin(target_districts)]

        hist_weather_df['year'] = hist_weather_df['date'].dt.year
        if 'precip' in hist_weather_df.columns: hist_weather_df.rename(columns={'precip': 'prec'}, inplace=True)
        if 'srad' in hist_weather_df.columns: hist_weather_df.rename(columns={'srad': 'rad'}, inplace=True)
        if 'tmean' not in hist_weather_df.columns:
            hist_weather_df['tmean'] = (hist_weather_df['tmin'] + hist_weather_df['tmax']) / 2

        logging.info("Calculating Winter Temperature Anomalies...")
        winter_df = hist_weather_df[hist_weather_df['date'].dt.month.isin([1, 2])]
        climatology = winter_df.groupby('district_no')['tmean'].mean()
        yearly_winter = winter_df.groupby(['district_no', 'year'])['tmean'].mean().reset_index()
        yearly_winter = yearly_winter.join(climatology, on='district_no', rsuffix='_clim')
        yearly_winter['temp_anomaly'] = yearly_winter['tmean'] - yearly_winter['tmean_clim']
        winter_anomalies = yearly_winter.set_index(['district_no', 'year'])[['temp_anomaly']]

        df_sat_raw = pd.read_csv(SATELLITE_FEATURES_PATH, dtype={'district_no': str})
        if target_districts:
            df_sat_raw = df_sat_raw[df_sat_raw['district_no'].isin(target_districts)]
        df_satellite = _infer_missing_ndvi_anomaly(df_sat_raw, hist_weather_df)

    except Exception as e:
        logging.error(f"Initial Data Load Failed: {e}", exc_info=True)
        sys.exit(1)

    # 1. WAV Calculation (Corrected Physics)
    df_wav = _precalculate_wav_from_era5(gdf_districts, static_df)

    # 2. Sowing Date Calculation (Corrected Physics)
    logging.info("Calculating Physics-Informed Sowing Dates...")
    if not FORECAST_PARTS_DIR.exists():
        logging.error("Forecast parts not found.")
        sys.exit(1)

    forecast_files = list(FORECAST_PARTS_DIR.glob("forecast_*.parquet"))
    if target_districts:
        forecast_files = [f for f in forecast_files if any(d in f.name for d in target_districts)]

    sowing_manager = DynamicSowingManager()
    sowing_results = Parallel(n_jobs=-1)(
        delayed(process_forecast_sowing_date)(f, sowing_manager, winter_anomalies)
        for f in tqdm(forecast_files, desc="Processing Forecast Sowing")
    )

    sowing_results = [res for res in sowing_results if res is not None]
    df_sowing = pd.DataFrame(sowing_results)

    # 3. Merge & Save
    df_final = pd.merge(df_sowing, df_wav, on=['year', 'district_no'])
    df_final = pd.merge(df_final, df_satellite, on=['year', 'district_no'], how='left')

    df_final['winter_cropland_ndvi_anomaly'] = df_final['winter_cropland_ndvi_anomaly'].fillna(0)

    # Adjust WAV slightly by NDVI (Vegetation Memory), but ensure it stays within physics
    # Note: NDVI sensitivity might push WAV above SMFCF or below 0.
    # We apply clip(lower=0) but we also trust the satellite signal to imply 'wetter/drier than ERA5 thought'
    df_final['WAV'] = df_final['WAV'] + (df_final['winter_cropland_ndvi_anomaly'] * NDVI_ANOMALY_SENSITIVITY)
    df_final['WAV'] = df_final['WAV'].clip(lower=0)

    # Add Crop Constants
    df_final['TDWI'] = cp_variety.get('TDWI', [100.0])[0]
    df_final['RDI'] = cp_variety.get('RDI', [10.0])[0]
    df_final['CROP_END_DATE'] = CONFIG['AGROMANAGEMENT']['CROP_END_DATE']
    df_final['DVSEND'] = cp_variety.get('DVSEND', [2.0])[0]

    output_path = global_config.PROCESSED_DATA_DIR / 'InitialConditions.csv'
    cols = ['district_no', 'year', 'sowing_date', 'WAV', 'TDWI', 'RDI', 'CROP_END_DATE', 'DVSEND']
    df_final[cols].to_csv(output_path, index=False)

    logging.info(f"✓ InitialConditions.csv generated successfully.")


if __name__ == "__main__":
    main()