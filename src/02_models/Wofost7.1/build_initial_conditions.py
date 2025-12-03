# File: src/02_models/Wofost7.1/build_initial_conditions.py
# REFACTORED (v12.2): Fixed NameError
# Restores NDVI_ANOMALY_SENSITIVITY constant.

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

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG
ERA5_SOIL_DATA_DIR = Path("data/01_raw/era5_land_monthly_soil")
SATELLITE_FEATURES_PATH = "data/03_processed/satellite_features_districts_2001-2024.csv"
FORECAST_PARTS_DIR = config.PROCESSED_DATA_DIR / 'forecast_weather_parts'

# Sensitivity Config
SOWING_SHIFT_FACTOR = 5.0  # Days earlier per 1°C winter warmth
NDVI_ANOMALY_SENSITIVITY = 5.0  # WAV adjustment scalar


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
        if 'prec' not in df.columns and 'precip' in df.columns:
            df.rename(columns={'precip': 'prec'}, inplace=True)

        if 'tmean' not in df.columns:
            if 'tmin' in df.columns and 'tmax' in df.columns:
                df['tmean'] = (df['tmin'] + df['tmax']) / 2.0
            else:
                return datetime.date(2000, 4, 15)

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
    logging.info("--- Pre-calculating WAV (Using ERA5 Soil Memory) ---")
    era5_files = sorted(list(ERA5_SOIL_DATA_DIR.glob("*_FEBRUARY.nc")))

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
                    if 'swvl2' in rds:
                        target_var = 'swvl2'
                    elif 'swvl1' in rds:
                        target_var = 'swvl1'
                    else:
                        continue

                    year_data = []
                    for _, row in gdf_districts.iterrows():
                        try:
                            val = rds[target_var].sel(longitude=row['longitude'], latitude=row['latitude'],
                                                      method='nearest').values.item()
                        except:
                            val = 0.3
                        year_data.append({'district_no': row['district_no'], 'era5_volumetric': val})

                    df_year = pd.DataFrame(year_data)
                    df_year['year'] = year
                    all_years_results.append(df_year)

        except Exception as e:
            logging.error(f"Failure processing {nc_file.name}: {e}")
            continue

    if not all_years_results:
        logging.warning("No ERA5 Soil Data found. Using default Field Capacity.")
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
        current_theta = np.clip(row['era5_volumetric'], row['SMW'], row['SMFCF'])
        awc = current_theta - row['SMW']
        wav = awc * row['RDMSOL']
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

        # Winter Adjustment
        shift = 0
        if winter_anomalies is not None:
            try:
                row = winter_anomalies.loc[(district_no, year)]
                anomaly = row['temp_anomaly']
                shift = -1 * anomaly * SOWING_SHIFT_FACTOR
            except KeyError:
                pass

        final_doy = median_doy + int(shift)
        final_doy = max(60, min(130, final_doy))

        start_of_year = datetime.date(year, 1, 1)
        final_sowing_date = start_of_year + datetime.timedelta(days=final_doy - 1)

        return {'year': year, 'district_no': district_no, 'sowing_date': final_sowing_date}

    except Exception as e:
        logging.error(f"Error processing forecast sowing: {e}")
        return None


def main():
    logging.info("--- Building InitialConditions.csv (WAV + Smart Sowing + Fixed) ---")

    try:
        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            cp_variety = yaml.safe_load(f)['CropParameters']['Varieties']['Sugarbeet_601']

        static_df = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_SOIL_FEATURES'], dtype={'district_no': str})
        if static_df['RDMSOL'].mean() < 10: static_df['RDMSOL'] *= 100

        gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH).rename(columns={'id': 'district_no'})
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

    df_wav = _precalculate_wav_from_era5(gdf_districts, static_df)

    logging.info("Calculating Smart Sowing Dates...")
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

    df_final = pd.merge(df_sowing, df_wav, on=['year', 'district_no'])
    df_final = pd.merge(df_final, df_satellite, on=['year', 'district_no'], how='left')

    df_final['winter_cropland_ndvi_anomaly'] = df_final['winter_cropland_ndvi_anomaly'].fillna(0)
    df_final['WAV'] = df_final['WAV'] + (df_final['winter_cropland_ndvi_anomaly'] * NDVI_ANOMALY_SENSITIVITY)
    df_final['WAV'] = df_final['WAV'].clip(lower=0)

    df_final['TDWI'] = cp_variety.get('TDWI', [100.0])[0]
    df_final['RDI'] = cp_variety.get('RDI', [10.0])[0]
    df_final['CROP_END_DATE'] = CONFIG['AGROMANAGEMENT']['CROP_END_DATE']
    df_final['DVSEND'] = cp_variety.get('DVSEND', [2.0])[0]

    output_path = config.PROCESSED_DATA_DIR / 'InitialConditions.csv'
    cols = ['district_no', 'year', 'sowing_date', 'WAV', 'TDWI', 'RDI', 'CROP_END_DATE', 'DVSEND']
    df_final[cols].to_csv(output_path, index=False)

    logging.info(f"✓ InitialConditions.csv generated successfully.")


if __name__ == "__main__":
    main()