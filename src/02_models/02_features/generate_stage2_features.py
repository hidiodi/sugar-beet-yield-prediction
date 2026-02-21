import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib
data_config = importlib.import_module("src.01_data.config")
models_config = importlib.import_module("src.02_models.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

STAGE1_PATH = models_config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
DAILY_WEATHER_DIR = data_config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']['DAILY_WEATHER_DIR']
OUTPUT_PATH = global_config.DATA_DIR / '05_model_input/stage2_refined_features.csv'


def calculate_weather_proxies(df_stage1):
    """
    Calculates pest and disease proxies from daily weather data.
    """
    logger.info("--- Calculating Weather-Based Proxies (Pests & Disease) ---")

    valid_districts = set(df_stage1['district_no'].unique())
    proxy_results = []

    all_weather_files = list(DAILY_WEATHER_DIR.glob("*.csv"))

    # Use lambda to avoid error if 'prec' exists but 'precip' does not (or vice versa)
    cols_check = lambda x: x in ['district_no', 'date', 'tmin', 'tmax', 'precip', 'prec']

    for f in tqdm(all_weather_files, desc="Processing Daily Weather"):
        try:
            df_daily = pd.read_csv(f, usecols=cols_check)

            # Standardization
            if 'prec' in df_daily.columns:
                df_daily.rename(columns={'prec': 'precip'}, inplace=True)

            # Check essential columns
            req_cols = ['district_no', 'date', 'tmin', 'tmax']
            if not all(c in df_daily.columns for c in req_cols):
                continue

            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily['year'] = df_daily['date'].dt.year
            df_daily['month'] = df_daily['date'].dt.month
            df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)

            # Filter efficiently
            df_daily = df_daily[df_daily['district_no'].isin(valid_districts)]
            if df_daily.empty: continue

            # Ensure precip exists for calculation
            if 'precip' not in df_daily.columns:
                df_daily['precip'] = 0.0

            # Aggregation logic
            for (district, year), group in df_daily.groupby(['district_no', 'year']):
                # 1. Pest Pressure: Mild Winter Days (Jan/Feb Tmin > 0)
                winter_mask = group['month'].isin([1, 2])
                mild_days = group[winter_mask & (group['tmin'] > 0)].shape[0]
                proxy_results.append({
                    'district_no': district,
                    'year': year,
                    'mild_winter_days': mild_days,
                })

        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")
            continue

    return pd.DataFrame(proxy_results)


def main():
    logger.info("--- Starting Stage 2 Feature Builder ---")

    if not STAGE1_PATH.exists():
        logger.error(f"Stage 1 features not found at {STAGE1_PATH}")
        return

    df = pd.read_csv(STAGE1_PATH)
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)

    # 1. Weather Proxies
    if DAILY_WEATHER_DIR.exists():
        df_proxies = calculate_weather_proxies(df)
        if not df_proxies.empty:
            df_proxies['district_no'] = df_proxies['district_no'].astype(str).str.zfill(5)
            df = pd.merge(df, df_proxies, on=['district_no', 'year'], how='left')
        else:
            logger.warning("No weather proxies calculated (check logs). Filling with 0.")
    else:
        logger.warning("Daily weather directory not found. Skipping weather proxies.")

    # 2. Vegetation Vigor Index
    if 'winter_cropland_ndvi_anomaly' in df.columns and 'spring_precip_anomaly_forecast' in df.columns:
        df['VegetationVigorIndex'] = (df['winter_cropland_ndvi_anomaly'] +
                                      0.5 * df['spring_precip_anomaly_forecast'])
    else:
        df['VegetationVigorIndex'] = 0

    # 3. Root Zone Water Depletion
    if 'effective_winter_water' in df.columns and 'summer_days_tmax_gt_30c' in df.columns:
        # Avoid division by zero
        supply = df['effective_winter_water'].clip(lower=1.0)
        demand = df['summer_days_tmax_gt_30c'].clip(lower=0.0) + 1.0
        df['RootZoneDepletion'] = demand / supply
    else:
        df['RootZoneDepletion'] = 0

    # Save ONLY New Features to avoid redundancy
    cols_to_keep = ['year', 'district_no', 'mild_winter_days', 'VegetationVigorIndex',
                    'RootZoneDepletion']
    df_final = df[cols_to_keep]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Stage 2 Features saved to {OUTPUT_PATH}")
    logger.info(f"Shape: {df_final.shape}")


if __name__ == "__main__":
    main()