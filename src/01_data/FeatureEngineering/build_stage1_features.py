# File: src/01_data/FeatureEngineering/build_stage1_features.py
# REFACTORED (v5.2): Restoring Spring Forecasts + Teleconnections
# Description:
#   1. RESTORES ECMWF Spring Forecasts (Temp/Precip).
#   2. KEEPS Summer Forecasts DELETED (Noise reduction).
#   3. ADDS Teleconnections (NAO/ENSO) for Regime Awareness.

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import geopandas as gpd

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.FEATURE_ENGINEERING_CONFIG


def process_antecedent_weather_only(weather_dir: Path, start_year: int, end_year: int, forecast_month: int = 3):
    """Aggregates observed weather up to March 1st."""
    logging.info("--- Processing Antecedent (Observed) Weather ---")
    all_weather_files = [weather_dir / f"historical_daily_weather_era5_{y}.csv" for y in
                         range(start_year - 1, end_year + 1)]
    df_daily = pd.concat([pd.read_csv(f) for f in all_weather_files if f.exists()], ignore_index=True)

    df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['year'] = df_daily['date'].dt.year

    # GDD Calculation
    df_daily['gdd'] = ((df_daily['tmax'] + df_daily['tmin']) / 2 - 5).clip(lower=0)

    # Normals (1981-2010)
    df_normals = df_daily[df_daily['year'].between(1981, 2010)].copy()
    daily_normals = df_normals.groupby(df_normals['date'].dt.dayofyear)['gdd'].mean().rename('gdd_30yr_avg')
    df_daily = pd.merge(df_daily, daily_normals, left_on=df_daily['date'].dt.dayofyear, right_index=True, how='left')
    df_daily['gdd_anomaly'] = df_daily['gdd'] - df_daily['gdd_30yr_avg']

    # Harvest Year Logic
    df_daily['harvest_year'] = df_daily['date'].apply(lambda d: d.year + 1 if d.month >= 10 else d.year)

    # Filter: Strictly BEFORE forecast month
    antecedent_mask = (
            ((df_daily['date'].dt.month < forecast_month) & (df_daily['year'] == df_daily['harvest_year'])) |
            ((df_daily['date'].dt.month >= 10) & (df_daily['year'] == df_daily['harvest_year'] - 1))
    )
    df_antecedent = df_daily[antecedent_mask].copy()

    logging.info(f"Aggregating antecedent features...")
    df_agg = df_antecedent.groupby(['district_no', 'harvest_year']).agg(
        antecedent_precip_sum=('precip', 'sum'),
        antecedent_frost_days=('tmin', lambda x: (x < 0).sum()),
        antecedent_gdd_sum_anomaly=('gdd_anomaly', 'sum')
    ).reset_index().rename(columns={'harvest_year': 'year'})

    return df_agg


def load_and_process_economics(producer_file, input_file):
    """Loads economic data with explicit type conversion."""
    try:
        logging.info("Processing Economic Data...")
        df_prod = pd.read_csv(producer_file)
        df_prod = df_prod[df_prod['ID'] == 'LWPR-132'].copy()
        df_prod = df_prod.melt(id_vars=['ID', 'Description'], var_name='year', value_name='producer_price_index')
        df_prod['year'] = pd.to_numeric(df_prod['year'], errors='coerce')
        df_prod['producer_price_index'] = pd.to_numeric(df_prod['producer_price_index'], errors='coerce')
        df_prod.dropna(subset=['year', 'producer_price_index'], inplace=True)
        df_prod['year'] = df_prod['year'].astype(int)

        df_input = pd.read_csv(input_file)
        IDS = {'LWBM-13': 'fertilizer_price_index', 'LWBM-11': 'seed_price_index'}
        df_input = df_input[df_input['ID'].isin(IDS.keys())].copy()
        df_input = df_input.melt(id_vars=['ID', 'Description'], var_name='period', value_name='val')
        df_input['year'] = pd.to_numeric(df_input['period'].str.split('/').str[1], errors='coerce')
        df_input['val'] = pd.to_numeric(df_input['val'], errors='coerce')
        df_input.dropna(subset=['year', 'val'], inplace=True)
        df_input['year'] = df_input['year'].astype(int)

        df_input = df_input.groupby(['year', 'ID'])['val'].mean().unstack().reset_index()
        df_input.rename(columns=IDS, inplace=True)

        df_econ = pd.merge(df_prod[['year', 'producer_price_index']], df_input, on='year', how='outer')
        return df_econ
    except Exception as e:
        logging.error(f"Economic data load failed: {e}")
        return pd.DataFrame(columns=['year'])


def main():
    logging.info("--- Starting Feature Engineering (Restoring Spring Forecasts) ---")
    paths = CONFIG['FILE_PATHS']
    paths['OUTPUT_DIR'].mkdir(exist_ok=True, parents=True)

    # 1. Load Base Master (Contains Teleconnections & Raw Forecasts)
    df = pd.read_csv(paths['MASTER_DATASET'])
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    df['kreisYield'] = pd.to_numeric(df['yield'], errors='coerce')
    df.dropna(subset=['kreisYield'], inplace=True)

    # Keep Teleconnections if available (NAO/ENSO are in the Master usually)
    # We explicitly select them to ensure they survive
    tele_cols = ['nao_winter_avg', 'enso_mei_winter_avg', 'sca_winter_avg']
    for col in tele_cols:
        if col not in df.columns:
            logging.warning(f"Teleconnection {col} missing from Master. Filling 0.")
            df[col] = 0.0

    # 2. RESTORE SPRING FORECASTS (The Signal)
    # The master dataset already has the raw ECMWF means merged in previous iterations.
    # We just need to make sure we KEEP them and maybe rename them for clarity.
    # If they are not in Master, we load from the intermediate file.

    # Check if forecast cols are in Master
    if 'spring_temp_anomaly_forecast' not in df.columns:
        logging.info("Loading ECMWF Forecasts from source...")
        df_fcst = pd.read_csv(paths['ECMWF_FORECAST_FEATURES_CSV'])
        df_fcst['district_no'] = df_fcst['district_no'].astype(str).str.zfill(5)
        # Aggregation: We want MEAN for Spring (Signal)
        # We deliberately IGNORE Summer (Noise)
        df_fcst_agg = df_fcst.groupby(['year', 'district_no']).agg({
            'spring_temp_anomaly_forecast': 'mean',
            'spring_precip_anomaly_forecast': 'mean'
        }).reset_index()
        df = pd.merge(df, df_fcst_agg, on=['year', 'district_no'], how='left')
    else:
        logging.info("Using existing Spring Forecasts from Master.")

    # 3. Load Statistical Trend & National Lag
    df_trend = pd.read_csv(paths['WALKFORWARD_FORECAST_CSV'])
    df_trend['district_no'] = df_trend['district_no'].astype(str).str.zfill(5)
    df_trend.rename(columns={'final_corrected_forecast': 'stat_trend_forecast'}, inplace=True)
    df = pd.merge(df, df_trend[['year', 'district_no', 'stat_trend_forecast']], on=['year', 'district_no'], how='left')

    nat_yield = df.groupby('year')['kreisYield'].mean().reset_index()
    nat_yield['national_avg_yield_lag1'] = nat_yield['kreisYield'].shift(1)
    df = pd.merge(df, nat_yield[['year', 'national_avg_yield_lag1']], on='year', how='left')

    # 4. Load WOFOST Ensemble (Risk)
    df_wofost = pd.read_csv(paths['WOFOST_ENSEMBLE_CSV'])
    df_wofost['district_no'] = df_wofost['district_no'].astype(str).str.zfill(5)
    DMC = config.WOFOST_CONFIG['CONSTANTS']['DMC_SUGARBEET']
    df_wofost['yield_fresh'] = (df_wofost['yield_water_limited'] / DMC) / 100.0

    df_risk = df_wofost.groupby(['year', 'district_no']).agg(
        wofost_esp_mean=('yield_fresh', 'mean'),
        wofost_esp_std=('yield_fresh', 'std'),
        wofost_esp_p10=('yield_fresh', lambda x: x.quantile(0.10)),
        wofost_esp_p90=('yield_fresh', lambda x: x.quantile(0.90)),
        wofost_water_stress_mean=('cumulative_water_stress', 'mean')
    ).reset_index()
    df = pd.merge(df, df_risk, on=['year', 'district_no'], how='left')

    # 5. Antecedent Weather (Conflicts Handled)
    df_weather = process_antecedent_weather_only(
        paths['DAILY_WEATHER_DIR'],
        CONFIG['WEATHER_FEATURE_YEAR_START'],
        CONFIG['WEATHER_FEATURE_YEAR_END']
    )
    # Drop stale columns to avoid _x/_y
    cols_to_drop = [c for c in df_weather.columns if c in df.columns and c not in ['year', 'district_no']]
    if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
    df = pd.merge(df, df_weather, on=['year', 'district_no'], how='left')

    # 6. Satellite (Winter Only)
    df_sat = pd.read_csv(paths['SATELLITE_FEATURES_CSV'])
    df_sat['district_no'] = df_sat['district_no'].astype(str).str.zfill(5)
    sat_cols = ['winter_cropland_ndvi_anomaly', 'winter_cropland_snow_cover_days']
    df_sat = df_sat[['year', 'district_no'] + [c for c in sat_cols if c in df_sat.columns]]

    cols_to_drop = [c for c in df_sat.columns if c in df.columns and c not in ['year', 'district_no']]
    if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
    df = pd.merge(df, df_sat, on=['year', 'district_no'], how='left')

    for c in sat_cols:
        if c in df.columns: df[c] = df[c].fillna(0)

    # 7. Economics
    df_econ = load_and_process_economics(paths['PRODUCER_PRICE_CSV'], paths['INPUT_PRICE_CSV'])
    if not df_econ.empty:
        cols_to_drop = [c for c in df_econ.columns if c in df.columns and c != 'year']
        if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
        df = pd.merge(df, df_econ, on='year', how='left')

        for c in ['producer_price_index', 'fertilizer_price_index']:
            if c in df.columns:
                df[f'{c}_lag1'] = df.groupby('district_no')[c].shift(1).ffill().bfill()

    # 8. Feature Engineering (Interactions)
    logging.info("Creating Interaction Terms...")

    df['trend_vs_phys_gap'] = df['wofost_esp_mean'] - df['stat_trend_forecast']
    df['wofost_skew'] = (df['wofost_esp_mean'] - df['wofost_esp_p10']) / (
                df['wofost_esp_p90'] - df['wofost_esp_p10'] + 1e-6)

    # INTERACTION: Spring Forecast x Antecedent State
    # "Warm Spring" acts differently if "Soil is Dry" vs "Soil is Wet"
    if 'spring_temp_anomaly_forecast' in df.columns and 'antecedent_precip_sum' in df.columns:
        df['spring_temp_x_antecedent_rain'] = df['spring_temp_anomaly_forecast'] * df['antecedent_precip_sum']

    # INTERACTION: Teleconnection x Forecast
    # "NAO+" usually implies Wet/Warm Winter/Spring in Germany.
    # If NAO is High AND Spring Forecast is Warm -> Stronger confidence.
    if 'nao_winter_avg' in df.columns and 'spring_temp_anomaly_forecast' in df.columns:
        df['nao_x_spring_temp'] = df['nao_winter_avg'] * df['spring_temp_anomaly_forecast']

    # Cleanup
    df.dropna(subset=['stat_trend_forecast', 'wofost_esp_mean'], inplace=True)
    weather_cols = ['antecedent_precip_sum', 'antecedent_gdd_sum_anomaly']
    for c in weather_cols:
        if c in df.columns: df[c] = df[c].fillna(0)

    output_file = paths['OUTPUT_FILE']
    df.to_csv(output_file, index=False)
    logging.info(f"✓ Feature Engineering Complete. Saved to {output_file}")

if __name__ == '__main__':
    main()