# File: src/01_data/FeatureEngineering/build_stage1_features.py
# REFACTORED (v7.0): Mechanism-Informed Features + Unit Fixes
# Updates:
#   1. Toxicology: Calculates 'prev_autumn_precip' for Herbicide Carryover.
#   2. Vectors: Adds Spatial Flag (Latitude < 52) to localize SBR risk.
#   3. Leaching: Soil Texture * Winter Rain interaction.
#   4. Unit Conversion: WOFOST kg DM -> dt Fresh.

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import glob

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.FEATURE_ENGINEERING_CONFIG


def process_antecedent_weather_only(weather_dir: Path, start_year: int, end_year: int, forecast_month: int = 3):
    """
    Aggregates observed weather with DWD-aligned biological thresholds AND Toxicology history.
    """
    logging.info("--- Processing Antecedent Weather (Bio + Toxicology) ---")

    # Robust file loading (Glob instead of hardcoded names)
    all_weather_files = list(weather_dir.glob("*.csv"))
    if not all_weather_files:
        logging.error(f"No weather files found in {weather_dir}")
        return pd.DataFrame()

    # Load all and filter by year range
    df_list = []
    for f in all_weather_files:
        try:
            temp = pd.read_csv(f)
            if 'date' in temp.columns:
                temp['date'] = pd.to_datetime(temp['date'])
                temp['year'] = temp['date'].dt.year
                # Filter early to save memory
                temp = temp[(temp['year'] >= start_year - 1) & (temp['year'] <= end_year)]
                if not temp.empty:
                    df_list.append(temp)
        except Exception as e:
            continue

    if not df_list:
        return pd.DataFrame()

    df_daily = pd.concat(df_list, ignore_index=True)
    df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)

    # Normalize Columns
    if 'prec' in df_daily.columns and 'precip' not in df_daily.columns:
        df_daily.rename(columns={'prec': 'precip'}, inplace=True)

    df_daily['tavg'] = (df_daily['tmin'] + df_daily['tmax']) / 2.0

    # Harvest Year Logic
    df_daily['harvest_year'] = df_daily['date'].apply(lambda d: d.year + 1 if d.month >= 10 else d.year)

    # --- 1. STANDARD ANTECEDENT (Oct-Feb) ---
    antecedent_mask = (
            ((df_daily['date'].dt.month < forecast_month) & (df_daily['year'] == df_daily['harvest_year'])) |
            ((df_daily['date'].dt.month >= 10) & (df_daily['year'] == df_daily['harvest_year'] - 1))
    )
    df_ant = df_daily[antecedent_mask].copy()

    # Bio-Physical Features
    df_ant['is_sowing_day'] = ((df_ant['tavg'] > 5) & (df_ant['precip'] < 1.0) & (df_ant['date'].dt.month == 2)).astype(
        int)
    df_ant['aphid_gdd'] = (df_ant['tmax'] - 3).clip(lower=0)
    df_ant['deep_frost_hit'] = (df_ant['tmin'] < -5).astype(int)

    df_agg = df_ant.groupby(['district_no', 'harvest_year']).agg(
        antecedent_precip_sum=('precip', 'sum'),
        sowing_potential_days=('is_sowing_day', 'sum'),
        winter_aphid_pressure=('aphid_gdd', 'sum'),
        winter_pest_kill_days=('deep_frost_hit', 'sum')
    ).reset_index().rename(columns={'harvest_year': 'year'})

    # --- 2. TOXICOLOGY (Aug-Nov of PREVIOUS YEAR) ---
    # Logic: Dry Autumn = Herbicide residues (Sulfonylureas) persist -> Root damage next spring.
    tox_mask = (
            (df_daily['year'] == df_daily['harvest_year'] - 1) &
            (df_daily['date'].dt.month.isin([8, 9, 10, 11]))
    )
    df_tox = df_daily[tox_mask].groupby(['district_no', 'harvest_year'])['precip'].sum().reset_index()
    df_tox.rename(columns={'precip': 'prev_autumn_precip', 'harvest_year': 'year'}, inplace=True)

    # Merge Toxicology into Main
    df_final = pd.merge(df_agg, df_tox, on=['district_no', 'year'], how='left')

    # Fill NA for toxicology (if missing, assume normal/wet i.e., 200mm, low risk)
    df_final['prev_autumn_precip'] = df_final['prev_autumn_precip'].fillna(200)

    # Create the Index: Low Rain = High Risk
    df_final['toxic_carryover_index'] = 1000.0 / (df_final['prev_autumn_precip'] + 10.0)

    return df_final


def load_and_process_economics(producer_file, input_file):
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
    logging.info("--- Starting Feature Engineering (v7.0 Mechanism-Informed) ---")
    paths = CONFIG['FILE_PATHS']
    paths['OUTPUT_DIR'].mkdir(exist_ok=True, parents=True)

    # 1. Load Base
    df = pd.read_csv(paths['MASTER_DATASET'])
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    df['kreisYield'] = pd.to_numeric(df['yield'], errors='coerce')
    df.dropna(subset=['kreisYield'], inplace=True)

    # 2. Restore Spring Forecasts
    if 'spring_temp_anomaly_forecast' not in df.columns:
        logging.info("Loading ECMWF Forecasts...")
        df_fcst = pd.read_csv(paths['ECMWF_FORECAST_FEATURES_CSV'])
        df_fcst['district_no'] = df_fcst['district_no'].astype(str).str.zfill(5)
        # Aggregate Spring/Summer means
        df_fcst_agg = df_fcst.groupby(['year', 'district_no']).agg({
            'spring_temp_anomaly_forecast': 'mean',
            'spring_precip_anomaly_forecast': 'mean',
            'summer_temp_anomaly_forecast': 'mean',
            'summer_solar_rad_anomaly_forecast': 'mean'
        }).reset_index()
        df = pd.merge(df, df_fcst_agg, on=['year', 'district_no'], how='left')

    # 3. Teleconnections
    if 'nao_winter_avg' not in df.columns: df['nao_winter_avg'] = 0.0

    # 4. Load Statistical Trend & Lag (Using Final Corrected as Anchor)
    df_trend = pd.read_csv(paths['WALKFORWARD_FORECAST_CSV'])
    df_trend['district_no'] = df_trend['district_no'].astype(str).str.zfill(5)
    # Using 'final_corrected_forecast' (ARIMA optimized) as the baseline anchor
    df_trend.rename(columns={'final_corrected_forecast': 'stat_trend_forecast'}, inplace=True)
    df = pd.merge(df, df_trend[['year', 'district_no', 'stat_trend_forecast']], on=['year', 'district_no'], how='left')

    nat_yield = df.groupby('year')['kreisYield'].mean().reset_index()
    nat_yield['national_avg_yield_lag1'] = nat_yield['kreisYield'].shift(1)
    df = pd.merge(df, nat_yield[['year', 'national_avg_yield_lag1']], on='year', how='left')

    # 5. Load WOFOST Metrics (Including NEW Risk Features)
    logging.info("Loading WOFOST Risk Metrics & Ensemble Cone...")

    # Load Risk Metrics (Anoxia, Sowing, Frost)
    df_wofost = pd.read_csv(paths['WOFOST_METRICS_CSV'])
    df_wofost['district_no'] = df_wofost['district_no'].astype(str).str.zfill(5)
    df_wofost.columns = df_wofost.columns.str.replace(r'[<>]', '', regex=True)

    # Load Ensemble Yields for Cone & Unit Conversion
    df_wofost_yield = pd.read_csv(paths['WOFOST_ENSEMBLE_CSV'])
    df_wofost_yield['district_no'] = df_wofost_yield['district_no'].astype(str).str.zfill(5)

    # --- CRITICAL UNIT CONVERSION (DM -> FRESH) ---
    DMC = config.WOFOST_CONFIG['CONSTANTS']['DMC_SUGARBEET']
    df_wofost_yield['yield_fresh'] = (df_wofost_yield['yield_water_limited'] / DMC) / 100.0

    df_cone = df_wofost_yield.groupby(['year', 'district_no']).agg(
        wofost_esp_mean=('yield_fresh', 'mean'),
        wofost_esp_std=('yield_fresh', 'std'),
        wofost_esp_p10=('yield_fresh', lambda x: x.quantile(0.10)),
        wofost_esp_p90=('yield_fresh', lambda x: x.quantile(0.90)),
        wofost_water_stress_mean=('cumulative_water_stress', 'mean')
    ).reset_index()

    # Merge Yield Cone
    df = pd.merge(df, df_cone, on=['year', 'district_no'], how='left')

    # Merge Risk Metrics
    cols_to_use = [c for c in df_wofost.columns if c not in df.columns or c in ['year', 'district_no']]
    df = pd.merge(df, df_wofost[cols_to_use], on=['year', 'district_no'], how='left')

    # 6. Antecedent Weather (Bio + Tox)
    df_weather = process_antecedent_weather_only(
        paths['DAILY_WEATHER_DIR'],
        CONFIG['WEATHER_FEATURE_YEAR_START'],
        CONFIG['WEATHER_FEATURE_YEAR_END']
    )
    if not df_weather.empty:
        cols_to_drop = [c for c in df_weather.columns if c in df.columns and c not in ['year', 'district_no']]
        if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
        df = pd.merge(df, df_weather, on=['year', 'district_no'], how='left')

    # 7. Satellite
    df_sat = pd.read_csv(paths['SATELLITE_FEATURES_CSV'])
    df_sat['district_no'] = df_sat['district_no'].astype(str).str.zfill(5)
    sat_cols = ['winter_cropland_ndvi_anomaly', 'winter_cropland_snow_cover_days']
    df_sat = df_sat[['year', 'district_no'] + [c for c in sat_cols if c in df_sat.columns]]

    cols_to_drop = [c for c in df_sat.columns if c in df.columns and c not in ['year', 'district_no']]
    if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
    df = pd.merge(df, df_sat, on=['year', 'district_no'], how='left')
    for c in sat_cols:
        if c in df.columns: df[c] = df[c].fillna(0)

    # 8. Economics
    df_econ = load_and_process_economics(paths['PRODUCER_PRICE_CSV'], paths['INPUT_PRICE_CSV'])
    if not df_econ.empty:
        cols_to_drop = [c for c in df_econ.columns if c in df.columns and c != 'year']
        if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
        df = pd.merge(df, df_econ, on='year', how='left')
        for c in ['producer_price_index', 'fertilizer_price_index']:
            if c in df.columns:
                df[f'{c}_lag1'] = df.groupby('district_no')[c].shift(1).ffill().bfill()

    # 9. FINAL FEATURE ENGINEERING (Interactions & Spatial Flags)
    logging.info("Calculating Final Interactions...")

    df['trend_vs_phys_gap'] = df['wofost_esp_mean'] - df['stat_trend_forecast']
    df['wofost_skew'] = (df['wofost_esp_mean'] - df['wofost_esp_p10']) / (
            df['wofost_esp_p90'] - df['wofost_esp_p10'] + 1e-6)

    # A. Spatial Flag for Vectors (Using 'latitude' which is standard)
    if 'latitude' in df.columns:
        df['is_vector_endemic_zone'] = (df['latitude'] < 52.0).astype(int)
        df['vector_pressure_local'] = df['winter_aphid_pressure'] * df['is_vector_endemic_zone']
    elif 'lat' in df.columns:  # Fallback if only 'lat' exists
        df['is_vector_endemic_zone'] = (df['lat'] < 52.0).astype(int)
        df['vector_pressure_local'] = df['winter_aphid_pressure'] * df['is_vector_endemic_zone']
    else:
        df['vector_pressure_local'] = 0

        # B. Nitrogen Leaching
    if 'avg_sand_0_30cm' in df.columns and 'antecedent_precip_sum' in df.columns:
        df['nitrogen_leaching_index'] = df['antecedent_precip_sum'] * df['avg_sand_0_30cm']

    # C. Forecast Interactions
    if 'spring_temp_anomaly_forecast' in df.columns and 'antecedent_precip_sum' in df.columns:
        df['spring_temp_x_antecedent_rain'] = df['spring_temp_anomaly_forecast'] * df['antecedent_precip_sum']
    if 'nao_winter_avg' in df.columns and 'spring_temp_anomaly_forecast' in df.columns:
        df['nao_x_spring_temp'] = df['nao_winter_avg'] * df['spring_temp_anomaly_forecast']

    # Cleanup
    df.dropna(subset=['stat_trend_forecast', 'wofost_esp_mean'], inplace=True)

    # Fill NAs for computed features
    fill_cols = ['antecedent_precip_sum', 'sowing_potential_days', 'winter_aphid_pressure',
                 'winter_pest_kill_days', 'toxic_carryover_index', 'nitrogen_leaching_index']
    for c in fill_cols:
        if c in df.columns: df[c] = df[c].fillna(0)

    output_file = paths['OUTPUT_FILE']
    df.to_csv(output_file, index=False)
    logging.info(f"✓ Feature Engineering Complete. Saved to {output_file}")
    logging.info(f"  New Mechanism Features: toxic_carryover_index, vector_pressure_local, nitrogen_leaching_index")


if __name__ == '__main__':
    main()