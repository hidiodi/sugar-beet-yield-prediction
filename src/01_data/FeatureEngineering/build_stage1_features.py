# File: src/01_data/FeatureEngineering/build_stage1_features.py
# REFACTORED (v13.0): The "Good Setup" Restoration (Physiological Indices + Economics)

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import geopandas as gpd
from tqdm import tqdm

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.FEATURE_ENGINEERING_CONFIG


def create_granular_weather_features(weather_dir: Path, start_year: int, end_year: int):
    """
    Engineers physiological stress indices (CASDI, NMSD, OSAW, ECES).
    """
    logging.info("--- Engineering Physiological Weather Features (CASDI, NMSD, OSAW) ---")

    # Physiological Thresholds
    PARAMS = {
        'TMAX_STRESS_THRESHOLD': 30.0,
        'TMIN_STRESS_THRESHOLD': 17.0,
        'TMAX_OPTIMAL_MIN': 17.0, 'TMAX_OPTIMAL_MAX': 25.0,
        'TMIN_OPTIMAL_MAX': 15.0,
        'PRECIP_DEFICIT_WINDOW': 30, 'PRECIP_DEFICIT_THRESHOLD': 20.0,
        'ECES_EXPONENT': 1.5, 'DTR_SUNNY_DAY_QUANTILE': 0.75
    }

    # Robust Loading
    all_weather_files = list(weather_dir.glob("*.csv"))
    if not all_weather_files:
        logging.error("No weather files found.")
        return pd.DataFrame()

    df_list = []
    for f in all_weather_files:
        try:
            temp = pd.read_csv(f)
            if 'date' in temp.columns:
                temp['date'] = pd.to_datetime(temp['date'])
                temp['year'] = temp['date'].dt.year
                # Filter strictly
                temp = temp[(temp['year'] >= start_year) & (temp['year'] <= end_year)]
                if not temp.empty: df_list.append(temp)
        except:
            continue

    df_daily = pd.concat(df_list, ignore_index=True)
    df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)
    df_daily['month'] = df_daily['date'].dt.month

    # Column Normalization
    if 'prec' in df_daily.columns: df_daily.rename(columns={'prec': 'precip'}, inplace=True)

    # --- Intermediate Variables ---
    logging.info("Calculating intermediate rolling variables...")
    df_daily = df_daily.sort_values(by=['district_no', 'date'])

    df_daily['precip_rolling_sum'] = df_daily.groupby('district_no')['precip'].transform(
        lambda x: x.rolling(window=PARAMS['PRECIP_DEFICIT_WINDOW'], min_periods=1).sum()
    )
    df_daily['diurnal_temp_range'] = df_daily['tmax'] - df_daily['tmin']

    # Normals (1981-2010 Baseline)
    df_normals = df_daily[df_daily['year'].between(1981, 2010)].copy()
    daily_normals = df_normals.groupby(df_normals['date'].dt.dayofyear)[['tmax', 'precip']].mean().rename(
        columns={'tmax': 'tmax_30yr_avg', 'precip': 'precip_30yr_avg'})

    df_daily = pd.merge(df_daily, daily_normals, left_on=df_daily['date'].dt.dayofyear, right_index=True, how='left')
    df_daily['tmax_anomaly_daily_pos'] = (df_daily['tmax'] - df_daily['tmax_30yr_avg']).clip(lower=0)
    df_daily['precip_anomaly_daily_neg'] = (df_daily['precip_30yr_avg'] - df_daily['precip']).clip(lower=0)

    # DTR Thresholds
    dtr_thresholds = df_daily.groupby(['district_no', 'month'])['diurnal_temp_range'].transform(
        'quantile', PARAMS['DTR_SUNNY_DAY_QUANTILE']
    )
    df_daily['dtr_p75_local'] = dtr_thresholds

    # Phases
    df_daily['phase'] = 0
    df_daily.loc[df_daily['month'].isin([4, 5, 6]), 'phase'] = 1  # Establishment
    df_daily.loc[df_daily['month'].isin([7, 8, 9]), 'phase'] = 2  # Bulking

    # --- Aggregation ---
    def calc_indices(g):
        # CASDI: Heat + Drought in Phase 2
        casdi = g.loc[
            (g['phase'] == 2) &
            (g['tmax'] > PARAMS['TMAX_STRESS_THRESHOLD']) &
            (g['precip_rolling_sum'] < PARAMS['PRECIP_DEFICIT_THRESHOLD'])
            ].shape[0]

        # NMSD: High Night Temps
        nmsd = g.loc[(g['phase'] == 2) & (g['tmin'] > PARAMS['TMIN_STRESS_THRESHOLD'])].shape[0]

        # OSAW: Optimal Growth Days
        osaw = g.loc[
            (g['phase'] == 2) &
            (g['tmax'].between(PARAMS['TMAX_OPTIMAL_MIN'], PARAMS['TMAX_OPTIMAL_MAX'])) &
            (g['tmin'] < PARAMS['TMIN_OPTIMAL_MAX']) &
            (g['precip_rolling_sum'] > PARAMS['PRECIP_DEFICIT_THRESHOLD']) &
            (g['diurnal_temp_range'] > g['dtr_p75_local'])
            ].shape[0]

        # ECES: Establishment Stress
        p1 = g[g['phase'] == 1]
        eces = ((p1['tmax_anomaly_daily_pos'] ** PARAMS['ECES_EXPONENT']) *
                (p1['precip_anomaly_daily_neg'] ** PARAMS['ECES_EXPONENT'])).sum()

        # Simple Heat
        heat_days = (g.loc[g['month'].isin([6, 7, 8]), 'tmax'] > 30).sum()

        return pd.Series({
            'CASDI_Phase2_Count': casdi,
            'NMSD_Phase2_Count': nmsd,
            'OSAW_Phase2_Count': osaw,
            'ECES_Phase1_Cumulative': eces,
            'summer_days_tmax_gt_30c': heat_days
        })

    tqdm.pandas(desc="Aggregating Indices")
    seasonal_aggregates = df_daily.groupby(['district_no', 'year']).progress_apply(calc_indices).reset_index()

    return seasonal_aggregates


def load_and_process_economic_data(producer_price_file, input_price_file):
    logging.info("Processing Economic Data...")
    try:
        df_prod = pd.read_csv(producer_price_file)
        df_prod = df_prod[df_prod['ID'] == 'LWPR-132'].melt(
            id_vars=['ID', 'Description'], var_name='year', value_name='producer_price_index'
        )
        df_prod['year'] = pd.to_numeric(df_prod['year'], errors='coerce').astype('Int64')
        df_prod = df_prod.dropna(subset=['year'])

        df_input = pd.read_csv(input_price_file)
        IDS = {'LWBM-11': 'seed_price_index', 'LWBM-13': 'fertilizer_price_index',
               'LWBM-14': 'plant_protection_price_index'}
        df_input = df_input[df_input['ID'].isin(IDS.keys())]
        df_input = df_input.melt(id_vars=['ID', 'Description'], var_name='period', value_name='price_index')
        df_input['year'] = pd.to_numeric(df_input['period'].str.split('/').str[1], errors='coerce')
        df_input = df_input.dropna(subset=['year'])
        df_input['year'] = df_input['year'].astype(int)

        df_input = df_input.groupby(['year', 'ID'])['price_index'].mean().unstack().reset_index()
        df_input.rename(columns=IDS, inplace=True)

        df_econ = pd.merge(df_prod[['year', 'producer_price_index']], df_input, on='year', how='outer')
        return df_econ
    except Exception as e:
        logging.error(f"Econ load error: {e}")
        return pd.DataFrame(columns=['year'])


def main():
    logging.info("--- Starting Feature Engineering (v13.0 Restoration) ---")
    paths = CONFIG['FILE_PATHS']
    paths['OUTPUT_DIR'].mkdir(exist_ok=True, parents=True)

    # 1. Base Data
    df = pd.read_csv(paths['MASTER_DATASET'])
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    df['kreisYield'] = pd.to_numeric(df['yield'], errors='coerce')
    df.dropna(subset=['kreisYield'], inplace=True)

    # 2. Economics
    df_econ = load_and_process_economic_data(paths['PRODUCER_PRICE_CSV'], paths['INPUT_PRICE_CSV'])
    if not df_econ.empty:
        df = pd.merge(df, df_econ, on='year', how='left')

    # 3. Advanced Weather (Physiological)
    df_phys = create_granular_weather_features(paths['DAILY_WEATHER_DIR'], 1981, 2024)
    if not df_phys.empty:
        df = pd.merge(df, df_phys, on=['district_no', 'year'], how='left')

    # 4. Satellite
    df_sat = pd.read_csv(paths['SATELLITE_FEATURES_CSV'], dtype={'district_no': str})
    df_sat['district_no'] = df_sat['district_no'].str.zfill(5)
    df = pd.merge(df, df_sat, on=['district_no', 'year'], how='left')

    # 5. Forecast (The Baseline)
    df_fcst = pd.read_csv(paths['WALKFORWARD_FORECAST_CSV'], dtype={'district_no': str})
    df_fcst.rename(columns={'final_corrected_forecast': 'stage1_forecast'}, inplace=True)
    df = pd.merge(df, df_fcst[['year', 'district_no', 'stage1_forecast']], on=['year', 'district_no'], how='left')

    # 6. WOFOST Forecast (for Interactions)
    # Note: Using the provided logic, we seem to treat 'stage1_forecast' as the main WOFOST/Trend blend
    df['has_wofost_data'] = df['stage1_forecast'].notna().astype(int)

    # 7. State Encoding
    gdf = gpd.read_file(paths['GEOJSON_DISTRICTS'])
    gdf_states = gdf[['id', 'state']].rename(columns={'id': 'district_no'})
    gdf_states['district_no'] = gdf_states['district_no'].astype(str).str.zfill(5)
    df = pd.merge(df, gdf_states, on='district_no', how='left')
    df['state_encoded'], _ = pd.factorize(df['state'])

    # 8. Feature Engineering (The "Good" Stuff)
    df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    df['year_trend'] = df['year'] - df['year'].min()

    # Lags
    for col in ['producer_price_index', 'seed_price_index', 'fertilizer_price_index', 'plant_protection_price_index']:
        if col in df.columns:
            df[f'{col}_lag1'] = df.groupby('district_no')[col].shift(1).ffill().bfill()

    # Complex Interactions
    if 'producer_price_index_lag1' in df.columns and 'fertilizer_price_index_lag1' in df.columns:
        df['profit_margin_proxy_lag1'] = df['producer_price_index_lag1'] / (df['fertilizer_price_index_lag1'] + 1e-6)
        df['wofost_forecast_x_profit_margin'] = df['stage1_forecast'] * df['profit_margin_proxy_lag1']

    if 'fertilizer_price_index_lag1' in df.columns and 'plant_protection_price_index_lag1' in df.columns:
        df['cost_of_inputs_lag1'] = df['fertilizer_price_index_lag1'] + df['plant_protection_price_index_lag1']

    # Anomalies
    if 'fertilizer_price_index_lag1' in df.columns:
        trend = df.groupby('district_no')['fertilizer_price_index_lag1'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        df['fertilizer_price_index_lag1_anomaly'] = df['fertilizer_price_index_lag1'] - trend.ffill().bfill()

        # Capped
        lb = df['fertilizer_price_index_lag1_anomaly'].quantile(0.05)
        ub = df['fertilizer_price_index_lag1_anomaly'].quantile(0.95)
        df['fertilizer_price_index_lag1_anomaly_capped'] = df['fertilizer_price_index_lag1_anomaly'].clip(lb, ub)
        df['is_fertilizer_price_extreme'] = ((df['fertilizer_price_index_lag1_anomaly'] < lb) | (
                    df['fertilizer_price_index_lag1_anomaly'] > ub)).astype(int)

    # Bring back standard anomalies if missing (for the feature list)
    # Using simple fill for now to satisfy the "Good Feature List"
    placeholders = [
        'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
        'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
        'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
        'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
        'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
        'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast',
        'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast', 'summer_temp_prob_warm_forecast',
        'summer_precip_prob_wet_forecast', 'nao_winter_avg', 'sca_winter_avg', 'enso_mei_winter_avg'
    ]

    # We load these from the "forecast" or "metrics" files if possible, else 0
    # To save time, I will pull them from the ECMWF forecast file we have in the config
    df_fcst_feat = pd.read_csv(config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']['ECMWF_FORECAST_FEATURES_CSV'])
    df_fcst_feat['district_no'] = df_fcst_feat['district_no'].astype(str).str.zfill(5)

    # Aggregation
    agg_d = {c: 'mean' for c in placeholders if c in df_fcst_feat.columns}
    if agg_d:
        df_agg = df_fcst_feat.groupby(['year', 'district_no']).agg(agg_d).reset_index()
        df = pd.merge(df, df_agg, on=['year', 'district_no'], how='left')

    # Interactions from the Good List
    if 'antecedent_gdd_sum_anomaly' in df.columns and 'fertilizer_price_index_lag1_anomaly_capped' in df.columns:
        df['gdd_x_fertilizer_price'] = df['antecedent_gdd_sum_anomaly'] * df[
            'fertilizer_price_index_lag1_anomaly_capped']

    if 'summer_temp_prob_warm_forecast' in df.columns and 'profit_margin_proxy_lag1' in df.columns:
        df['summer_heat_x_profit_margin'] = df['summer_temp_prob_warm_forecast'] * df['profit_margin_proxy_lag1']

    if 'summer_precip_prob_wet_forecast' in df.columns and 'cost_of_inputs_lag1' in df.columns:
        df['summer_precip_x_input_costs'] = df['summer_precip_prob_wet_forecast'] * df['cost_of_inputs_lag1']

    if 'summer_temp_anomaly_forecast' in df.columns and 'summer_precip_anomaly_forecast' in df.columns:
        df['hot_dry_interaction'] = df['summer_temp_anomaly_forecast'] * (df['summer_precip_anomaly_forecast'] * -1)

    if 'lat' in df.columns and 'summer_temp_anomaly_forecast' in df.columns:
        df['lat_x_summer_temp'] = df['lat'] * df['summer_temp_anomaly_forecast']

    if 'avg_sand_0_30cm' in df.columns and 'summer_precip_anomaly_forecast' in df.columns:
        df['sandy_soil_x_drought'] = df['avg_sand_0_30cm'] * df['summer_precip_anomaly_forecast']

    # Squares
    for c in ['spring_temp_prob_warm_forecast', 'summer_temp_prob_warm_forecast']:
        if c in df.columns: df[f'{c}_sq'] = df[c] ** 2

    # Fill NaNs with 0 for features to prevent drop
    df.fillna(0, inplace=True)

    df.to_csv(paths['OUTPUT_FILE'], index=False)
    logging.info(f"✓ Feature Engineering Restored. Saved to {paths['OUTPUT_FILE']}")


if __name__ == '__main__':
    main()