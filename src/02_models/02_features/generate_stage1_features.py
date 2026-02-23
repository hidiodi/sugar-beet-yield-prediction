# File: src/01_data/FeatureEngineering/build_stage1_features.py
# REFACTORED (v17.0): Scenario Analysis Mode
# Restores Observed Summer features to allow "Perfect Information" stress testing.

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
from src import config as global_config
import importlib
data_config = importlib.import_module("src.01_data.config")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = data_config.FEATURE_ENGINEERING_CONFIG


def calculate_z_score(series):
    # FIX: Use expanding window to prevent look-ahead bias.
    # We only use data from the past up to the current row.

    # 1. Calculate expanding stats (min_periods=2 to allow std calc)
    exp_mean = series.expanding(min_periods=1).mean()
    exp_std = series.expanding(min_periods=2).std()

    # 2. Handle initial NaNs or zero std deviations
    exp_std = exp_std.fillna(0)

    # 3. Avoid division by zero
    z_scores = (series - exp_mean) / exp_std

    # Replace infinite values (div by 0) with 0 or a neutral value
    z_scores = z_scores.replace([np.inf, -np.inf], 0).fillna(0)

    return z_scores

def load_forecasts_with_mapping(forecast_file):
    try:
        df = pd.read_csv(forecast_file)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        numeric_cols = [c for c in df.columns if c not in ['district_no', 'year', 'member']]
        df_agg = df.groupby(['year', 'district_no'])[numeric_cols].mean().reset_index()
        return df_agg
    except:
        return pd.DataFrame()


def load_wofost_sowing_dates(initial_conditions_path):
    try:
        df = pd.read_csv(initial_conditions_path)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        df['sowing_date'] = pd.to_datetime(df['sowing_date'])
        df['sowing_doy'] = df['sowing_date'].dt.dayofyear
        df['sowing_doy_anomaly'] = df.groupby('district_no')['sowing_doy'].transform(lambda x: x - x.expanding().mean())
        return df[['district_no', 'year', 'sowing_doy', 'sowing_doy_anomaly']]
    except:
        return pd.DataFrame()


def load_wofost_risk_metrics(metrics_path):
    df = pd.read_csv(metrics_path, dtype={'district_no': str})
    raw_path = CONFIG['FILE_PATHS']['WOFOST_ENSEMBLE_CSV']
    df_yield = pd.read_csv(raw_path, usecols=['year', 'district_no', 'yield_water_limited'], dtype={'district_no': str})
    df_yield = df_yield.groupby(['year', 'district_no'])['yield_water_limited'].mean().reset_index()
    df_yield.rename(columns={'yield_water_limited': 'wofost_yield_water_limited'}, inplace=True)
    df = pd.merge(df, df_yield, on=['year', 'district_no'], how='left')
    df.rename(columns={'cumulative_water_stress_mean': 'cumulative_water_stress'}, inplace=True)
    return df


def create_winter_recharge_features(weather_dir: Path, start_year: int, end_year: int):
    all_weather_files = list(weather_dir.glob("*.csv"))
    if not all_weather_files: return pd.DataFrame()
    df_list = []
    load_start = start_year - 1
    for f in tqdm(all_weather_files, desc="Loading Winter Weather"):
        try:
            temp = pd.read_csv(f, usecols=lambda x: x in ['district_no', 'date', 'precip', 'prec', 'tmax'])
            if 'date' in temp.columns:
                temp['date'] = pd.to_datetime(temp['date'])
                temp['year'] = temp['date'].dt.year
                temp['month'] = temp['date'].dt.month
                temp = temp[(temp['year'] >= load_start) & (temp['year'] <= end_year)]
                if not temp.empty: df_list.append(temp)
        except:
            continue
    if not df_list: return pd.DataFrame()
    df_daily = pd.concat(df_list, ignore_index=True)
    df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)
    if 'prec' in df_daily.columns: df_daily.rename(columns={'prec': 'precip'}, inplace=True)
    df_daily['crop_year'] = df_daily['year']
    df_daily.loc[df_daily['month'] >= 10, 'crop_year'] = df_daily['year'] + 1
    winter_mask = df_daily['month'].isin([10, 11, 12, 1, 2])
    df_winter = df_daily[winter_mask].copy()
    recharge = df_winter.groupby(['district_no', 'crop_year'])['precip'].sum().reset_index()
    recharge.rename(columns={'precip': 'winter_precip_sum'}, inplace=True)
    feb_mask = df_daily['month'] == 2
    frosts = df_daily[feb_mask & (df_daily['tmax'] < 0)].groupby(['district_no', 'crop_year'])[
        'tmax'].count().reset_index()
    frosts.rename(columns={'tmax': 'feb_frost_days'}, inplace=True)
    df_features = pd.merge(recharge, frosts, on=['district_no', 'crop_year'], how='left')
    df_features.fillna(0, inplace=True)
    df_features['winter_precip_anomaly'] = df_features.groupby('district_no')['winter_precip_sum'].transform(
        lambda x: x - x.expanding().mean())
    return df_features

def load_economics(prod_file, input_file):
    try:
        df_prod = pd.read_csv(prod_file)
        df_prod = df_prod[df_prod['ID'] == 'LWPR-132'].melt(id_vars=['ID', 'Description'], var_name='year',
                                                            value_name='producer_price_index')
        df_prod['year'] = pd.to_numeric(df_prod['year'], errors='coerce')
        df_prod.dropna(subset=['year'], inplace=True)
        df_prod['year'] = df_prod['year'].astype(int)
        df_in = pd.read_csv(input_file)
        IDS = {'LWBM-11': 'seed_price_index', 'LWBM-12': 'energy_price_index', 'LWBM-13': 'fertilizer_price_index',
               'LWBM-14': 'plant_protection_price_index'}
        df_in = df_in[df_in['ID'].isin(IDS.keys())].melt(id_vars=['ID', 'Description'], var_name='period',
                                                         value_name='val')
        df_in['year'] = pd.to_numeric(df_in['period'].str.split('/').str[1], errors='coerce')
        df_in.dropna(subset=['year'], inplace=True)
        df_in['year'] = df_in['year'].astype(int)
        df_in = df_in.groupby(['year', 'ID'])['val'].mean().unstack().reset_index().rename(columns=IDS)
        return pd.merge(df_prod[['year', 'producer_price_index']], df_in, on='year', how='outer')
    except:
        return pd.DataFrame()


def probability_to_days(df):
    """
    Loads 'data/processed/heat_signal_multivariate.csv' and merges 'pred_days'.
    *** MODIFIED: Fallback to heuristic is REMOVED. NaN will be returned if CSV is missing or for gaps. ***
    """
    # Original Heuristic Fallback logic (REMOVED)
    # p = df[probability_col].copy() if probability_col in df.columns else pd.Series(0, index=df.index)
    # if p.max() > 1.0: p = p / 100.0
    # baseline_prob = 0.333
    # heuristic_days = historical_mean_days + ((p - baseline_prob) * sensitivity)
    # heuristic_days = heuristic_days.clip(lower=0.0, upper=40.0)

    # 1. Load & Merge CSV
    signal_path = global_config.BASE_DIR / 'data/processed/heat_signal_multivariate_moderate.csv'

    if signal_path.exists():
        try:
            logging.info(f"Loading multivariate signal from {signal_path} (No Fallback)...")
            df_sig = pd.read_csv(signal_path, dtype={'district_no': str})
            df_sig['district_no'] = df_sig['district_no'].str.zfill(5)

            # Rename to avoid collision and select only needed cols
            df_sig = df_sig[['district_no', 'year', 'pred_days']].rename(columns={'pred_days': 'mv_days'})

            # Merge onto a copy of the key columns to preserve index alignment
            df_keys = df[['district_no', 'year']].copy()
            df_keys['__orig_idx'] = df_keys.index

            merged = pd.merge(df_keys, df_sig, on=['district_no', 'year'], how='left')
            merged.set_index('__orig_idx', inplace=True)
            merged.sort_index(inplace=True)

            # Return CSV value. NaN for gaps (Fallback removed)
            # Original: return merged['mv_days'].combine_first(heuristic_days)
            return merged['mv_days']

        except Exception as e:
            logging.warning(f"CSV Merge failed ({e}). Returning all NaNs.")
            return pd.Series(np.nan, index=df.index)
    else:
        logging.warning("Multivariate signal CSV not found. Returning all NaNs.")
        return pd.Series(np.nan, index=df.index)

def main():
    logging.info("--- Starting Feature Engineering (v17.1 - Corrected Order of Operations & Math) ---")
    paths = CONFIG['FILE_PATHS']
    paths['OUTPUT_DIR'].mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(paths['MASTER_DATASET'])
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    df['kreisYield'] = pd.to_numeric(df['yield'], errors='coerce')
    df.dropna(subset=['kreisYield'], inplace=True)


    # 2. Base Forecasts & Custom Heat Days (RESTORED)
    logging.info("Loading ECMWF Forecasts and Heat Signal...")
    df_fcst = load_forecasts_with_mapping(paths['ECMWF_FORECAST_FEATURES_CSV'])
    if not df_fcst.empty:
        cols = [c for c in df_fcst.columns if c not in df.columns or c in ['year', 'district_no']]
        df = pd.merge(df, df_fcst[cols], on=['year', 'district_no'], how='left')

        # Restore the custom Heat Signal
        df['summer_days_tmax_gt_30c'] = probability_to_days(df)
        logging.info("Applied custom Heat Signal conversion.")


    # 3. Base Weather & Sowing
    df_recharge = create_winter_recharge_features(paths['DAILY_WEATHER_DIR'], 1981, 2024)
    if not df_recharge.empty:
        df = pd.merge(df, df_recharge, left_on=['district_no', 'year'], right_on=['district_no', 'crop_year'],
                      how='left')
        df.drop(columns=['crop_year'], inplace=True)

    df_sowing = load_wofost_sowing_dates(CONFIG['FILE_PATHS']['WOFOST_INITIAL_CONDITIONS'])
    if not df_sowing.empty:
        df = pd.merge(df, df_sowing, on=['district_no', 'year'], how='left')

    df_sat = pd.read_csv(paths['SATELLITE_FEATURES_CSV'], dtype={'district_no': str})
    df_sat['district_no'] = df_sat['district_no'].str.zfill(5)
    cols = [c for c in df_sat.columns if c not in df.columns or c in ['year', 'district_no']]
    df = pd.merge(df, df_sat[cols], on=['district_no', 'year'], how='left')

    # --- CRITICAL FIX: LOAD TREND EARLY ---
    df_trend = pd.read_csv(paths['WALKFORWARD_FORECAST_CSV'], dtype={'district_no': str})
    df_trend.rename(columns={'final_corrected_forecast': 'stage1_forecast'}, inplace=True)
    df = pd.merge(df, df_trend[['year', 'district_no', 'stage1_forecast']], on=['year', 'district_no'], how='left')
    df['stage1_forecast'] = df['stage1_forecast'].fillna(
        df.groupby('district_no')['kreisYield'].transform(lambda x: x.shift(1).expanding().mean())
    )
    df['has_wofost_data'] = df['stage1_forecast'].notna().astype(int)

    # --- CRITICAL FIX: CORRECT EVAPORATION MATH EARLY ---
    # Evaporation is negative in ECMWF. Add it to precip to get net balance.
    if 'summer_precip_anomaly_forecast' in df.columns and 'summer_evaporation_anomaly_forecast' in df.columns:
        df['summer_water_balance_anomaly'] = df['summer_precip_anomaly_forecast'] + df[
            'summer_evaporation_anomaly_forecast']
    else:
        df['summer_water_balance_anomaly'] = 0

    # --- CONDITIONAL PHYSICS (Now that dependencies exist) ---
    sowing_factor = (150 - df['sowing_doy']).clip(lower=0) if 'sowing_doy' in df.columns else pd.Series(0, index=df.index)
    rad_magnitude = df[
        'summer_solar_rad_anomaly_forecast'].abs() if 'summer_solar_rad_anomaly_forecast' in df.columns else pd.Series(0, index=df.index)
    heat_days = df['summer_days_tmax_gt_30c'] if 'summer_days_tmax_gt_30c' in df.columns else pd.Series(0,
                                                                                                        index=df.index)

    direction_multiplier = np.where(heat_days > 5, -1.0, 1.0)
    penalty_weight = np.where(heat_days > 5, (1.0 + heat_days / 10.0), 1.0)
    # Ensure the calculation results in a Series before clipping
    potential_raw = (sowing_factor * rad_magnitude * direction_multiplier * penalty_weight)
    if isinstance(potential_raw, (int, float)):
        potential_raw = pd.Series(potential_raw, index=df.index)
    df['solar_capture_potential'] = potential_raw.clip(upper=250, lower=-2000)

    df['is_heat_crash'] = (df['solar_capture_potential'] < -50).astype(int)
    df['is_solar_bumper'] = (df['solar_capture_potential'] > 50).astype(int)
    df['trend_x_crash'] = df['stage1_forecast'] * df['is_heat_crash']
    df['trend_x_bumper'] = df['stage1_forecast'] * df['is_solar_bumper']

    df['heat_stress_sq'] = heat_days ** 2
    df['solar_potential_cubed'] = (df['solar_capture_potential'] / 100.0) ** 3

    wba = df['summer_water_balance_anomaly']
    df['dwd_severe_stress_days'] = (wba < -0.1).astype(int) * heat_days
    df['dwd_optimal_growth_zone'] = ((wba > 0) & (heat_days < 5)).astype(int)

    # Load WOFOST Metrics for Anoxia
    df_metrics = load_wofost_risk_metrics(CONFIG['FILE_PATHS']['WOFOST_METRICS_CSV'])
    if not df_metrics.empty:
        df = pd.merge(df, df_metrics, on=['district_no', 'year'], how='left')
        for c in ['anoxia_events', 'prob_sowing_failure', 'harvest_respiration_risk', 'prob_terminal_freeze']:
            if c in df.columns: df[c] = df[c].fillna(0)

    if 'anoxia_events' in df.columns:
        df['dwd_oxygen_stress'] = (df['anoxia_events'] > 3).astype(int)
    else:
        df['dwd_oxygen_stress'] = (wba > 0.3).astype(int)

    if 'sowing_doy' in df.columns:
        df['growing_season_length'] = 295 - df['sowing_doy']
        df['early_sowing_factor'] = (150 - df['sowing_doy']).clip(lower=0)
    else:
        df['growing_season_length'] = 0
        df['early_sowing_factor'] = 0

    if 'sowing_doy' in df.columns and 'winter_precip_sum' in df.columns:
        root_access_factor = (100 - df['sowing_doy']).clip(lower=0, upper=20) / 20.0
        df['effective_winter_water'] = df['winter_precip_sum'] * root_access_factor
        df['trend_x_winter_water'] = df['stage1_forecast'] * (df['effective_winter_water'] / 250.0).clip(upper=1.5)
        df['winter_water_surplus'] = (df['effective_winter_water'] - 200.0).clip(lower=0)
    else:
        df['effective_winter_water'] = 0
        df['trend_x_winter_water'] = 0
        df['winter_water_surplus'] = 0

    if 'sowing_doy' in df.columns and 'summer_days_tmax_gt_30c' in df.columns:
        df['late_sowing_x_summer_heat'] = df['sowing_doy'] * df['summer_days_tmax_gt_30c']
    else:
        df['late_sowing_x_summer_heat'] = 0

    df['is_gdr'] = (df['district_no'].str[:2].astype(int) >= 11).astype(int)
    df['state_encoded'] = df['district_no'].str[:2].astype(int)

    if 'wofost_forecast_x_profit_margin' not in df.columns and 'profit_margin_proxy_lag1' in df.columns:
        df['wofost_forecast_x_profit_margin'] = df['stage1_forecast'] * df['profit_margin_proxy_lag1']

    norm_solar = df['solar_capture_potential'] / 1000.0
    norm_water = (df['effective_winter_water'] / 40000.0).clip(upper=1.0)
    df['trend_x_physics'] = df['stage1_forecast'] * (norm_solar + norm_water)

    # V14 Physics Logic (Z-Scores)
    logging.info("Applying V14 Physics Logic...")
    groups = df.groupby('district_no')

    df['z_heat'] = groups['summer_days_tmax_gt_30c'].transform(
        calculate_z_score) if 'summer_days_tmax_gt_30c' in df.columns else 0
    df['z_bal'] = groups['summer_water_balance_anomaly'].transform(
        calculate_z_score) if 'summer_water_balance_anomaly' in df.columns else 0
    df['z_tank'] = groups['effective_winter_water'].transform(
        calculate_z_score) if 'effective_winter_water' in df.columns else 0
    df['z_rain'] = groups['summer_precip_anomaly_forecast'].transform(
        calculate_z_score) if 'summer_precip_anomaly_forecast' in df.columns else 0
    df['z_anoxia'] = groups['anoxia_events'].transform(calculate_z_score) if 'anoxia_events' in df.columns else 0
    df['z_sow'] = groups['sowing_doy'].transform(calculate_z_score) if 'sowing_doy' in df.columns else 0

    scorch = np.maximum(df['z_heat'].clip(lower=0) * (df['z_bal'] * -1).clip(lower=0),
                        (df['z_heat'].clip(lower=0) - 1.5).clip(lower=0) * 2.0)
    drown = (df['z_anoxia'] - 0.8).clip(lower=0) * 2.0
    late_start = (df['z_sow'] - 1.5).clip(lower=0) * 2.0

    df['Index_Failure'] = np.maximum.reduce([scorch, drown, late_start])
    df['Index_Bumper'] = ((df['z_tank'].clip(lower=0) + df['z_rain'].clip(lower=0)) / 2.0) * (df['z_heat'] * -1).clip(
        lower=0)

    df['trend_x_failure'] = df['stage1_forecast'] * np.tanh(df['Index_Failure'])
    # Overwrite the previous trend_x_bumper with the superior Z-score version
    df['trend_x_bumper'] = df['stage1_forecast'] * np.tanh(df['Index_Bumper'])

    # NOTE: We keep economic features (lagged) for Model A
    drop_cols = ['yield', 'state']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df.to_csv(paths['OUTPUT_FILE'], index=False)
    logging.info(f"✓ Feature Engineering (v17.1 Corrected Mode) Complete. Saved to {paths['OUTPUT_FILE']}")


if __name__ == '__main__':
    main()