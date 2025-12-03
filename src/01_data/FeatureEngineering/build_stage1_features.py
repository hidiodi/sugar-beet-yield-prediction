# File: src/01_data/FeatureEngineering/build_stage1_features.py
# REFACTORED (v15.0): Exact Restoration of "Good Setup"
# Ensures all polynomial, interaction, and economic anomaly features are present.

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
    logging.info("--- Engineering Physiological Weather Features ---")
    PARAMS = config.FEATURE_ENGINEERING_CONFIG['PHYSIOLOGY_PARAMS']  # Use from config if available or defaults below
    # Fallback if config is missing specific keys
    if 'TMAX_STRESS_THRESHOLD' not in PARAMS:
        PARAMS = {
            'TMAX_STRESS_THRESHOLD': 30.0, 'TMIN_STRESS_THRESHOLD': 17.0,
            'TMAX_OPTIMAL_MIN': 17.0, 'TMAX_OPTIMAL_MAX': 25.0, 'TMIN_OPTIMAL_MAX': 15.0,
            'PRECIP_DEFICIT_WINDOW': 30, 'PRECIP_DEFICIT_THRESHOLD': 20.0,
            'ECES_EXPONENT': 1.5, 'DTR_SUNNY_DAY_QUANTILE': 0.75
        }

    all_weather_files = list(weather_dir.glob("*.csv"))
    if not all_weather_files: return pd.DataFrame()

    df_list = []
    for f in tqdm(all_weather_files, desc="Loading Daily Weather"):
        try:
            temp = pd.read_csv(f, usecols=lambda x: x in ['district_no', 'date', 'tmin', 'tmax', 'precip', 'prec'])
            if 'date' in temp.columns:
                temp['date'] = pd.to_datetime(temp['date'])
                temp['year'] = temp['date'].dt.year
                temp = temp[(temp['year'] >= start_year) & (temp['year'] <= end_year)]
                if not temp.empty: df_list.append(temp)
        except:
            continue

    if not df_list: return pd.DataFrame()
    df_daily = pd.concat(df_list, ignore_index=True)
    df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)
    df_daily['month'] = df_daily['date'].dt.month
    if 'prec' in df_daily.columns: df_daily.rename(columns={'prec': 'precip'}, inplace=True)

    df_daily.sort_values(by=['district_no', 'date'], inplace=True)
    df_daily['precip_rolling_sum'] = df_daily.groupby('district_no')['precip'].transform(
        lambda x: x.rolling(30, min_periods=1).sum())
    df_daily['diurnal_temp_range'] = df_daily['tmax'] - df_daily['tmin']

    # Climatology
    daily_clim = df_daily.groupby(df_daily['date'].dt.dayofyear)[['tmax', 'precip']].mean()
    df_daily['doy'] = df_daily['date'].dt.dayofyear
    df_daily = df_daily.merge(daily_clim, left_on='doy', right_index=True, suffixes=('', '_clim'))
    df_daily['tmax_anomaly_daily_pos'] = (df_daily['tmax'] - df_daily['tmax_clim']).clip(lower=0)
    df_daily['precip_anomaly_daily_neg'] = (df_daily['precip_clim'] - df_daily['precip']).clip(lower=0)

    dtr_thresh = df_daily.groupby(['district_no', 'month'])['diurnal_temp_range'].transform('quantile', 0.75)
    df_daily['dtr_p75_local'] = dtr_thresh

    df_daily['phase'] = 0
    df_daily.loc[df_daily['month'].isin([4, 5, 6]), 'phase'] = 1
    df_daily.loc[df_daily['month'].isin([7, 8, 9]), 'phase'] = 2

    def calc_indices(g):
        casdi = ((g['phase'] == 2) & (g['tmax'] > 30) & (g['precip_rolling_sum'] < 20)).sum()
        nmsd = ((g['phase'] == 2) & (g['tmin'] > 17)).sum()
        osaw = ((g['phase'] == 2) & (g['tmax'].between(17, 25)) & (g['tmin'] < 15) & (g['precip_rolling_sum'] > 20) & (
                    g['diurnal_temp_range'] > g['dtr_p75_local'])).sum()
        p1 = g[g['phase'] == 1]
        eces = ((p1['tmax_anomaly_daily_pos'] ** 1.5) * (p1['precip_anomaly_daily_neg'] ** 1.5)).sum()
        heat = ((g['month'].isin([6, 7, 8])) & (g['tmax'] > 30)).sum()
        return pd.Series({'CASDI_Phase2_Count': casdi, 'NMSD_Phase2_Count': nmsd, 'OSAW_Phase2_Count': osaw,
                          'ECES_Phase1_Cumulative': eces, 'summer_days_tmax_gt_30c': heat})

    return df_daily.groupby(['district_no', 'year']).apply(calc_indices).reset_index()


def load_forecasts_with_mapping(forecast_file):
    logging.info("Loading Forecasts...")
    try:
        df = pd.read_csv(forecast_file)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        # Assuming the file is already aggregated or we aggregate here
        numeric_cols = [c for c in df.columns if c not in ['district_no', 'year', 'member']]
        df_agg = df.groupby(['year', 'district_no'])[numeric_cols].mean().reset_index()

        # Mapping based on typical output
        # If your intermediate file already has the right names, this is a pass-through
        # But we ensure they exist.
        return df_agg
    except:
        return pd.DataFrame()


def load_economics(prod_file, input_file):
    logging.info("Loading Economics...")
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


def main():
    logging.info("--- Starting Feature Engineering (v15.0 - The Restoration) ---")
    paths = CONFIG['FILE_PATHS']
    paths['OUTPUT_DIR'].mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(paths['MASTER_DATASET'])
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    df['kreisYield'] = pd.to_numeric(df['yield'], errors='coerce')
    df.dropna(subset=['kreisYield'], inplace=True)

    # 1. Economics
    df_econ = load_economics(paths['PRODUCER_PRICE_CSV'], paths['INPUT_PRICE_CSV'])
    if not df_econ.empty: df = pd.merge(df, df_econ, on='year', how='left')

    # 2. Weather Physio
    df_phys = create_granular_weather_features(paths['DAILY_WEATHER_DIR'], 1981, 2024)
    if not df_phys.empty: df = pd.merge(df, df_phys, on=['district_no', 'year'], how='left')

    # 3. Forecasts
    df_fcst = load_forecasts_with_mapping(paths['ECMWF_FORECAST_FEATURES_CSV'])
    if not df_fcst.empty:
        cols = [c for c in df_fcst.columns if c not in df.columns or c in ['year', 'district_no']]
        df = pd.merge(df, df_fcst[cols], on=['year', 'district_no'], how='left')

    # 4. Satellite
    df_sat = pd.read_csv(paths['SATELLITE_FEATURES_CSV'], dtype={'district_no': str})
    df_sat['district_no'] = df_sat['district_no'].str.zfill(5)
    cols = [c for c in df_sat.columns if c not in df.columns or c in ['year', 'district_no']]
    df = pd.merge(df, df_sat[cols], on=['district_no', 'year'], how='left')

    # 5. Baseline (Stage1 Forecast)
    df_trend = pd.read_csv(paths['WALKFORWARD_FORECAST_CSV'], dtype={'district_no': str})
    df_trend.rename(columns={'final_corrected_forecast': 'stage1_forecast'}, inplace=True)
    df = pd.merge(df, df_trend[['year', 'district_no', 'stage1_forecast']], on=['year', 'district_no'], how='left')

    # 6. WOFOST (for imputing stage1 if needed)
    df['stage1_forecast'] = df['stage1_forecast'].fillna(df.groupby('district_no')['kreisYield'].transform('mean'))
    df['has_wofost_data'] = df['stage1_forecast'].notna().astype(int)

    # 7. States
    gdf = gpd.read_file(paths['GEOJSON_DISTRICTS'])
    gdf_states = gdf[['id', 'state']].rename(columns={'id': 'district_no'})
    gdf_states['district_no'] = gdf_states['district_no'].astype(str).str.zfill(5)
    df = pd.merge(df, gdf_states, on='district_no', how='left')
    df['state_encoded'], _ = pd.factorize(df['state'])

    # --- START INTEGRATION: GDR FLAG ---
    # States 12-16 are the "New States" (former GDR)
    # Using >= 12 excludes Berlin (11) which is historically mixed.
    df['is_gdr'] = (df['district_no'].str[:2].astype(int) >= 11).astype(int)

    # 8. Feature Generation
    df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    df['year_trend'] = df['year'] - df['year'].min()

    # Economic Lags
    for c in ['producer_price_index', 'seed_price_index', 'fertilizer_price_index', 'plant_protection_price_index',
              'energy_price_index']:
        if c in df.columns:
            df[f'{c}_lag1'] = df.groupby('district_no')[c].shift(1).ffill().bfill()

    # Economic Anomalies & Interactions
    if 'producer_price_index_lag1' in df.columns and 'fertilizer_price_index_lag1' in df.columns:
        df['profit_margin_proxy_lag1'] = df['producer_price_index_lag1'] / (df['fertilizer_price_index_lag1'] + 1e-6)
        df['wofost_forecast_x_profit_margin'] = df['stage1_forecast'] * df['profit_margin_proxy_lag1']

    if 'fertilizer_price_index_lag1' in df.columns and 'plant_protection_price_index_lag1' in df.columns:
        df['cost_of_inputs_lag1'] = df['fertilizer_price_index_lag1'] + df['plant_protection_price_index_lag1']

    # Specific Anomalies
    for c in ['producer_price_index_lag1', 'seed_price_index_lag1', 'energy_price_index_lag1',
              'plant_protection_price_index_lag1']:
        if c in df.columns:
            trend = df.groupby('district_no')[c].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df[f'{c}_anomaly'] = df[c] - trend

    if 'fertilizer_price_index_lag1' in df.columns:
        trend = df.groupby('district_no')['fertilizer_price_index_lag1'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df['fertilizer_price_index_lag1_anomaly'] = df['fertilizer_price_index_lag1'] - trend
        lb = df['fertilizer_price_index_lag1_anomaly'].quantile(0.05)
        ub = df['fertilizer_price_index_lag1_anomaly'].quantile(0.95)
        df['fertilizer_price_index_lag1_anomaly_capped'] = df['fertilizer_price_index_lag1_anomaly'].clip(lb, ub)
        df['is_fertilizer_price_extreme'] = ((df['fertilizer_price_index_lag1_anomaly'] < lb) | (
                df['fertilizer_price_index_lag1_anomaly'] > ub)).astype(int)

    # --- NEW: Bumper Crop Logic (The 2014 Fix) ---
    # MOVED OUTSIDE THE FERTILIZER IF-BLOCK

    # 1. The "Early Start, Sustained Growth" Indicator
    if 'spring_temp_anomaly_forecast' in df.columns and 'summer_precip_anomaly_forecast' in df.columns:
        df['spring_warmth_x_summer_rain'] = df['spring_temp_anomaly_forecast'] * df['summer_precip_anomaly_forecast']
    else:
        df['spring_warmth_x_summer_rain'] = 0

    # 2. The "Photosynthesis Potential"
    if 'summer_solar_rad_anomaly_forecast' in df.columns and 'summer_precip_anomaly_forecast' in df.columns:
        df['summer_rad_x_summer_rain'] = df['summer_solar_rad_anomaly_forecast'] * df['summer_precip_anomaly_forecast']
    else:
        df['summer_rad_x_summer_rain'] = 0

    # 3. Regional Water Value (GDR Specific)
    if 'is_gdr' in df.columns and 'summer_precip_anomaly_forecast' in df.columns:
        df['is_gdr_x_summer_rain'] = df['is_gdr'] * df['summer_precip_anomaly_forecast']
    else:
        df['is_gdr_x_summer_rain'] = 0

    # 4. Clay Soil Buffer (The Drought "Airbag")
    # Clay holds water. If summer precip is negative (drought), high clay reduces the damage.
    if 'avg_clay_0_30cm' in df.columns and 'summer_precip_anomaly_forecast' in df.columns:
        df['clay_soil_x_drought'] = df['avg_clay_0_30cm'] * df['summer_precip_anomaly_forecast']
    else:
        df['clay_soil_x_drought'] = 0

    # 5. Excess Spring Wetness (The "Muddy Boots" Factor)
    # High rain anomaly + High probability of wetness = High risk of delayed planting.
    if 'spring_precip_anomaly_forecast' in df.columns and 'spring_precip_prob_wet_forecast' in df.columns:
        df['excess_spring_wetness'] = df['spring_precip_anomaly_forecast'] * df['spring_precip_prob_wet_forecast']
    else:
        df['excess_spring_wetness'] = 0

    # Interactions
    if 'antecedent_gdd_sum_anomaly' in df.columns and 'fertilizer_price_index_lag1_anomaly_capped' in df.columns:
        df['gdd_x_fertilizer_price'] = df['antecedent_gdd_sum_anomaly'] * df[
            'fertilizer_price_index_lag1_anomaly_capped']
    else:
        df['gdd_x_fertilizer_price'] = 0

    if 'spring_temp_anomaly_forecast' in df.columns and 'spring_precip_anomaly_forecast' in df.columns:
        df['spring_temp_x_spring_precip'] = df['spring_temp_anomaly_forecast'] * df['spring_precip_anomaly_forecast']
    else:
        df['spring_temp_x_spring_precip'] = 0

    if 'summer_temp_prob_warm_forecast' in df.columns and 'profit_margin_proxy_lag1' in df.columns:
        df['summer_heat_x_profit_margin'] = df['summer_temp_prob_warm_forecast'] * df['profit_margin_proxy_lag1']
    else:
        df['summer_heat_x_profit_margin'] = 0

    if 'summer_precip_prob_wet_forecast' in df.columns and 'cost_of_inputs_lag1' in df.columns:
        df['summer_precip_x_input_costs'] = df['summer_precip_prob_wet_forecast'] * df['cost_of_inputs_lag1']
    else:
        df['summer_precip_x_input_costs'] = 0

    if 'summer_temp_anomaly_forecast' in df.columns and 'summer_precip_anomaly_forecast' in df.columns:
        df['hot_dry_interaction'] = df['summer_temp_anomaly_forecast'] * (df['summer_precip_anomaly_forecast'] * -1)
    else:
        df['hot_dry_interaction'] = 0

    if 'lat' in df.columns and 'summer_temp_anomaly_forecast' in df.columns:
        df['lat_x_summer_temp'] = df['lat'] * df['summer_temp_anomaly_forecast']
    else:
        df['lat_x_summer_temp'] = 0

    if 'avg_sand_0_30cm' in df.columns and 'summer_precip_anomaly_forecast' in df.columns:
        df['sandy_soil_x_drought'] = df['avg_sand_0_30cm'] * df['summer_precip_anomaly_forecast']
    else:
        df['sandy_soil_x_drought'] = 0

    # Squares
    for c in ['antecedent_gdd_sum_anomaly', 'spring_temp_prob_warm_forecast', 'summer_temp_prob_warm_forecast',
              'spring_precip_prob_wet_forecast', 'summer_precip_prob_wet_forecast', 'summer_precip_anomaly_forecast']:
        if c in df.columns:
            df[f'{c}_sq'] = df[c] ** 2
        else:
            df[f'{c}_sq'] = 0

    df.fillna(0, inplace=True)

    # Drop source columns to keep file clean-ish
    drop_cols = ['yield', 'producer_price_index', 'seed_price_index', 'fertilizer_price_index',
                 'plant_protection_price_index', 'energy_price_index', 'state']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df.to_csv(paths['OUTPUT_FILE'], index=False)
    logging.info(f"✓ Feature Engineering (v15.0) Complete. Saved to {paths['OUTPUT_FILE']}")


if __name__ == '__main__':
    main()