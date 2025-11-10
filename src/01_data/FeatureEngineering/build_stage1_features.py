import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import geopandas as gpd
from tqdm import tqdm

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use the FEATURE_ENGINEERING_CONFIG dictionary from the central config file
CONFIG = config.FEATURE_ENGINEERING_CONFIG


# --- FINAL ROBUST FUNCTION: PHYSIOLOGICAL WEATHER FEATURE ENGINEERING ---
def create_granular_weather_features(weather_dir: Path, start_year: int, end_year: int):
    """
    Loads daily weather files to engineer a rich set of self-contained,
    physiologically-grounded seasonal weather features. This version is robust
    and relies only on columns known to exist (tmin, tmax, precip).
    """
    logging.info("--- Starting Final, Robust, Self-Contained Physiological Weather Feature Engineering ---")

    # --- Configuration: Final Physiological Thresholds ---
    PHYSIOLOGY_PARAMS = CONFIG['PHYSIOLOGY_PARAMS']
    logging.info(f"Using physiological parameters: {PHYSIOLOGY_PARAMS}")

    all_weather_files = [weather_dir / f"historical_daily_weather_era5_{year}.csv" for year in range(start_year, end_year + 1)]
    df_daily = pd.concat([pd.read_csv(f) for f in all_weather_files if f.exists()])
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['month'] = df_daily['date'].dt.month
    df_daily['year'] = df_daily['date'].dt.year

    # --- 1. Engineer Daily Intermediate Variables ---
    logging.info("Calculating intermediate variables (anomalies, rolling precip, DTR)...")
    df_daily = df_daily.sort_values(by=['district_no', 'date'])
    df_daily['precip_rolling_sum'] = df_daily.groupby('district_no')['precip'].transform(
        lambda x: x.rolling(window=PHYSIOLOGY_PARAMS['PRECIP_DEFICIT_WINDOW'], min_periods=1).sum()
    )
    df_daily['diurnal_temp_range'] = df_daily['tmax'] - df_daily['tmin']
    df_normals = df_daily[df_daily['year'].between(1981, 2010)].copy()
    daily_normals = df_normals.groupby(df_normals['date'].dt.dayofyear)[['tmax', 'precip']].mean().rename(
        columns={'tmax': 'tmax_30yr_avg', 'precip': 'precip_30yr_avg'})
    df_daily = pd.merge(df_daily, daily_normals, left_on=df_daily['date'].dt.dayofyear, right_index=True, how='left')
    df_daily['tmax_anomaly_daily_pos'] = (df_daily['tmax'] - df_daily['tmax_30yr_avg']).clip(lower=0)
    df_daily['precip_anomaly_daily_neg'] = (df_daily['precip_30yr_avg'] - df_daily['precip']).clip(lower=0)
    dtr_thresholds = df_daily.groupby(['district_no', 'month'])['diurnal_temp_range'].transform(
        'quantile', PHYSIOLOGY_PARAMS['DTR_SUNNY_DAY_QUANTILE']
    )
    df_daily['dtr_p75_local'] = dtr_thresholds

    # --- 2. Define Phenological Windows ---
    df_daily['phase'] = 0
    df_daily.loc[df_daily['month'].isin([4, 5, 6]), 'phase'] = 1
    df_daily.loc[df_daily['month'].isin([7, 8, 9]), 'phase'] = 2

    # --- 3. Aggregate Features by District and Year (ROBUST METHOD) ---
    tqdm.pandas(desc="Aggregating Physiological Weather Features")
    seasonal_aggregates = df_daily.groupby(['district_no', 'year']).progress_apply(lambda g: pd.Series({
        'CASDI_Phase2_Count': g.loc[
            (g['phase'] == 2) &
            (g['tmax'] > PHYSIOLOGY_PARAMS['TMAX_STRESS_THRESHOLD']) &
            (g['precip_rolling_sum'] < PHYSIOLOGY_PARAMS['PRECIP_DEFICIT_THRESHOLD'])
        ].shape[0],

        'NMSD_Phase2_Count': g.loc[
            (g['phase'] == 2) & (g['tmin'] > PHYSIOLOGY_PARAMS['TMIN_STRESS_THRESHOLD'])
        ].shape[0],

        'OSAW_Phase2_Count': g.loc[
            (g['phase'] == 2) &
            (g['tmax'] > PHYSIOLOGY_PARAMS['TMAX_OPTIMAL_MIN']) &
            (g['tmax'] < PHYSIOLOGY_PARAMS['TMAX_OPTIMAL_MAX']) &
            (g['tmin'] < PHYSIOLOGY_PARAMS['TMIN_OPTIMAL_MAX']) &
            (g['precip_rolling_sum'] > PHYSIOLOGY_PARAMS['PRECIP_DEFICIT_THRESHOLD']) &
            (g['diurnal_temp_range'] > g['dtr_p75_local'])
        ].shape[0],

        'ECES_Phase1_Cumulative': ((g.loc[g['phase'] == 1, 'tmax_anomaly_daily_pos'] ** PHYSIOLOGY_PARAMS['ECES_EXPONENT']) * \
                                  (g.loc[g['phase'] == 1, 'precip_anomaly_daily_neg'] ** PHYSIOLOGY_PARAMS['ECES_EXPONENT'])).sum(),

        'summer_days_tmax_gt_30c': (g.loc[g['month'].isin([6, 7, 8]), 'tmax'] > 30).sum(),
    })).reset_index()

    logging.info("✓ Advanced physiological weather features created successfully (final robust version).")
    return seasonal_aggregates


def load_and_process_economic_data(producer_price_file, input_price_file):
    logging.info("Loading and processing granular economic data sources...")
    try:
        df_prod_raw = pd.read_csv(producer_price_file)
        df_prod = df_prod_raw[df_prod_raw['ID'] == 'LWPR-132'].melt(
            id_vars=['ID', 'Description'], var_name='year', value_name='producer_price_index'
        )
        df_prod['year'] = pd.to_numeric(df_prod['year'])
        df_prod = df_prod[['year', 'producer_price_index']]
        df_input_raw = pd.read_csv(input_price_file)
        INPUT_COST_IDS = {
            'LWBM-11': 'seed_price_index', 'LWBM-12': 'energy_price_index',
            'LWBM-13': 'fertilizer_price_index', 'LWBM-14': 'plant_protection_price_index'
        }
        df_input = df_input_raw[df_input_raw['ID'].isin(INPUT_COST_IDS.keys())]
        df_input_melted = df_input.melt(
            id_vars=['ID', 'Description'], var_name='period', value_name='price_index'
        )
        df_input_melted['price_index'] = pd.to_numeric(df_input_melted['price_index'], errors='coerce')
        df_input_melted['year'] = pd.to_numeric(df_input_melted['period'].str.split('/').str[1], errors='coerce')
        df_input_melted.dropna(subset=['year'], inplace=True)
        df_input_melted['year'] = df_input_melted['year'].astype(int)
        df_annual_avg = df_input_melted.groupby(['year', 'ID'])['price_index'].mean().reset_index()
        df_input_final = df_annual_avg.pivot(index='year', columns='ID', values='price_index').reset_index()
        df_input_final.rename(columns=INPUT_COST_IDS, inplace=True)
        df_economic = pd.merge(df_prod, df_input_final, on='year', how='outer')
        logging.info("Granular economic datasets successfully merged.")
        return df_economic
    except Exception as e:
        logging.error(f"Failed to load or process economic data files. Details: {e}", exc_info=True)
        return None

def main():
    """Main function to build the definitive, augmented feature set."""
    logging.info("--- Starting Definitive Hybrid Feature Engineering Pipeline ---")

    # Consolidated file paths from config
    file_paths = CONFIG['FILE_PATHS']
    master_file = file_paths['MASTER_DATASET']
    producer_price_file = file_paths['PRODUCER_PRICE_CSV']
    input_price_file = file_paths['INPUT_PRICE_CSV']
    satellite_features_file = file_paths['SATELLITE_FEATURES_CSV']
    geojson_file = file_paths['GEOJSON_DISTRICTS']
    weather_dir = file_paths['DAILY_WEATHER_DIR']
    output_path = file_paths['OUTPUT_DIR']
    output_file = file_paths['OUTPUT_FILE']
    output_path.mkdir(exist_ok=True, parents=True)

    walkforward_forecast_file = file_paths['WALKFORWARD_FORECAST_CSV']
    if not walkforward_forecast_file.exists():
        logging.error(f"FATAL: The required forecast file was not found at {walkforward_forecast_file}. Cannot proceed.")
        sys.exit(1)
    logging.info(f"Using honest, walk-forward forecast file: {walkforward_forecast_file}")

    logging.info("STAGE 1: Loading and Merging Base Data")
    master_df = pd.read_csv(master_file)
    master_df['district_no'] = master_df['district_no'].astype(str).str.zfill(5)
    df_economic = load_and_process_economic_data(producer_price_file, input_price_file)
    if df_economic is None: sys.exit(1)
    merged_df = pd.merge(master_df, df_economic, on='year', how='left')
    merged_df['kreisYield'] = pd.to_numeric(merged_df['yield'], errors='coerce')
    merged_df.dropna(subset=['kreisYield'], inplace=True)

    logging.info("STAGE 2: Augmenting with All Additional Feature Sets")

    df_advanced_weather = create_granular_weather_features(
        weather_dir,
        start_year=CONFIG['WEATHER_FEATURE_YEAR_START'],
        end_year=CONFIG['WEATHER_FEATURE_YEAR_END']
    )
    df_advanced_weather['district_no'] = df_advanced_weather['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, df_advanced_weather, on=['district_no', 'year'], how='left')

    df_satellite = pd.read_csv(satellite_features_file)
    df_satellite['district_no'] = df_satellite['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, df_satellite, on=['district_no', 'year'], how='left')
    logging.info("Satellite data merged.")

    df_forecast = pd.read_csv(walkforward_forecast_file)
    df_forecast['district_no'] = df_forecast['district_no'].astype(str).str.zfill(5)
    columns_to_merge = ['year', 'district_no', 'final_corrected_forecast', 'base_trend_forecast']
    merged_df = pd.merge(merged_df, df_forecast[columns_to_merge], on=['year', 'district_no'], how='left')
    logging.info("✓ Walk-forward forecast features successfully merged.")

    gdf = gpd.read_file(geojson_file)
    gdf_states = gdf[['id', 'state']].rename(columns={'id': 'district_no'})
    gdf_states['district_no'] = gdf_states['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, gdf_states, on='district_no', how='left')
    merged_df['state_encoded'], _ = pd.factorize(merged_df['state'])
    logging.info("✓ State-level features created successfully using Label Encoding.")

    logging.info("STAGE 3: Engineering the Complete Hybrid Feature Set")

    merged_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    merged_df['year_trend'] = merged_df['year'] - merged_df['year'].min()
    merged_df = merged_df.sort_values(by=['district_no', 'year'])

    lagged_cols = {
        'producer_price_index': 'producer_price_index_lag1', 'seed_price_index': 'seed_price_index_lag1',
        'energy_price_index': 'energy_price_index_lag1', 'fertilizer_price_index': 'fertilizer_price_index_lag1',
        'plant_protection_price_index': 'plant_protection_price_index_lag1'
    }
    for col, new_col in lagged_cols.items():
        merged_df[new_col] = merged_df.groupby('district_no')[col].shift(1)
    merged_df['profit_margin_proxy_lag1'] = merged_df['producer_price_index_lag1'] / (
                merged_df['fertilizer_price_index_lag1'] + 1e-6)
    merged_df['cost_of_inputs_lag1'] = merged_df['fertilizer_price_index_lag1'] + merged_df[
        'plant_protection_price_index_lag1'] + merged_df['seed_price_index_lag1']

    merged_df.rename(columns={'final_corrected_forecast': 'wofost_forecast_yield_fresh_dt'}, inplace=True)
    merged_df['wofost_forecast_x_profit_margin'] = merged_df['wofost_forecast_yield_fresh_dt'] * merged_df[
        'profit_margin_proxy_lag1']
    merged_df['has_wofost_data'] = merged_df['wofost_forecast_yield_fresh_dt'].notna().astype(int)

    economic_features_to_detrend = list(lagged_cols.values())
    for feature in economic_features_to_detrend:
        trend = merged_df.groupby('district_no')[feature].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
        merged_df[f'{feature}_anomaly'] = merged_df[feature] - trend.ffill().bfill()

    lower_bound_series = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: s.expanding(min_periods=20).quantile(0.05)).ffill().bfill()
    upper_bound_series = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: s.expanding(min_periods=20).quantile(0.95)).ffill().bfill()
    merged_df['fertilizer_price_index_lag1_anomaly_capped'] = merged_df['fertilizer_price_index_lag1_anomaly'].clip(
        lower=lower_bound_series, upper=upper_bound_series)
    merged_df['is_fertilizer_price_extreme'] = np.where(
        (merged_df['fertilizer_price_index_lag1_anomaly'] < lower_bound_series) | (
                    merged_df['fertilizer_price_index_lag1_anomaly'] > upper_bound_series), 1, 0)

    merged_df['gdd_x_fertilizer_price'] = merged_df['antecedent_gdd_sum_anomaly'] * merged_df[
        'fertilizer_price_index_lag1_anomaly_capped']
    merged_df['spring_temp_x_spring_precip'] = merged_df['spring_temp_anomaly_forecast'] * merged_df[
        'spring_precip_anomaly_forecast']
    merged_df['summer_heat_x_profit_margin'] = merged_df['summer_temp_prob_warm_forecast'] * merged_df[
        'profit_margin_proxy_lag1']
    merged_df['summer_precip_x_input_costs'] = merged_df['summer_precip_prob_wet_forecast'] * merged_df[
        'cost_of_inputs_lag1']
    merged_df['hot_dry_interaction'] = merged_df['summer_temp_anomaly_forecast'] * (
                merged_df['summer_precip_anomaly_forecast'] * -1)
    merged_df['lat_x_summer_temp'] = merged_df['lat'] * merged_df['summer_temp_anomaly_forecast']
    merged_df['sandy_soil_x_drought'] = merged_df['avg_sand_0_30cm'] * merged_df['summer_precip_anomaly_forecast']

    merged_df['antecedent_gdd_sum_anomaly_sq'] = merged_df['antecedent_gdd_sum_anomaly'] ** 2
    for feature in ['spring_temp_prob_warm_forecast', 'summer_temp_prob_warm_forecast',
                    'spring_precip_prob_wet_forecast', 'summer_precip_prob_wet_forecast']:
        merged_df[f'{feature}_sq'] = merged_df[feature] ** 2

    merged_df['summer_precip_anomaly_forecast_sq'] = merged_df['summer_precip_anomaly_forecast'] ** 2

    cols_to_remove = [
        'yield', 'producer_price_index', 'seed_price_index', 'energy_price_index', 'fertilizer_price_index',
        'plant_protection_price_index', 'state', 'state_name', 'crs_anomaly'
    ]
    merged_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    logging.info(f"Dropped intermediate source columns to finalize dataset.")

    logging.info(f"Saving Stage 1 model-ready dataset to '{output_file}'...")
    merged_df.to_csv(output_file, index=False, float_format='%.6f')

    logging.info(f"Stage 1 pre-season forecast dataset created!")
    logging.info(f"Dataset saved to '{output_file}' with {merged_df.shape[0]} rows.")
    logging.info(f"Final columns ({len(merged_df.columns)} total): {merged_df.columns.tolist()}")


if __name__ == '__main__':
    main()
