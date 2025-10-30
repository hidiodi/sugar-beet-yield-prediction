import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import geopandas as gpd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- NEW FUNCTION: GRANULAR WEATHER FEATURE ENGINEERING ---
def create_granular_weather_features(weather_dir: Path, start_year: int, end_year: int):
    """
    Loads all daily weather files and engineers a rich set of seasonal weather features.
    """
    logging.info("--- Starting Granular Weather Feature Engineering ---")

    all_weather_files = [weather_dir / f"historical_daily_weather_era5_{year}.csv" for year in
                         range(start_year, end_year + 1)]
    df_weather = pd.concat([pd.read_csv(f) for f in all_weather_files if f.exists()])

    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_weather['month'] = df_weather['date'].dt.month
    df_weather['year'] = df_weather['date'].dt.year

    summer_months = [6, 7, 8]
    df_weather['precip_mm'] = df_weather['precip'] / 100000.0
    df_weather['is_very_heavy_rain'] = (df_weather['precip_mm'] > 20).astype(int)
    df_weather['is_heatwave_day'] = (df_weather['tmax'] > 30).astype(int)

    tqdm.pandas(desc="Aggregating Granular Weather")
    seasonal_aggregates = df_weather.groupby(['district_no', 'year']).progress_apply(lambda g: pd.Series({
        'summer_days_precip_gt_20mm': g[g['month'].isin(summer_months)]['is_very_heavy_rain'].sum(),
        'summer_days_tmax_gt_30c': g[g['month'].isin(summer_months)]['is_heatwave_day'].sum(),
    })).reset_index()

    logging.info("✓ Granular weather features (extreme event counters) created successfully.")
    return seasonal_aggregates


def load_and_process_economic_data(producer_price_file, input_price_file):
    # This function is correct and remains unchanged.
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

    # Consolidated file paths
    master_file = Path('data/04_master/master_dataset.csv')
    producer_price_file = Path('data/01_raw/Bundesdatenbank/61211-0001_de.csv')
    input_price_file = Path('data/01_raw/Bundesdatenbank/61221-0003_de.csv')
    satellite_features_file = Path('data/03_primary/satellite_features_districts_2001-2024.csv')
    geojson_file = Path('data/01_raw/districts_official.geojson')
    weather_dir = Path('data/02_intermediate/daily_weather')
    output_path = Path('data/05_model_input/')
    output_file = output_path / 'stage1_preseason_features.csv'
    output_path.mkdir(exist_ok=True, parents=True)

    # Dynamically find the latest WOFOST simulation output
    walkforward_forecast_file = Path('data/09_model_output_walkforward_final/final_honest_forecasts.csv')
    if not walkforward_forecast_file.exists():
        logging.error(f"FATAL: The required forecast file was not found at {walkforward_forecast_file}. Cannot proceed.")
        sys.exit(1)
    logging.info(f"Using honest, walk-forward forecast file: {walkforward_forecast_file}")


    # --- STAGE 1: LOAD & MERGE BASE DATA ---
    logging.info("STAGE 1: Loading and Merging Base Data")
    master_df = pd.read_csv(master_file)
    master_df['district_no'] = master_df['district_no'].astype(str).str.zfill(5)
    df_economic = load_and_process_economic_data(producer_price_file, input_price_file)
    if df_economic is None: sys.exit(1)
    merged_df = pd.merge(master_df, df_economic, on='year', how='left')
    merged_df['kreisYield'] = pd.to_numeric(merged_df['yield'], errors='coerce')
    merged_df.dropna(subset=['kreisYield'], inplace=True)

    # --- STAGE 2: AUGMENT WITH ALL ADDITIONAL FEATURE SETS ---
    logging.info("STAGE 2: Augmenting with All Additional Feature Sets")

    # 2a: Merge Granular Weather Features
    df_granular_weather = create_granular_weather_features(weather_dir, start_year=1981, end_year=2024)
    df_granular_weather['district_no'] = df_granular_weather['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, df_granular_weather, on=['district_no', 'year'], how='left')

    # 2b: Merge satellite data
    df_satellite = pd.read_csv(satellite_features_file)
    df_satellite['district_no'] = df_satellite['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, df_satellite, on=['district_no', 'year'], how='left')
    logging.info("Satellite data merged.")

    # 2c: Merge the new, honest walk-forward forecast as a feature
    df_forecast = pd.read_csv(walkforward_forecast_file)
    df_forecast['district_no'] = df_forecast['district_no'].astype(str).str.zfill(5)

    # Select the relevant forecast columns to use as features
    # 'final_corrected_forecast' is our best prediction.
    # 'base_trend_forecast' can also be a useful feature on its own.
    columns_to_merge = ['year', 'district_no', 'final_corrected_forecast', 'base_trend_forecast']

    merged_df = pd.merge(merged_df, df_forecast[columns_to_merge], on=['year', 'district_no'], how='left')
    logging.info("✓ Walk-forward forecast features successfully merged.")

    # 2d: Merge and encode state names
    gdf = gpd.read_file(geojson_file)
    gdf_states = gdf[['id', 'state']].rename(columns={'id': 'district_no'})
    gdf_states['district_no'] = gdf_states['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, gdf_states, on='district_no', how='left')
    merged_df['state_encoded'], _ = pd.factorize(merged_df['state'])
    logging.info("✓ State-level features created successfully using Label Encoding.")

    # --- STAGE 3: ENGINEERING THE COMPLETE HYBRID FEATURE SET ---
    logging.info("STAGE 3: Engineering the Complete Hybrid Feature Set")

    merged_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
    merged_df['year_trend'] = merged_df['year'] - merged_df['year'].min()
    merged_df = merged_df.sort_values(by=['district_no', 'year'])

    # This section now restores ALL original feature engineering

    # Lagged economic features
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

    # Process the new forecast features
    # RENAME our new forecast to the name expected by downstream feature engineering steps.
    merged_df.rename(columns={'final_corrected_forecast': 'wofost_forecast_yield_fresh_dt'}, inplace=True)

    # The unit conversion line is NO LONGER NEEDED as our new forecast is already in fresh weight dt/ha.
    # The following lines will now use our superior forecast as their input.
    merged_df['wofost_forecast_x_profit_margin'] = merged_df['wofost_forecast_yield_fresh_dt'] * merged_df[
        'profit_margin_proxy_lag1']
    merged_df['has_wofost_data'] = merged_df['wofost_forecast_yield_fresh_dt'].notna().astype(int)

    # Economic anomalies
    economic_features_to_detrend = list(lagged_cols.values())
    for feature in economic_features_to_detrend:
        trend = merged_df.groupby('district_no')[feature].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
        merged_df[f'{feature}_anomaly'] = merged_df[feature] - trend.ffill().bfill()

    # Capped and extreme economic features
    lower_bound_series = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: s.expanding(min_periods=20).quantile(0.05)).ffill().bfill()
    upper_bound_series = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: s.expanding(min_periods=20).quantile(0.95)).ffill().bfill()
    merged_df['fertilizer_price_index_lag1_anomaly_capped'] = merged_df['fertilizer_price_index_lag1_anomaly'].clip(
        lower=lower_bound_series, upper=upper_bound_series)
    merged_df['is_fertilizer_price_extreme'] = np.where(
        (merged_df['fertilizer_price_index_lag1_anomaly'] < lower_bound_series) | (
                    merged_df['fertilizer_price_index_lag1_anomaly'] > upper_bound_series), 1, 0)

    # Original interaction features
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

    # Original polynomial features
    merged_df['antecedent_gdd_sum_anomaly_sq'] = merged_df['antecedent_gdd_sum_anomaly'] ** 2
    for feature in ['spring_temp_prob_warm_forecast', 'summer_temp_prob_warm_forecast',
                    'spring_precip_prob_wet_forecast', 'summer_precip_prob_wet_forecast']:
        merged_df[f'{feature}_sq'] = merged_df[feature] ** 2

    # "Wetness penalty" feature
    merged_df['summer_precip_anomaly_forecast_sq'] = merged_df['summer_precip_anomaly_forecast'] ** 2

    # --- V5 FEATURE ENGINEERING: TARGETED NUMERICAL INTERACTIONS ---
    logging.info("Creating targeted numerical interaction features based on V4 diagnostics...")

    # Isolate the precipitation effect for the most problematic state
    merged_df['state6_precip_interaction'] = np.where(
        merged_df['state_encoded'] == 6,
        merged_df['summer_precip_anomaly_forecast'],
        0
    )
    logging.info("Creating hyper-targeted interaction feature for State 11 drought/clay issue...")

    # Define the high-risk conditions based on our final diagnosis
    is_state_11 = merged_df['state_encoded'] == 11
    is_drought = merged_df['summer_precip_anomaly_forecast'] < -0.1
    is_high_clay = merged_df['avg_clay_0_30cm'] > 25

    # Create a binary flag that is 1 only when all three conditions are met
    merged_df['is_drought_high_clay_in_state_11'] = (is_state_11 & is_drought & is_high_clay).astype(int)

    logging.info("✓ Created definitive 'Toxic Combo' feature for State 11.")

    logging.info("✓ Created state-specific precipitation interaction terms.")

    # --- FINAL CLEANUP ---
    cols_to_remove = [
        'yield', 'producer_price_index', 'seed_price_index', 'energy_price_index', 'fertilizer_price_index',
        'plant_protection_price_index', 'state', 'state_name', 'crs_anomaly'
    ]
    merged_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    logging.info(f"Dropped intermediate source columns to finalize dataset.")

    # --- SAVE FINAL DATASET ---
    logging.info(f"Saving Stage 1 model-ready dataset to '{output_file}'...")
    merged_df.to_csv(output_file, index=False, float_format='%.6f')

    logging.info(f"Stage 1 pre-season forecast dataset created!")
    logging.info(f"Dataset saved to '{output_file}' with {merged_df.shape[0]} rows.")
    logging.info(f"Final columns ({len(merged_df.columns)} total): {merged_df.columns.tolist()}")


if __name__ == '__main__':
    main()
