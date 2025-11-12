# vsm_cybernetic_model/pipelines/prepare_foundational_features.py
import numpy as np
import pandas as pd
import logging
import sys

from vsm_cybernetic_model.configs import main_config as cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_process_economic_data(producer_price_file, input_price_file):
    # This helper function is correct and remains unchanged.
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


def prepare_all_foundational_features():
    logging.info("--- Starting Comprehensive Foundational Feature Preparation ---")

    master_file = cfg.DATA_DIR / "04_master" / "master_dataset.csv"
    producer_price_file = cfg.RAW_DATA_DIR / "Bundesdatenbank" / "61211-0001_de.csv"
    input_price_file = cfg.RAW_DATA_DIR / "Bundesdatenbank" / "61221-0003_de.csv"
    satellite_features_file = cfg.PRIMARY_DATA_DIR / "satellite_features_districts_2001-2024.csv"

    logging.info("STAGE 1: Loading and Merging Base Data")
    master_df = pd.read_csv(master_file)
    master_df['district_no'] = master_df['district_no'].astype(str).str.zfill(5)
    logging.info(f"Loaded master feature file from '{master_file}'")
    master_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

    df_economic = load_and_process_economic_data(producer_price_file, input_price_file)
    if df_economic is None: sys.exit(1)
    merged_df = pd.merge(master_df, df_economic, on='year', how='left')

    if 'yield' in merged_df.columns:
        merged_df['kreisYield'] = pd.to_numeric(merged_df['yield'], errors='coerce')
        merged_df.dropna(subset=['kreisYield'], inplace=True)

    logging.info("STAGE 2: Merging Additional Feature Sets")
    try:
        df_satellite = pd.read_csv(satellite_features_file)
        df_satellite['district_no'] = df_satellite['district_no'].astype(str).str.zfill(5)
        merged_df = pd.merge(merged_df, df_satellite, on=['district_no', 'year'], how='left')
        logging.info("Satellite data merged successfully.")
    except FileNotFoundError:
        logging.warning(f"Satellite features file not found at '{satellite_features_file}'. Proceeding without it.")

    logging.info("STAGE 3: Engineering All Final Features")
    merged_df = merged_df.sort_values(by=['district_no', 'year'])

    # --- Economic Features (Lagged) ---
    lag1_features = [
        'national_avg_yield', 'producer_price_index', 'seed_price_index',
        'energy_price_index', 'fertilizer_price_index', 'plant_protection_price_index'
    ]
    merged_df['national_avg_yield'] = merged_df.groupby('year')['kreisYield'].transform('mean')
    for feature in lag1_features:
        merged_df[f'{feature}_lag1'] = merged_df.groupby('district_no')[feature].shift(1)

    epsilon = 1e-6
    merged_df['profit_margin_proxy_lag1'] = merged_df['producer_price_index_lag1'] / (
                merged_df['fertilizer_price_index_lag1'] + epsilon)
    merged_df['cost_of_inputs_lag1'] = merged_df['fertilizer_price_index_lag1'] + merged_df[
        'plant_protection_price_index_lag1'] + merged_df['seed_price_index_lag1']

    # --- Economic Features (Momentum) ---
    merged_df['profit_margin_proxy_lag2'] = merged_df.groupby('district_no')['profit_margin_proxy_lag1'].shift(1)
    merged_df['cost_of_inputs_lag2'] = merged_df.groupby('district_no')['cost_of_inputs_lag1'].shift(1)
    merged_df['profit_margin_momentum'] = merged_df['profit_margin_proxy_lag1'] - merged_df['profit_margin_proxy_lag2']
    merged_df['cost_of_inputs_momentum'] = merged_df['cost_of_inputs_lag1'] - merged_df['cost_of_inputs_lag2']

    # --- START: ADDED MISSING BLOCK ---
    logging.info("Detrending economic features to create stable anomalies...")
    economic_features_to_detrend = [
        'producer_price_index_lag1', 'seed_price_index_lag1', 'energy_price_index_lag1',
        'fertilizer_price_index_lag1', 'plant_protection_price_index_lag1'
    ]
    for feature in economic_features_to_detrend:
        trend = merged_df.groupby('district_no')[feature].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        ).ffill().bfill()
        merged_df[f'{feature}_anomaly'] = merged_df[feature] - trend

    logging.info("Capping extreme values for fertilizer price...")

    def expanding_quantile(s, q):
        return s.expanding(min_periods=20).quantile(q)

    lower_bound = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: expanding_quantile(s, q=0.05)).ffill().bfill()
    upper_bound = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: expanding_quantile(s, q=0.95)).ffill().bfill()
    merged_df['fertilizer_price_index_lag1_anomaly_capped'] = merged_df['fertilizer_price_index_lag1_anomaly'].clip(
        lower=lower_bound, upper=upper_bound)
    merged_df['is_fertilizer_price_extreme'] = np.where(
        (merged_df['fertilizer_price_index_lag1_anomaly'] < lower_bound) | (
                    merged_df['fertilizer_price_index_lag1_anomaly'] > upper_bound), 1, 0)
    # --- END: ADDED MISSING BLOCK ---

    # --- Binary Weather Flags ---
    merged_df['is_summer_forecast_dry'] = np.where(merged_df['summer_precip_prob_wet_forecast'] < 0.3, 1, 0)
    hot_threshold = merged_df['summer_temp_anomaly_forecast'].quantile(0.95)
    dry_threshold = merged_df['summer_precip_anomaly_forecast'].quantile(0.05)
    merged_df['is_extreme_heat_forecast'] = np.where(merged_df['summer_temp_anomaly_forecast'] > hot_threshold, 1, 0)
    merged_df['is_extreme_drought_forecast'] = np.where(merged_df['summer_precip_anomaly_forecast'] < dry_threshold, 1,
                                                        0)

    # --- Interaction Features ---
    merged_df['gdd_x_fertilizer_price'] = merged_df['antecedent_gdd_sum_anomaly'] * merged_df[
        'fertilizer_price_index_lag1']
    merged_df['spring_temp_x_spring_precip'] = merged_df['spring_temp_anomaly_forecast'] * merged_df[
        'spring_precip_anomaly_forecast']
    merged_df['summer_heat_x_profit_margin'] = merged_df['summer_temp_prob_warm_forecast'] * merged_df[
        'profit_margin_proxy_lag1']
    merged_df['summer_precip_x_input_costs'] = merged_df['summer_precip_prob_wet_forecast'] * merged_df[
        'cost_of_inputs_lag1']
    merged_df['drought_x_heat'] = merged_df['is_extreme_heat_forecast'] * merged_df['is_extreme_drought_forecast']
    merged_df['hot_dry_interaction'] = merged_df['summer_temp_anomaly_forecast'] * (
                merged_df['summer_precip_anomaly_forecast'] * -1)
    merged_df['lat_x_summer_temp'] = merged_df['lat'] * merged_df['summer_temp_anomaly_forecast']
    merged_df['lon_x_spring_precip'] = merged_df['lon'] * merged_df['spring_precip_anomaly_forecast']
    merged_df['sandy_soil_x_drought'] = merged_df['avg_sand_0_30cm'] * merged_df['summer_precip_anomaly_forecast']

    # --- Polynomial Features ---
    poly_features = {
        'antecedent_gdd_sum_anomaly': [2, 3], 'summer_temp_anomaly_forecast': [2, 3],
        'summer_precip_anomaly_forecast': [2, 3], 'spring_temp_prob_warm_forecast': [2],
        'summer_temp_prob_warm_forecast': [2], 'spring_precip_prob_wet_forecast': [2],
        'summer_precip_prob_wet_forecast': [2]
    }
    for feature, powers in poly_features.items():
        for power in powers:
            suffix = '_sq' if power == 2 else '_cubed'
            merged_df[f'{feature}{suffix}'] = merged_df[feature] ** power

    logging.info("All features engineered successfully.")

    logging.info("Finalizing dataset for VSM pipeline...")
    output_file = cfg.FOUNDATIONAL_FEATURES_HUMAN
    merged_df.to_csv(output_file, index=False)

    print(f"Saved comprehensive foundational features to '{output_file}'")
    print("--- Foundational Feature Preparation Complete ---")


if __name__ == "__main__":
    prepare_all_foundational_features()