import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# (The helper function load_and_process_economic_data remains unchanged)
def load_and_process_economic_data(producer_price_file, input_price_file):
    """
    MODIFIED to use a granular and curated set of price indices relevant to sugar beets.
    """
    logging.info("Loading and processing granular economic data sources...")
    try:
        # --- Producer Price (Sugar Beets) ---
        df_prod_raw = pd.read_csv(producer_price_file)
        # We are confident LWPR-132 is the correct ID for sugar beets.
        df_prod = df_prod_raw[df_prod_raw['ID'] == 'LWPR-132'].melt(
            id_vars=['ID', 'Description'], var_name='year', value_name='producer_price_index'
        )
        df_prod['year'] = pd.to_numeric(df_prod['year'])
        df_prod = df_prod[['year', 'producer_price_index']]

        # --- Input Prices (The Four Pillars) ---
        df_input_raw = pd.read_csv(input_price_file)

        # Define our curated set of input cost IDs and their desired column names
        INPUT_COST_IDS = {
            'LWBM-11': 'seed_price_index',
            'LWBM-12': 'energy_price_index',
            'LWBM-13': 'fertilizer_price_index',
            'LWBM-14': 'plant_protection_price_index'
        }

        df_input = df_input_raw[df_input_raw['ID'].isin(INPUT_COST_IDS.keys())]

        # Melt, process, and pivot the data
        df_input_melted = df_input.melt(
            id_vars=['ID', 'Description'], var_name='period', value_name='price_index'
        )
        df_input_melted['price_index'] = pd.to_numeric(df_input_melted['price_index'], errors='coerce')
        df_input_melted['year'] = pd.to_numeric(df_input_melted['period'].str.split('/').str[1], errors='coerce')
        df_input_melted.dropna(subset=['year'], inplace=True)
        df_input_melted['year'] = df_input_melted['year'].astype(int)
        df_annual_avg = df_input_melted.groupby(['year', 'ID'])['price_index'].mean().reset_index()
        df_input_final = df_annual_avg.pivot(index='year', columns='ID', values='price_index').reset_index()

        # Rename columns using our dictionary
        df_input_final.rename(columns=INPUT_COST_IDS, inplace=True)

        # --- Merge economic datasets ---
        df_economic = pd.merge(df_prod, df_input_final, on='year', how='outer')
        logging.info(" -> Granular economic datasets successfully merged.")
        return df_economic

    except Exception as e:
        logging.error(f"Failed to load or process economic data files. Details: {e}", exc_info=True)
        return None


def main():
    """Main function to build features for a Stage 1 (pre-season) forecast model."""
    logging.info("--- Starting Stage 1 (Pre-Season Forecast) Data Pipeline ---")

    # --- Configuration ---
    master_file = Path('data/04_master/master_dataset.csv')
    producer_price_file = Path('data/01_raw/Bundesdatenbank/61211-0001_de.csv')
    input_price_file = Path('data/01_raw/Bundesdatenbank/61221-0003_de.csv')
    satellite_features_file = Path('data/03_primary/satellite_features_districts_2001-2024.csv')
    output_path = Path('data/05_model_input/')
    output_file = output_path / 'stage1_preseason_features.csv'
    output_path.mkdir(exist_ok=True, parents=True)

    # --- STAGE 1: Data Loading ---
    logging.info("\n--- STAGE 1: Loading and Merging Base Data ---")
    master_df = pd.read_csv(master_file)
    master_df['district_no'] = master_df['district_no'].astype(str).str.zfill(5)

    if 'crs_anomaly' in master_df.columns:
        logging.info("Removing junk 'crs_anomaly' column from master data.")
        master_df = master_df.drop(columns=['crs_anomaly'])

    df_economic = load_and_process_economic_data(producer_price_file, input_price_file)
    if df_economic is None: sys.exit(1)

    overlapping_cols = [col for col in df_economic.columns if col in master_df.columns and col != 'year']
    if overlapping_cols:
        master_df = master_df.drop(columns=overlapping_cols)

    merged_df = pd.merge(master_df, df_economic, on='year', how='left')
    merged_df['kreisYield'] = pd.to_numeric(merged_df['yield'], errors='coerce')
    merged_df.dropna(subset=['kreisYield'], inplace=True)

    # --- STAGE 2: Merging All Pre-Processed Feature Sets ---
    logging.info("\n--- STAGE 2: Merging Additional Feature Sets ---")
    logging.info(" -> Advanced weather features already present in master dataset. Skipping redundant merge.")

    df_satellite = pd.read_csv(satellite_features_file)
    df_satellite['district_no'] = df_satellite['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, df_satellite, on=['district_no', 'year'], how='left')
    logging.info(" -> Satellite data merged.")
    merged_df['has_satellite_data'] = (merged_df['year'] >= 2001).astype(int)
    satellite_cols = [
        'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
        'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days'
    ]
    merged_df[satellite_cols] = merged_df[satellite_cols].fillna(0)
    logging.info(" -> Satellite features successfully integrated.")

    # --- STAGE 3: Final Feature Engineering & Cleanup ---
    logging.info("\n--- STAGE 3: Engineering Final Features and Cleaning ---")

    merged_df['year_trend'] = merged_df['year'] - merged_df['year'].min()

    logging.info("Creating lagged features for forecasting...")
    merged_df = merged_df.sort_values(by=['district_no', 'year'])
    merged_df['national_avg_yield'] = merged_df.groupby('year')['kreisYield'].transform('mean')
    merged_df['national_avg_yield_lag1'] = merged_df.groupby('district_no')['national_avg_yield'].shift(1)
    merged_df['producer_price_index_lag1'] = merged_df.groupby('district_no')['producer_price_index'].shift(1)
    merged_df['seed_price_index_lag1'] = merged_df.groupby('district_no')['seed_price_index'].shift(1)
    merged_df['energy_price_index_lag1'] = merged_df.groupby('district_no')['energy_price_index'].shift(1)
    merged_df['fertilizer_price_index_lag1'] = merged_df.groupby('district_no')['fertilizer_price_index'].shift(1)
    merged_df['plant_protection_price_index_lag1'] = merged_df.groupby('district_no')[
        'plant_protection_price_index'].shift(1)

    epsilon = 1e-6
    merged_df['profit_margin_proxy_lag1'] = merged_df['producer_price_index_lag1'] / (
                merged_df['fertilizer_price_index_lag1'] + epsilon)
    merged_df['cost_of_inputs_lag1'] = merged_df['fertilizer_price_index_lag1'] + merged_df[
        'plant_protection_price_index_lag1'] + merged_df['seed_price_index_lag1']
    logging.info(" -> Lagged features (lag1) created.")

    logging.info("Detrending economic features to create stable anomalies (using causal mean)...")
    economic_features_to_detrend = ['producer_price_index_lag1', 'seed_price_index_lag1', 'energy_price_index_lag1',
                                    'fertilizer_price_index_lag1', 'plant_protection_price_index_lag1']
    for feature in economic_features_to_detrend:
        trend = merged_df.groupby('district_no')[feature].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        trend = trend.fillna(method='ffill').fillna(method='bfill')
        merged_df[f'{feature}_anomaly'] = merged_df[feature] - trend
    logging.info(" -> Economic feature anomalies created correctly.")


    # <--- NEW ADVANCED FEATURE ENGINEERING BLOCK STARTS HERE --->
    logging.info("Engineering advanced features based on model analysis...")

    logging.info(" -> Capping extreme values for fertilizer price using a leak-proof expanding window...")

    def expanding_quantile(s, q):
        return s.expanding(min_periods=20).quantile(q)

    lower_bound_series = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: expanding_quantile(s, q=0.05)
    )
    upper_bound_series = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: expanding_quantile(s, q=0.95)
    )

    lower_bound_series = lower_bound_series.fillna(method='ffill').fillna(method='bfill')
    upper_bound_series = upper_bound_series.fillna(method='ffill').fillna(method='bfill')

    merged_df['fertilizer_price_index_lag1_anomaly_capped'] = merged_df['fertilizer_price_index_lag1_anomaly'].clip(
        lower=lower_bound_series, upper=upper_bound_series
    )

    merged_df['is_fertilizer_price_extreme'] = np.where(
        (merged_df['fertilizer_price_index_lag1_anomaly'] < lower_bound_series) |
        (merged_df['fertilizer_price_index_lag1_anomaly'] > upper_bound_series), 1, 0
    )

    logging.info(" -> Creating threshold-based weather features...")
    merged_df['is_summer_forecast_dry'] = np.where(merged_df['summer_precip_prob_wet_forecast'] < 0.3, 1, 0)

    logging.info(" -> Engineering explicit interactions and polynomials...")
    merged_df['gdd_x_fertilizer_price'] = merged_df['antecedent_gdd_sum_anomaly'] * merged_df[
        'fertilizer_price_index_lag1_anomaly_capped']
    merged_df['spring_temp_x_spring_precip'] = merged_df['spring_temp_anomaly_forecast'] * merged_df[
        'spring_precip_anomaly_forecast']
    merged_df['antecedent_gdd_sum_anomaly_sq'] = merged_df['antecedent_gdd_sum_anomaly'] ** 2

    logging.info(" -> Advanced features created successfully.")
    # <--- NEW ADVANCED FEATURE ENGINEERING BLOCK ENDS HERE --->

    logging.info("Creating high-impact interaction features using correct seasonal forecast names...")
    merged_df['summer_heat_x_profit_margin'] = merged_df['summer_temp_prob_warm_forecast'] * merged_df[
        'profit_margin_proxy_lag1']
    merged_df['summer_precip_x_input_costs'] = merged_df['summer_precip_prob_wet_forecast'] * merged_df[
        'cost_of_inputs_lag1']
    logging.info(" -> Interaction features created.")

    logging.info("Creating polynomial features to capture non-linear extreme effects...")
    critical_weather_features = [
        'spring_temp_prob_warm_forecast',
        'summer_temp_prob_warm_forecast',
        'spring_precip_prob_wet_forecast',
        'summer_precip_prob_wet_forecast'
    ]
    for feature in critical_weather_features:
        merged_df[f'{feature}_sq'] = merged_df[feature] ** 2
    logging.info(" -> Polynomial features created.")

    merged_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

    # ============================ IMPROVEMENT: FINAL COLUMN CLEANUP ============================
    # Explicitly remove intermediate columns and any raw current-year data that would
    # constitute a data leak if used accidentally as a feature. We only want to keep
    # the lagged (_lag1) and anomaly versions, which are safe for forecasting.
    cols_to_remove = [
        'yield',
        'state_name',
        'national_avg_yield',
        'producer_price_index',
        'seed_price_index',
        'energy_price_index',
        'fertilizer_price_index',
        'plant_protection_price_index'
    ]
    merged_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    logging.info(f" -> Dropped intermediate and data-leaking columns to finalize dataset.")
    # ===========================================================================================


    initial_rows = len(merged_df)
    merged_df.dropna(inplace=True)
    rows_dropped = initial_rows - len(merged_df)
    if rows_dropped > 0:
        logging.info(f"Dropped {rows_dropped} rows with missing values (expected for early years with NaNs).")

    logging.info(f"Saving Stage 1 model-ready dataset to '{output_file}'...")
    merged_df.to_csv(output_file, index=False, float_format='%.6f')

    logging.info(f"\n--- SUCCESS: Stage 1 pre-season forecast dataset created! ---")
    logging.info(f"Dataset saved to '{output_file}' with {merged_df.shape[0]} rows.")
    logging.info(f"Final columns ({len(merged_df.columns)} total): {merged_df.columns.tolist()}")


if __name__ == '__main__':
    main()

