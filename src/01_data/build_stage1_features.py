import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_process_economic_data(producer_price_file, input_price_file):
    """
    Uses a granular and curated set of price indices relevant to sugar beets.
    """
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
            'LWBM-11': 'seed_price_index',
            'LWBM-12': 'energy_price_index',
            'LWBM-13': 'fertilizer_price_index',
            'LWBM-14': 'plant_protection_price_index'
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
    """Main function to build features for a Stage 1 (pre-season) forecast model."""
    logging.info("Starting Stage 1 (Pre-Season Forecast) Data Pipeline")

    master_file = Path('data/04_master/master_dataset.csv')
    producer_price_file = Path('data/01_raw/Bundesdatenbank/61211-0001_de.csv')
    input_price_file = Path('data/01_raw/Bundesdatenbank/61221-0003_de.csv')
    satellite_features_file = Path('data/03_primary/satellite_features_districts_2001-2024.csv')
    output_path = Path('data/05_model_input/')

    # --- CHANGE 1: DEFINE WOFOST PATH & DYNAMICALLY FIND THE LATEST FILE ---
    # Explanation: Instead of hardcoding a filename that changes, this logic
    # automatically finds the most recent simulation output file. This makes
    # the pipeline robust and removes the need for manual code edits.
    wofost_output_dir = Path('data/06_model_output/multi_year_final/')
    try:
        wofost_files = list(wofost_output_dir.glob('final_comparison_*.csv'))
        if not wofost_files: raise FileNotFoundError
        wofost_output_file = max(wofost_files, key=lambda p: p.stat().st_mtime)
        logging.info(f"Dynamically found WOFOST output file: {wofost_output_file}")
    except FileNotFoundError:
        logging.error(f"FATAL: No 'final_comparison_*.csv' file found in {wofost_output_dir}. Cannot proceed.")
        sys.exit(1)
    # --- END OF CHANGE 1 ---

    output_file = output_path / 'stage1_preseason_features.csv'
    output_path.mkdir(exist_ok=True, parents=True)

    logging.info("STAGE 1: Loading and Merging Base Data")
    master_df = pd.read_csv(master_file)
    master_df['district_no'] = master_df['district_no'].astype(str).str.zfill(5)

    if 'crs_anomaly' in master_df.columns:
        logging.info("Removing 'crs_anomaly' column from master data.")
        master_df = master_df.drop(columns=['crs_anomaly'])

    df_economic = load_and_process_economic_data(producer_price_file, input_price_file)
    if df_economic is None: sys.exit(1)

    overlapping_cols = [col for col in df_economic.columns if col in master_df.columns and col != 'year']
    if overlapping_cols:
        master_df = master_df.drop(columns=overlapping_cols)

    merged_df = pd.merge(master_df, df_economic, on='year', how='left')
    merged_df['kreisYield'] = pd.to_numeric(merged_df['yield'], errors='coerce')
    merged_df.dropna(subset=['kreisYield'], inplace=True)

    logging.info("STAGE 2: Merging Additional Feature Sets")
    logging.info("Advanced weather features already present in master dataset. Skipping redundant merge.")

    df_satellite = pd.read_csv(satellite_features_file)
    df_satellite['district_no'] = df_satellite['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, df_satellite, on=['district_no', 'year'], how='left')
    logging.info("Satellite data merged.")
    merged_df['has_satellite_data'] = (merged_df['year'] >= 2001).astype(int)
    satellite_cols = [
        'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
        'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days'
    ]
    merged_df[satellite_cols] = merged_df[satellite_cols].fillna(0)
    logging.info("Satellite features successfully integrated.")

    # --- CHANGE 2: MERGE WOFOST SIMULATION OUTPUTS ---
    # Explanation: This new block loads the WOFOST simulation results and merges
    # them into the main dataframe. We carefully select only the pre-season
    # forecast features to avoid any data leakage.
    logging.info(f"Merging WOFOST biophysical simulation outputs from {wofost_output_file.name}...")
    try:
        df_wofost = pd.read_csv(wofost_output_file)
        df_wofost['district_no'] = df_wofost['district_no'].astype(str).str.zfill(5)

        wofost_features_to_add = [
            'year', 'district_no',
            'forecast_yield_dry_kgha',
            'forecast_uncertainty_std'
        ]

        merged_df = pd.merge(merged_df, df_wofost[wofost_features_to_add], on=['year', 'district_no'], how='left')
        logging.info("✓ WOFOST simulation features successfully merged.")

    except Exception as e:
        logging.error(f"Failed to load or merge WOFOST data. Error: {e}", exc_info=True)
        merged_df['forecast_yield_dry_kgha'] = np.nan
        merged_df['forecast_uncertainty_std'] = np.nan
    # --- END OF CHANGE 2 ---

    merged_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

    logging.info("STAGE 3: Engineering Final Features and Cleaning")

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
    logging.info("Lagged features (lag1) created.")

    # --- CHANGE 3: PROCESS MERGED WOFOST DATA INTO MODEL-READY FEATURES ---
    # Explanation: This new block converts the raw WOFOST outputs into valuable
    # features. It creates the main forecast feature in the correct units (fresh
    # dt/ha) and engineers new interaction features based on our analysis to help
    # the model perform better in extreme years. NaN handling makes it robust.
    logging.info("Processing merged WOFOST features...")
    DMC_SUGARBEET = 0.25
    KG_PER_DT = 100

    # 1. Create the primary feature: WOFOST forecast in fresh weight dt/ha
    merged_df['wofost_forecast_yield_fresh_dt'] = merged_df['forecast_yield_dry_kgha'] / (DMC_SUGARBEET * KG_PER_DT)

    # 2. Engineer interaction features to cover XGBoost's blind spots
    merged_df['wofost_forecast_x_profit_margin'] = merged_df['wofost_forecast_yield_fresh_dt'] * merged_df[
        'profit_margin_proxy_lag1']
    merged_df['wofost_uncertainty_x_sandy_soil'] = merged_df['forecast_uncertainty_std'] * merged_df['avg_sand_0_30cm']

    # 3. Handle missing values and create an indicator
    wofost_cols_to_fill = [
        'wofost_forecast_yield_fresh_dt',
        'wofost_forecast_x_profit_margin',
        'wofost_uncertainty_x_sandy_soil',
        'forecast_uncertainty_std'
    ]
    merged_df['has_wofost_data'] = merged_df['wofost_forecast_yield_fresh_dt'].notna().astype(int)
    for col in wofost_cols_to_fill:
        merged_df[col] = merged_df[col].fillna(0)

    logging.info("✓ Engineered new features from WOFOST outputs.")
    # --- END OF CHANGE 3 ---

    logging.info("Detrending economic features to create stable anomalies (using causal mean)...")
    economic_features_to_detrend = ['producer_price_index_lag1', 'seed_price_index_lag1', 'energy_price_index_lag1',
                                    'fertilizer_price_index_lag1', 'plant_protection_price_index_lag1']
    for feature in economic_features_to_detrend:
        trend = merged_df.groupby('district_no')[feature].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        trend = trend.ffill().bfill()
        merged_df[f'{feature}_anomaly'] = merged_df[feature] - trend
    logging.info("Economic feature anomalies created correctly.")

    logging.info("Engineering advanced features based on model analysis...")
    logging.info("Capping extreme values for fertilizer price using a leak-proof expanding window...")

    def expanding_quantile(s, q):
        return s.expanding(min_periods=20).quantile(q)

    lower_bound_series = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: expanding_quantile(s, q=0.05)
    )
    upper_bound_series = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: expanding_quantile(s, q=0.95)
    )

    lower_bound_series = lower_bound_series.ffill().bfill()
    upper_bound_series = upper_bound_series.ffill().bfill()

    merged_df['fertilizer_price_index_lag1_anomaly_capped'] = merged_df['fertilizer_price_index_lag1_anomaly'].clip(
        lower=lower_bound_series, upper=upper_bound_series
    )

    merged_df['is_fertilizer_price_extreme'] = np.where(
        (merged_df['fertilizer_price_index_lag1_anomaly'] < lower_bound_series) |
        (merged_df['fertilizer_price_index_lag1_anomaly'] > upper_bound_series), 1, 0
    )

    logging.info("Creating threshold-based weather features...")
    merged_df['is_summer_forecast_dry'] = np.where(merged_df['summer_precip_prob_wet_forecast'] < 0.3, 1, 0)

    logging.info("Engineering explicit interactions and polynomials...")
    merged_df['gdd_x_fertilizer_price'] = merged_df['antecedent_gdd_sum_anomaly'] * merged_df[
        'fertilizer_price_index_lag1_anomaly_capped']
    merged_df['spring_temp_x_spring_precip'] = merged_df['spring_temp_anomaly_forecast'] * merged_df[
        'spring_precip_anomaly_forecast']
    merged_df['antecedent_gdd_sum_anomaly_sq'] = merged_df['antecedent_gdd_sum_anomaly'] ** 2

    logging.info("Advanced features created successfully.")

    logging.info("Creating high-impact interaction features using correct seasonal forecast names...")
    merged_df['summer_heat_x_profit_margin'] = merged_df['summer_temp_prob_warm_forecast'] * merged_df[
        'profit_margin_proxy_lag1']
    merged_df['summer_precip_x_input_costs'] = merged_df['summer_precip_prob_wet_forecast'] * merged_df[
        'cost_of_inputs_lag1']

    merged_df['summer_precip_anomaly_forecast_sq'] = merged_df['summer_precip_anomaly_forecast'] ** 2

    logging.info("Interaction features created.")



    logging.info("Engineering features to explicitly capture extreme weather forecast signals...")

    EXTREME_HOT_Q = 0.95
    EXTREME_DRY_Q = 0.05

    hot_threshold = merged_df.groupby('district_no')['summer_temp_anomaly_forecast'].transform(
        lambda s: s.expanding(min_periods=20).quantile(EXTREME_HOT_Q)
    )
    dry_threshold = merged_df.groupby('district_no')['summer_precip_anomaly_forecast'].transform(
        lambda s: s.expanding(min_periods=20).quantile(EXTREME_DRY_Q)
    )

    hot_threshold = hot_threshold.ffill().bfill()
    dry_threshold = dry_threshold.ffill().bfill()

    merged_df['is_extreme_heat_forecast'] = np.where(merged_df['summer_temp_anomaly_forecast'] > hot_threshold, 1, 0)
    merged_df['is_extreme_drought_forecast'] = np.where(merged_df['summer_precip_anomaly_forecast'] < dry_threshold, 1,
                                                        0)

    merged_df['drought_x_heat'] = merged_df['is_extreme_heat_forecast'] * merged_df['is_extreme_drought_forecast']

    logging.info(f"Created {merged_df['drought_x_heat'].sum()} instances of extreme 'drought_x_heat' events.")

    # <--- NEW ADVANCED FEATURE ENGINEERING V2 BLOCK STARTS HERE --->
    logging.info("Engineering V2 advanced interaction features...")

    # --- 1. Magnitude-Aware Extreme Weather Interactions ---
    # This is a direct improvement on the ineffective 'drought_x_heat' binary feature.
    # It preserves the magnitude of both anomalies. A hot and dry forecast will result in a large positive value.
    # We multiply by -1 on precip because a large negative anomaly (drought) should increase the interaction term.
    merged_df['hot_dry_interaction'] = merged_df['summer_temp_anomaly_forecast'] * (
                merged_df['summer_precip_anomaly_forecast'] * -1)

    # --- 2. Geographic & Soil-Weather Interactions ---
    # Allows the model to learn that weather impacts differ by region.
    # For example, a +2C anomaly is more dangerous in the warmer south (higher latitude value in this dataset)
    # A drought is more severe in sandy soils (low water retention) than in clay soils.
    merged_df['lat_x_summer_temp'] = merged_df['lat'] * merged_df['summer_temp_anomaly_forecast']
    merged_df['lon_x_spring_precip'] = merged_df['lon'] * merged_df['spring_precip_anomaly_forecast']
    merged_df['sandy_soil_x_drought'] = merged_df['avg_sand_0_30cm'] * merged_df['summer_precip_anomaly_forecast']

    # --- 3. Economic Momentum Features ---
    # Captures the year-over-year change in economic conditions. A rapid increase in costs or prices
    # may influence farmer decisions more than the absolute level.
    merged_df['profit_margin_proxy_lag2'] = merged_df.groupby('district_no')['profit_margin_proxy_lag1'].shift(1)
    merged_df['cost_of_inputs_lag2'] = merged_df.groupby('district_no')['cost_of_inputs_lag1'].shift(1)

    merged_df['profit_margin_momentum'] = merged_df['profit_margin_proxy_lag1'] - merged_df['profit_margin_proxy_lag2']
    merged_df['cost_of_inputs_momentum'] = merged_df['cost_of_inputs_lag1'] - merged_df['cost_of_inputs_lag2']

    # Clean up intermediate columns used for momentum calculation
    merged_df.drop(columns=['profit_margin_proxy_lag2', 'cost_of_inputs_lag2'], inplace=True)
    logging.info(" -> V2 advanced interaction features created successfully.")
    # <--- NEW ADVANCED FEATURE ENGINEERING V2 BLOCK ENDS HERE --->

    logging.info("Engineering polynomial features to capture non-linear effects...")
    prob_features_poly = [
        'spring_temp_prob_warm_forecast',
        'summer_temp_prob_warm_forecast',
        'spring_precip_prob_wet_forecast',
        'summer_precip_prob_wet_forecast'
    ]
    for feature in prob_features_poly:
        merged_df[f'{feature}_sq'] = merged_df[feature] ** 2
    logging.info("Created squared probability features.")

    anomaly_features_poly = [
        'summer_temp_anomaly_forecast',
        'summer_precip_anomaly_forecast',
        'antecedent_gdd_sum_anomaly'
    ]
    for feature in anomaly_features_poly:
        merged_df[f'{feature}_sq'] = merged_df[feature] ** 2
        merged_df[f'{feature}_cubed'] = merged_df[feature] ** 3
    logging.info("Created squared & cubed anomaly features for extreme event modeling.")

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
    logging.info(f"Dropped intermediate and data-leaking columns to finalize dataset.")

    '''
    initial_rows = len(merged_df)
    merged_df.dropna(inplace=True)
    rows_dropped = initial_rows - len(merged_df)
    if rows_dropped > 0:
        logging.info(f"Dropped {rows_dropped} rows with missing values (expected for early years with NaNs).")
    '''

    logging.info(f"Saving Stage 1 model-ready dataset to '{output_file}'...")
    merged_df.to_csv(output_file, index=False, float_format='%.6f')

    logging.info(f"Stage 1 pre-season forecast dataset created!")
    logging.info(f"Dataset saved to '{output_file}' with {merged_df.shape[0]} rows.")
    logging.info(f"Final columns ({len(merged_df.columns)} total): {merged_df.columns.tolist()}")


if __name__ == '__main__':
    main()