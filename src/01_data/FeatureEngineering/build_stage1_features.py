# File: src/01_data/FeatureEngineering/build_stage1_features.py
# MODIFIED (v4.0): Overhauled feature generation to incorporate monthly forecast
#                  granularity, solar radiation, national yield trends, and
#                  new interaction terms.

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


def create_hybrid_antecedent_and_forecast_features(
        weather_dir: Path,
        ecmwf_by_member_forecast_path: Path,
        start_year: int,
        end_year: int,
        forecast_month: int = 3
):
    """
    Constructs a robust, dataleak-proof feature set by combining:
    1. Observed antecedent weather (AGERA5).
    2. Distributional and probabilistic features from the full ECMWF ensemble,
       including monthly, seasonal, and solar radiation metrics.
    """
    logging.info("--- Starting Hybrid Antecedent (Observed) + ENSEMBLE Forecast Feature Engineering ---")

    # --- PART 1: Engineer Antecedent Features from AGERA5 (Observed History) ---
    logging.info(f"Loading AGERA5 data from {start_year - 1} to {end_year} for antecedent features...")
    all_weather_files = [weather_dir / f"historical_daily_weather_era5_{y}.csv" for y in
                         range(start_year - 1, end_year + 1)]
    df_daily = pd.concat([pd.read_csv(f) for f in all_weather_files if f.exists()], ignore_index=True)
    df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['year'] = df_daily['date'].dt.year
    df_daily = df_daily.sort_values(by=['district_no', 'date'])
    df_daily['gdd'] = ((df_daily['tmax'] + df_daily['tmin']) / 2 - 5).clip(lower=0)
    df_normals_30yr = df_daily[df_daily['year'].between(1981, 2010)].copy()
    daily_normals = df_normals_30yr.groupby(df_normals_30yr['date'].dt.dayofyear)['gdd'].mean().rename('gdd_30yr_avg')
    df_daily = pd.merge(df_daily, daily_normals, left_on=df_daily['date'].dt.dayofyear, right_index=True, how='left')
    df_daily['gdd_anomaly'] = df_daily['gdd'] - df_daily['gdd_30yr_avg']
    df_daily['forecast_year'] = df_daily['date'].apply(lambda d: d.year + 1 if d.month >= 10 else d.year)
    antecedent_window_df = df_daily[(df_daily['date'].dt.month <= forecast_month) | (df_daily['date'].dt.month >= 10)]

    logging.info("Aggregating leak-proof antecedent features...")
    antecedent_features = antecedent_window_df.groupby(['district_no', 'forecast_year']).agg(
        antecedent_precip_sum=('precip', 'sum'),
        antecedent_frost_days=('tmin', lambda x: (x < 0).sum()),
        antecedent_heavy_precip_days=('precip', lambda x: (x > 10).sum()),
        antecedent_gdd_sum_anomaly=('gdd_anomaly', 'sum')
    ).reset_index().rename(columns={'forecast_year': 'year'})
    logging.info(f"Generated {len(antecedent_features)} rows of antecedent features.")

    # --- PART 2: Engineer Distributional Features from ECMWF Ensemble Forecast ---
    logging.info(f"Loading ENSEMBLE forecast data from {ecmwf_by_member_forecast_path}...")
    df_fcst_member = pd.read_csv(ecmwf_by_member_forecast_path)
    df_fcst_member['district_no'] = df_fcst_member['district_no'].astype(str).str.zfill(5)

    # Define aggregation functions with explicit names to avoid lambda conflicts
    p10 = lambda x: x.quantile(0.10);
    p10.__name__ = 'p10'
    p90 = lambda x: x.quantile(0.90);
    p90.__name__ = 'p90'
    prob_hot = lambda x: (x > 1.5).mean();
    prob_hot.__name__ = 'prob_hot'
    prob_dry = lambda x: (x < -0.5).mean();
    prob_dry.__name__ = 'prob_dry'

    aggs = {
        # --- Retain Existing Seasonal Features ---
        'spring_temp_anomaly_forecast': ['mean', 'std', p10, p90],
        'summer_temp_anomaly_forecast': ['mean', 'std', p10, p90, prob_hot],
        'spring_precip_anomaly_forecast': ['mean', 'std'],
        'summer_precip_anomaly_forecast': ['mean', 'std', p10, p90, prob_dry],

        # --- NEW: Deconstruct Summer by Month ---
        'temp_anomaly_forecast_6': [p10],  # June
        'temp_anomaly_forecast_7': [p10],  # July
        'temp_anomaly_forecast_8': [p10],  # August

        # --- NEW: Add Solar Radiation Risk ---
        'summer_solar_rad_anomaly_forecast': ['mean', 'std', p10, p90]
    }

    # Filter for columns that actually exist in the input file
    valid_aggs = {k: v for k, v in aggs.items() if k in df_fcst_member.columns}
    logging.info(f"Aggregating distributional features for: {list(valid_aggs.keys())}")

    df_fcst_agg = df_fcst_member.groupby(['year', 'district_no']).agg(valid_aggs).reset_index()

    # Flatten the multi-level column index
    df_fcst_agg.columns = ['_'.join(col).strip() for col in df_fcst_agg.columns.values]

    # Clean up column names
    df_fcst_agg.rename(columns={
        'year_': 'year',
        'district_no_': 'district_no',
        'summer_temp_anomaly_forecast_prob_hot': 'prob_hot_summer',
        'summer_precip_anomaly_forecast_prob_dry': 'prob_dry_summer',
        'temp_anomaly_forecast_6_p10': 'june_temp_anomaly_p10',
        'temp_anomaly_forecast_7_p10': 'july_temp_anomaly_p10',
        'temp_anomaly_forecast_8_p10': 'august_temp_anomaly_p10'
    }, inplace=True)

    # Engineer new risk/interaction features from the aggregated distributions
    df_fcst_agg['forecast_uncertainty_summer_temp'] = df_fcst_agg['summer_temp_anomaly_forecast_std']
    df_fcst_agg['forecast_hot_dry_risk_p90'] = df_fcst_agg['summer_temp_anomaly_forecast_p90'].clip(lower=0) * \
                                               df_fcst_agg['summer_precip_anomaly_forecast_p10'].clip(upper=0) * -1

    # --- PART 3: Combine and Finalize ---
    logging.info("Merging observed antecedent features with ensemble forecast features...")
    final_df = pd.merge(antecedent_features, df_fcst_agg, on=['year', 'district_no'], how='inner')
    final_df = final_df[final_df['year'].between(start_year, end_year)]

    logging.info(f"✓ Hybrid weather feature set with ENSEMBLE STATS created. Final shape: {final_df.shape}")
    return final_df


def load_and_process_economic_data(producer_price_file, input_price_file):
    # This function is unchanged.
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
    logging.info("--- Starting Definitive Hybrid Feature Engineering Pipeline (v4.0) ---")

    file_paths = CONFIG['FILE_PATHS']
    output_path = file_paths['OUTPUT_DIR']
    output_file = file_paths['OUTPUT_FILE']
    output_path.mkdir(exist_ok=True, parents=True)

    walkforward_forecast_file = file_paths['WALKFORWARD_FORECAST_CSV']
    if not walkforward_forecast_file.exists():
        logging.error(f"FATAL: Required WOFOST file not found at {walkforward_forecast_file}. Cannot proceed.")
        sys.exit(1)

    logging.info("STAGE 1: Loading and Merging Base Data")
    master_df = pd.read_csv(file_paths['MASTER_DATASET'])
    master_df['district_no'] = master_df['district_no'].astype(str).str.zfill(5)

    # --- NEW: Add National Average Yield Lag ---
    logging.info("Calculating national average yield lag feature...")
    national_yield = master_df.groupby('year')['yield'].mean().reset_index()
    national_yield.rename(columns={'yield': 'national_avg_yield'}, inplace=True)
    national_yield['national_avg_yield_lag1'] = national_yield['national_avg_yield'].shift(1)
    master_df = pd.merge(master_df, national_yield[['year', 'national_avg_yield_lag1']], on='year', how='left')

    df_economic = load_and_process_economic_data(file_paths['PRODUCER_PRICE_CSV'], file_paths['INPUT_PRICE_CSV'])
    if df_economic is None: sys.exit(1)
    merged_df = pd.merge(master_df, df_economic, on='year', how='left')
    merged_df['kreisYield'] = pd.to_numeric(merged_df['yield'], errors='coerce')
    merged_df.dropna(subset=['kreisYield'], inplace=True)

    logging.info("STAGE 2: Augmenting with All Additional Feature Sets")

    df_weather_features = create_hybrid_antecedent_and_forecast_features(
        weather_dir=file_paths['DAILY_WEATHER_DIR'],
        ecmwf_by_member_forecast_path=file_paths['ECMWF_FORECAST_FEATURES_CSV'],
        start_year=CONFIG['WEATHER_FEATURE_YEAR_START'],
        end_year=CONFIG['WEATHER_FEATURE_YEAR_END']
    )
    df_weather_features['district_no'] = df_weather_features['district_no'].astype(str).str.zfill(5)

    # Handle potential column collisions before merging
    logging.info(f"DIAGNOSTIC: Columns in base data before weather merge: {merged_df.columns.tolist()}")
    logging.info(f"DIAGNOSTIC: Columns in new weather data: {df_weather_features.columns.tolist()}")

    cols_to_drop = [col for col in df_weather_features.columns if
                    col in merged_df.columns and col not in ['year', 'district_no']]
    if cols_to_drop:
        logging.warning(f"Dropping existing stale columns from base data to prevent merge conflicts: {cols_to_drop}")
        merged_df = merged_df.drop(columns=cols_to_drop)

    merged_df = pd.merge(merged_df, df_weather_features, on=['district_no', 'year'], how='left')
    logging.info("✓ Hybrid (antecedent + ENSEMBLE forecast) weather features merged.")

    df_satellite = pd.read_csv(file_paths['SATELLITE_FEATURES_CSV'])
    df_satellite['district_no'] = df_satellite['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, df_satellite, on=['district_no', 'year'], how='left')

    df_forecast = pd.read_csv(walkforward_forecast_file)
    df_forecast['district_no'] = df_forecast['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, df_forecast[['year', 'district_no', 'final_corrected_forecast']],
                         on=['year', 'district_no'], how='left')

    gdf = gpd.read_file(file_paths['GEOJSON_DISTRICTS'])
    gdf_states = gdf[['id', 'state']].rename(columns={'id': 'district_no'})
    gdf_states['district_no'] = gdf_states['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, gdf_states, on='district_no', how='left')
    merged_df['state_encoded'], _ = pd.factorize(merged_df['state'])

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

    lower_bound = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: s.expanding(min_periods=20).quantile(0.05)).ffill().bfill()
    upper_bound = merged_df.groupby('district_no')['fertilizer_price_index_lag1_anomaly'].transform(
        lambda s: s.expanding(min_periods=20).quantile(0.95)).ffill().bfill()
    merged_df['fertilizer_price_index_lag1_anomaly_capped'] = merged_df['fertilizer_price_index_lag1_anomaly'].clip(
        lower=lower_bound, upper=upper_bound)
    merged_df['is_fertilizer_price_extreme'] = np.where(
        (merged_df['fertilizer_price_index_lag1_anomaly'] < lower_bound) | (
                merged_df['fertilizer_price_index_lag1_anomaly'] > upper_bound), 1, 0)

    merged_df['gdd_x_fertilizer_price'] = merged_df['antecedent_gdd_sum_anomaly'] * merged_df[
        'fertilizer_price_index_lag1_anomaly_capped']

    # --- UPDATED & NEW Interaction Terms ---
    logging.info("Creating new and updated interaction features...")
    # Update to use new explicit column name for mean forecast
    merged_df['hot_dry_interaction'] = merged_df['summer_temp_anomaly_forecast_mean'] * (
            merged_df['summer_precip_anomaly_forecast_mean'] * -1)

    # NEW: Economic x Spring Weather Interaction
    merged_df['profit_margin_proxy_lag1_x_spring_temp_anomaly_mean'] = merged_df['profit_margin_proxy_lag1'] * \
                                                                       merged_df['spring_temp_anomaly_forecast_mean']

    # NEW: Antecedent x Spring Weather Interaction
    merged_df['antecedent_precip_sum_x_spring_temp_anomaly_mean'] = merged_df['antecedent_precip_sum'] * \
                                                                    merged_df['spring_temp_anomaly_forecast_mean']

    merged_df['summer_temp_forecast_range'] = merged_df['summer_temp_anomaly_forecast_p90'] - merged_df['summer_temp_anomaly_forecast_p10']
    merged_df['summer_precip_forecast_range'] = merged_df['summer_precip_anomaly_forecast_p90'] - merged_df['summer_precip_anomaly_forecast_p10']

    print("Created new features: 'summer_temp_forecast_range', 'summer_precip_forecast_range'")

    merged_df['enso_x_spring_temp_forecast'] = merged_df['enso_mei_winter_avg'] * merged_df['spring_temp_anomaly_forecast_mean']
    merged_df['sca_x_spring_temp_forecast'] = merged_df['sca_winter_avg'] * merged_df['spring_temp_anomaly_forecast_mean']

    print("Created new features: 'enso_x_spring_temp_forecast', 'sca_x_spring_temp_forecast'")

    output_file = file_paths['OUTPUT_FILE']
    merged_df.to_csv(output_file, index=False, float_format='%.6f')
    logging.info(f"Stage 1 dataset with ENSEMBLE features created and saved to '{output_file}'")
    logging.info(f"Final columns now include monthly temp risk, solar radiation, national trend, and new interactions.")
    logging.info(f"Final dataset shape: {merged_df.shape}")


if __name__ == '__main__':
    main()