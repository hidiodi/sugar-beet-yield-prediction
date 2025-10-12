# File: build_stage1_features.py
# Description: MODIFIED to create features for a STAGE 1 (pre-season) forecast model.
# Creates lagged features and only uses data available at the end of March.
# *** FINAL VERSION: Integrates satellite features AND engineers evolutionary trend features. ***

import pandas as pd
import xarray as xr
import logging
from pathlib import Path
import numpy as np
import dask
from dask.diagnostics import progress
from dask.distributed import Client, LocalCluster
import sys
import os
import time

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==============================================================================
# HELPER FUNCTIONS (No changes here)
# ==============================================================================

def load_and_process_economic_data(producer_price_file, input_price_file):
    """
    Loads, processes, and merges the separate economic data files.
    """
    logging.info("Loading and processing external economic data sources...")
    try:
        # --- Producer Prices (Annual Data) ---
        df_prod_raw = pd.read_csv(producer_price_file)
        df_prod = df_prod_raw[df_prod_raw['ID'] == 'LWPR-132'].melt(
            id_vars=['ID', 'Description'], var_name='year', value_name='producer_price_index'
        )
        df_prod['year'] = pd.to_numeric(df_prod['year'])
        df_prod = df_prod[['year', 'producer_price_index']]

        # --- Input Prices (Quarterly Data) ---
        df_input_raw = pd.read_csv(input_price_file)
        df_input = df_input_raw[df_input_raw['ID'].isin(['LWBM-13', 'LWBM-12'])]
        df_input_melted = df_input.melt(
            id_vars=['ID', 'Description'], var_name='period', value_name='price_index'
        )
        df_input_melted['price_index'] = pd.to_numeric(df_input_melted['price_index'], errors='coerce')
        df_input_melted['year'] = pd.to_numeric(df_input_melted['period'].str.split('/').str[1], errors='coerce')
        df_input_melted.dropna(subset=['year'], inplace=True)
        df_input_melted['year'] = df_input_melted['year'].astype(int)
        df_annual_avg = df_input_melted.groupby(['year', 'ID'])['price_index'].mean().reset_index()
        df_input_final = df_annual_avg.pivot(index='year', columns='ID', values='price_index').reset_index()
        df_input_final.rename(columns={
            'LWBM-13': 'fertilizer_price_index',
            'LWBM-12': 'energy_price_index'
        }, inplace=True)

        # --- Merge economic datasets ---
        df_economic = pd.merge(df_prod, df_input_final, on='year', how='outer')
        logging.info(" -> Economic datasets successfully merged.")
        return df_economic

    except Exception as e:
        logging.error(f"Failed to load or process economic data files. Details: {e}", exc_info=True)
        return None


def engineer_antecedent_weather(df, agera5_path):
    """
    Calculates and merges antecedent winter weather anomalies using optimized Xarray.
    """
    logging.info("Starting optimized antecedent winter weather feature engineering...")
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logging.error("'latitude' or 'longitude' not found. Cannot calculate weather features.")
        return df

    client = None
    try:
        client = Client(n_workers=2, threads_per_worker=4, memory_limit='8GB')
        logging.info(f"Dask Dashboard link: {client.dashboard_link}")
    except Exception as e:
        logging.warning(f"Could not start Dask client. Proceeding without explicit workers. Error: {e}")

    try:
        agera5_ds = xr.open_dataset(agera5_path, chunks={'time': 500, 'lat': 40, 'lon': 50})
        temp_var, precip_var = 'Temperature_Air_2m_Mean_24h', 'Precipitation_Flux'

        def get_winter_year(time):
            return np.where(time.dt.month >= 10, time.dt.year + 1, time.dt.year)

        agera5_ds = agera5_ds.assign_coords(winter_year=('time', get_winter_year(agera5_ds.time)))
        winter_ds = agera5_ds.sel(time=agera5_ds.time.dt.month.isin([10, 11, 12, 1, 2, 3]))

        with progress.ProgressBar():
            winter_temp_agg = winter_ds[temp_var].groupby('winter_year').mean(dim='time')
            winter_precip_agg = winter_ds[precip_var].groupby('winter_year').sum(dim='time')
            winter_stats = xr.Dataset({
                'winter_temp_mean_grid': winter_temp_agg,
                'winter_precip_sum_grid': winter_precip_agg
            }).compute()

        unique_locations = df[['district_no', 'latitude', 'longitude']].dropna().drop_duplicates().set_index(
            'district_no')
        location_coords = xr.Dataset.from_dataframe(unique_locations).rename({'latitude': 'lat', 'longitude': 'lon'})

        actual_temp_at_locs = winter_stats['winter_temp_mean_grid'].interp(
            lon=location_coords.lon, lat=location_coords.lat, method="linear"
        ).compute()
        actual_precip_at_locs = winter_stats['winter_precip_sum_grid'].interp(
            lon=location_coords.lon, lat=location_coords.lat, method="linear"
        ).compute()

        winter_temp_normal_locs = actual_temp_at_locs.mean(dim='winter_year')
        winter_precip_normal_locs = actual_precip_at_locs.mean(dim='winter_year')
        temp_anomaly_ds = actual_temp_at_locs - winter_temp_normal_locs
        precip_anomaly_ds = (actual_precip_at_locs / (winter_precip_normal_locs + 1e-6)) - 1.0

        anomaly_ds = xr.Dataset({
            'winter_temp_anomaly': temp_anomaly_ds,
            'winter_precip_anomaly': precip_anomaly_ds,
        })
        df_weather_features = anomaly_ds.to_dataframe().reset_index().rename(columns={'winter_year': 'year'})
        df_weather_features['district_no'] = df_weather_features['district_no'].astype(str).str.zfill(5)
        return pd.merge(df, df_weather_features, on=['year', 'district_no'], how='left')
    except Exception as e:
        logging.error(f"An error occurred during weather feature engineering: {e}", exc_info=True)
        return df
    finally:
        if client: client.close()


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    """Main function to build features for a Stage 1 (pre-season) forecast model."""
    logging.info("--- Starting Stage 1 (Pre-Season Forecast) Data Pipeline ---")

    # --- Configuration ---
    master_file = Path('data/04_master/master_dataset.csv')
    producer_price_file = Path('data/01_raw/61211-0002_de/61211-0001_de.csv')
    input_price_file = Path('data/01_raw/61211-0002_de/61221-0003_de.csv')
    agera5_file = Path('data/02_intermediate/agera5_germany_merged.nc')
    satellite_features_file = Path('data/03_primary/satellite_features_districts_2001-2021.csv')
    output_path = Path('data/05_model_input/')
    output_file = output_path / 'stage1_preseason_features.csv'
    output_path.mkdir(exist_ok=True, parents=True)

    # ==============================================================================
    # STAGE 1: Data Loading and Initial Transformation
    # ==============================================================================
    logging.info("\n--- STAGE 1: Loading and Merging Base Data ---")
    master_df = pd.read_csv(master_file)
    master_df['district_no'] = master_df['district_no'].astype(str).str.zfill(5)
    logging.info(f"Successfully loaded '{master_file}'.")
    df_economic = load_and_process_economic_data(producer_price_file, input_price_file)
    if df_economic is None: sys.exit(1)
    merged_df = pd.merge(master_df, df_economic, on='year', how='left')
    logging.info("Economic data successfully merged.")
    merged_df['kreisYield'] = pd.to_numeric(merged_df['yield'], errors='coerce') * 10
    initial_rows = len(merged_df)
    merged_df.dropna(subset=['kreisYield'], inplace=True)
    if (rows_dropped := initial_rows - len(merged_df)) > 0:
        logging.info(f"Dropped {rows_dropped} rows where 'kreisYield' (target variable) was missing or invalid.")

    # ==============================================================================
    # STAGE 1.5: Integrating Satellite Features
    # ==============================================================================
    logging.info("\n--- STAGE 1.5: Merging Satellite Features ---")
    if not satellite_features_file.exists():
        logging.error(f"FATAL: Satellite features file not found at '{satellite_features_file}'. Please run the GEE script first.")
        sys.exit(1)

    logging.info(f"Loading satellite features from '{satellite_features_file}'...")
    df_satellite = pd.read_csv(satellite_features_file)
    df_satellite['district_no'] = df_satellite['district_no'].astype(str).str.zfill(5)
    merged_df = pd.merge(merged_df, df_satellite, on=['district_no', 'year'], how='left')
    logging.info("Satellite data merged. Applying 'Hybrid Feature with Flag' method...")

    merged_df['has_satellite_data'] = (merged_df['year'] >= 2001).astype(int)
    satellite_cols = [
        'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
        'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days'
    ]
    merged_df[satellite_cols] = merged_df[satellite_cols].fillna(0)
    logging.info(" -> Satellite features successfully integrated.")


    # ==============================================================================
    # STAGE 2: Feature Engineering for Pre-Season
    # ==============================================================================
    logging.info("\n--- STAGE 2: Engineering Pre-Season Features ---")

    # Engineer winter weather (available by March)
    df_featured = engineer_antecedent_weather(merged_df, agera5_file)
    logging.info(f"Antecedent weather features engineered. Shape is now: {df_featured.shape}")

    # === NEW: Engineering features for structural agricultural change ===
    logging.info("Engineering features for long-term agricultural trends...")
    # 1. Continuous time trend for technology and genetic improvement
    df_featured['year_trend'] = df_featured['year'] - df_featured['year'].min()
    # 2. Flag for the post-2017 EU sugar quota abolition economic shock
    df_featured['post_quota_era'] = (df_featured['year'] >= 2017).astype(int)
    logging.info(" -> Trend and economic shock features created.")

    # --- Create Lagged Features (last year's data) ---
    logging.info("Creating lagged features for forecasting...")
    df_featured = df_featured.sort_values(by=['district_no', 'year'])
    df_featured['national_avg_yield'] = df_featured.groupby('year')['kreisYield'].transform('mean')
    df_featured['national_avg_yield_lag1'] = df_featured.groupby('district_no')['national_avg_yield'].shift(1)
    df_featured['producer_price_index_lag1'] = df_featured.groupby('district_no')['producer_price_index'].shift(1)
    logging.info(" -> Lagged features (lag1) created.")

    # ==============================================================================
    # STAGE 3: Final Cleanup, Imputation, and Saving
    # ==============================================================================
    logging.info("\n--- STAGE 3: Finalizing Dataset ---")

    logging.info("Dropping columns that are unknown at the time of a pre-season forecast...")
    cols_to_remove = [
        'yield', 'state_name', 'latitude', 'longitude',
    ]
    df_featured.drop(columns=cols_to_remove, inplace=True, errors='ignore')

    initial_rows = len(df_featured)
    df_featured.dropna(inplace=True)
    logging.info(
        f"Dropped {initial_rows - len(df_featured)} rows due to lagging operation (first year of data for each district).")

    # --- Final Imputation for any remaining missing values ---
    logging.info("Performing final imputation on remaining numeric columns...")
    numeric_cols = df_featured.select_dtypes(include=np.number).columns.tolist()
    df_imputed = df_featured.copy()
    df_imputed[numeric_cols] = df_imputed.groupby('district_no')[numeric_cols].transform(lambda x: x.fillna(x.mean()))
    df_imputed.fillna(df_imputed.mean(numeric_only=True), inplace=True)

    # --- Save Final Dataset ---
    logging.info(f"Saving Stage 1 model-ready dataset to '{output_file}'...")
    df_imputed.to_csv(output_file, index=False, float_format='%.6f')

    logging.info(f"\n-------------------------------------------------")
    logging.info(f"--- SUCCESS: Stage 1 pre-season forecast dataset created! ---")
    logging.info(f"Dataset saved to '{output_file}' with {df_imputed.shape[0]} rows.")
    logging.info(f"Final columns ({len(df_imputed.columns)} total): {df_imputed.columns.tolist()}")
    logging.info(f"-------------------------------------------------")

if __name__ == '__main__':
    main()