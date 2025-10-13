import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import logging
import numpy as np

# --- Setup detailed, informative logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')

# --- 1. CONFIGURATION ---
BASE_DIR = Path.cwd()
SEAS5_INPUT_DIR = BASE_DIR / "data/01_raw/SEAS5_monthly_germany"
GEOJSON_PATH = BASE_DIR / "data/01_raw/districts_official.geojson"
OUTPUT_CSV_PATH = BASE_DIR / "data/02_intermediate/seas5_forecast_features_1979_2021.csv"

CLIMATOLOGY_YEARS = list(range(1993, 2017))
AVAILABLE_SEAS5_YEARS = list(range(1981, 2023))
FINAL_DATASET_YEARS = list(range(1979, 2022))


def calculate_climatology(seas5_dir, climatology_years):
    logging.info(f"*** STAGE 1: Calculating Climatology ({climatology_years[0]}-{climatology_years[-1]}) ***")
    files_to_load = [f for f in seas5_dir.glob("*.nc") if int(f.stem.split('_')[3]) in climatology_years]
    if not files_to_load:
        logging.critical("CRITICAL FAILURE: No NetCDF files found for climatology.")
        return None
    with xr.open_mfdataset(files_to_load, combine='nested', concat_dim='time', join='override') as ds:
        if 'tprate' in ds.data_vars: ds = ds.rename({'tprate': 'tp'})
        climatology = ds.mean(dim='time')
        if 'forecast_reference_time' in climatology.coords:
            climatology = climatology.drop_vars('forecast_reference_time')
        logging.info("  -> Climatology calculation successful.")
        return climatology.load()


def process_seas5_forecasts():
    climatology = calculate_climatology(SEAS5_INPUT_DIR, CLIMATOLOGY_YEARS)
    if climatology is None: return

    logging.info("*** STAGE 2: Preparing District Centroids for Lookup ***")
    districts_gdf = gpd.read_file(GEOJSON_PATH)
    districts_gdf = districts_gdf.to_crs("EPSG:4326")
    district_centroids = districts_gdf.centroid
    target_lons = xr.DataArray(district_centroids.x, dims=['district'], coords={'district': districts_gdf['id']})
    target_lats = xr.DataArray(district_centroids.y, dims=['district'], coords={'district': districts_gdf['id']})
    logging.info("  -> District centroids prepared for xarray lookup.")

    processed_features = []
    logging.info(f"*** STAGE 3: Processing Yearly Forecasts with Robust Lookup ***")
    for year in AVAILABLE_SEAS5_YEARS:
        logging.info(f"\n--- Processing Year: {year} ---")
        file_path = SEAS5_INPUT_DIR / f"seas5_monthly_germany_{year}_march_start.nc"
        if not file_path.exists():
            logging.warning(f"  -> WARNING: File not found for year {year}. Skipping.")
            continue

        with xr.open_dataset(file_path) as ds:
            if 'tprate' in ds.data_vars: ds = ds.rename({'tprate': 'tp'})
            if 'forecast_reference_time' in ds.coords:
                ds = ds.drop_vars('forecast_reference_time')

            aligned_ds, aligned_climatology = xr.align(ds, climatology, join='left')
            anomaly = aligned_ds.mean(dim='number') - aligned_climatology.mean(dim='number')
            probability = (aligned_ds > aligned_climatology).mean(dim='number')

            spring_anomaly = anomaly.sel(forecastMonth=slice(1, 3)).mean(dim='forecastMonth')
            summer_anomaly = anomaly.sel(forecastMonth=slice(4, 6)).mean(dim='forecastMonth')
            spring_prob = probability.sel(forecastMonth=slice(1, 3)).mean(dim='forecastMonth')
            summer_prob = probability.sel(forecastMonth=slice(4, 6)).mean(dim='forecastMonth')

            spring_anomaly_nearest = spring_anomaly.sel(longitude=target_lons, latitude=target_lats, method='nearest')
            summer_anomaly_nearest = summer_anomaly.sel(longitude=target_lons, latitude=target_lats, method='nearest')
            spring_prob_nearest = spring_prob.sel(longitude=target_lons, latitude=target_lats, method='nearest')
            summer_prob_nearest = summer_prob.sel(longitude=target_lons, latitude=target_lats, method='nearest')

            # ============================ THE FINAL FIX ============================
            # ### CRITICAL FIX: Reset the index after converting to DataFrame ###
            # This flattens the MultiIndex created by xarray and turns the
            # district ID and the unwanted 'number' coordinate into regular columns.
            df_spring_anomaly = spring_anomaly_nearest.to_dataframe().reset_index()
            df_summer_anomaly = summer_anomaly_nearest.to_dataframe().reset_index()
            df_spring_prob = spring_prob_nearest.to_dataframe().reset_index()
            df_summer_prob = summer_prob_nearest.to_dataframe().reset_index()

            # Now, we explicitly use the 'district' column which is guaranteed to be clean.
            year_df = pd.DataFrame({
                'year': year,
                'district_no': df_spring_anomaly['district'],
                'spring_temp_anomaly_forecast': df_spring_anomaly['t2m'],
                'spring_precip_anomaly_forecast': df_spring_anomaly['tp'] * 86400 * 1000,
                'summer_temp_anomaly_forecast': df_summer_anomaly['t2m'],
                'summer_precip_anomaly_forecast': df_summer_anomaly['tp'] * 86400 * 1000,
                'spring_temp_prob_warm_forecast': df_spring_prob['t2m'],
                'spring_precip_prob_wet_forecast': df_spring_prob['tp'],
                'summer_temp_prob_warm_forecast': df_summer_prob['t2m'],
                'summer_precip_prob_wet_forecast': df_summer_prob['tp']
            })
            # =======================================================================
            processed_features.append(year_df)
            logging.info(f"  -> Successfully extracted features for all {len(districts_gdf)} districts.")

    logging.info("\n*** STAGE 4: Finalizing and Validating Dataset ***")
    processed_df = pd.concat(processed_features, ignore_index=True)

    placeholder_years = [y for y in FINAL_DATASET_YEARS if y not in AVAILABLE_SEAS5_YEARS]
    placeholder_df = pd.DataFrame()
    if placeholder_years:
        placeholder_rows = []
        neutral_prob = 0.5
        for year in placeholder_years:
            for dist_id in districts_gdf['id']:
                placeholder_rows.append({
                    'year': year, 'district_no': dist_id,
                    'spring_temp_anomaly_forecast': 0.0, 'spring_precip_anomaly_forecast': 0.0,
                    'summer_temp_anomaly_forecast': 0.0, 'summer_precip_anomaly_forecast': 0.0,
                    'spring_temp_prob_warm_forecast': neutral_prob, 'spring_precip_prob_wet_forecast': neutral_prob,
                    'summer_temp_prob_warm_forecast': neutral_prob, 'summer_precip_prob_wet_forecast': neutral_prob
                })
        placeholder_df = pd.DataFrame(placeholder_rows)

    final_df = pd.concat([processed_df, placeholder_df], ignore_index=True)
    final_df = final_df[final_df['year'].isin(FINAL_DATASET_YEARS)].copy()
    final_df.sort_values(by=['year', 'district_no'], inplace=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)

    logging.info("  -> Final dataset saved. Validating content...")
    validation_df = pd.read_csv(OUTPUT_CSV_PATH)
    print("\n--- Validation of a sample forecast year (2021) ---")
    print(validation_df[validation_df.year == 2021].head())
    logging.info("\nVALIDATION COMPLETE: The district_no column should now be clean and correct.")


if __name__ == "__main__":
    process_seas5_forecasts()