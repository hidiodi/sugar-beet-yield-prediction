import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import rasterio.features
import shapely.geometry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ... (all functions before `create_hybrid_forecast_features` are correct and omitted for brevity) ...
def load_and_prepare_data(path_yield_data, path_districts_geo, path_weather_data):
    """
    Loads and prepares all necessary input files for processing.
    (This function is unchanged)
    """
    logging.info("Loading and preparing input files...")
    df_yield = pd.read_csv(path_yield_data)
    df_yield = df_yield[
        df_yield['year'].between(1980, 2024)].copy()
    df_yield['district_no'] = df_yield['district_no'].astype(str).str.zfill(5)

    gdf_districts = gpd.read_file(path_districts_geo)
    if 'id' in gdf_districts.columns:
        gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
    elif 'index' in gdf_districts.columns and 'district_no' not in gdf_districts.columns:
        gdf_districts.rename(columns={'index': 'district_no'}, inplace=True)

    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    gdf_districts['district_no_int'] = pd.to_numeric(gdf_districts['district_no'])

    ds_weather = xr.open_dataset(path_weather_data)
    ds_weather.rio.write_crs("EPSG:4326", inplace=True)

    ds_weather['Temperature_Air_2m_Mean_24h'] -= 273.15
    ds_weather['Temperature_Air_2m_Max_24h'] -= 273.15
    ds_weather['Temperature_Air_2m_Min_24h'] -= 273.15

    return df_yield, gdf_districts, ds_weather


def rasterize_districts(gdf_districts, ds_weather):
    """
    Creates a raster layer aligned to the weather dataset's grid.
    (This function is unchanged)
    """
    logging.info("Rasterizing district geometries onto weather grid...")
    transform = ds_weather.rio.transform()
    out_shape = (ds_weather.rio.height, ds_weather.rio.width)
    geometries_and_values = [
        (shapely.geometry.mapping(geom), val)
        for geom, val in zip(gdf_districts.geometry, gdf_districts['district_no_int'])
    ]
    district_data = rasterio.features.rasterize(
        shapes=geometries_and_values, out_shape=out_shape, transform=transform,
        fill=0, all_touched=True, dtype=np.int32
    )
    district_raster = xr.DataArray(
        district_data, coords={'lat': ds_weather.lat, 'lon': ds_weather.lon},
        dims=['lat', 'lon'], name='district_id_raster'
    )
    ds_weather = ds_weather.assign_coords(district_id_raster=district_raster)
    logging.info("District rasterization complete.")
    return ds_weather


def calculate_agera5_antecedent_features_raw(ds_weather):
    """
    Calculates district-level ANTECEDENT weather features (Oct-Feb) from AGERA5.
    This is data that is KNOWN at the time of a forecast in March.
    """
    logging.info("--- Starting calculation of ANTECEDENT (Oct-Feb) AgERA5 Features ---")

    logging.info("Calculating daily spatial mean weather for each district...")
    grouped_by_district = ds_weather.groupby('district_id_raster')
    spatial_dims = [d for d in grouped_by_district.dims if d not in ['time', 'district_id_raster']]
    if not spatial_dims: raise ValueError("Could not determine the stacked spatial dimension.")
    daily_mean_weather = grouped_by_district.mean(dim=spatial_dims[0])

    if 0 in daily_mean_weather.district_id_raster.values:
        daily_mean_weather = daily_mean_weather.sel(
            district_id_raster=daily_mean_weather.district_id_raster != 0)

    logging.info("Ensuring daily weather data is sorted chronologically...")
    daily_mean_weather = daily_mean_weather.sortby('time')

    results = []
    for year in tqdm(range(1980, 2025), desc="Processing Antecedent Features by Year"):
        antecedent_data = daily_mean_weather.sel(time=slice(f'{year - 1}-10-01', f'{year}-02-28'))

        frost_days = (antecedent_data['Temperature_Air_2m_Min_24h'] < 0).sum(dim='time')
        heavy_precip_days = (antecedent_data['Precipitation_Flux'] > 10).sum(dim='time')
        gdd_base = 5.0
        daily_gdd = (antecedent_data['Temperature_Air_2m_Mean_24h'] - gdd_base).clip(min=0)
        gdd_sum_winter = daily_gdd.sum(dim='time')

        year_ds = xr.Dataset({
            'antecedent_frost_days': frost_days,
            'antecedent_heavy_precip_days': heavy_precip_days,
            'antecedent_gdd_sum': gdd_sum_winter,
        })
        year_ds = year_ds.assign_coords(year=year)
        year_df = year_ds.to_dataframe()
        results.append(year_df)

    final_df = pd.concat(results).reset_index()
    final_df.rename(columns={'district_id_raster': 'district_no_int'}, inplace=True)
    final_df['district_no'] = final_df['district_no_int'].astype(str).str.zfill(5)
    final_df.drop(columns=['district_no_int'], inplace=True)

    logging.info("--- Finished Antecedent Feature Calculation ---")
    return final_df


def calculate_anomalies(df, base_period=(1991, 2020)):
    """
    Calculates feature anomalies relative to a base period.
    (This function is unchanged)
    """
    logging.info(f"Calculating anomalies relative to {base_period[0]}-{base_period[1]} climatology...")
    feature_cols = [col for col in df.columns if col not in ['district_no', 'year']]
    climatology_df = df[df['year'].between(base_period[0], base_period[1])]
    district_climatology = climatology_df.groupby('district_no')[feature_cols].mean().reset_index()
    df_with_climatology = pd.merge(df, district_climatology, on='district_no', suffixes=('', '_clim'))

    for col in feature_cols:
        df_with_climatology[f'{col}_anomaly'] = df_with_climatology[col] - df_with_climatology[f'{col}_clim']

    anomaly_cols = ['year', 'district_no'] + [f'{col}_anomaly' for col in feature_cols]
    final_anomaly_df = df_with_climatology[anomaly_cols]
    logging.info("Anomaly calculation complete.")
    return final_anomaly_df


def create_hybrid_forecast_features(path_seas5_forecast, path_agera5_truth):
    """
    Creates a hybrid forecast dataset, now ensuring consistent data types for 'district_no'.
    """
    logging.info("Creating hybrid forecast feature set (anomaly + probability)...")
    df_forecast = pd.read_csv(path_seas5_forecast)
    df_truth = pd.read_csv(path_agera5_truth)

    # ============================ THE FIX ============================
    # ### CRITICAL FIX: Enforce string type on 'district_no' for merging ###
    # This prevents the ValueError caused by merging string and integer columns.
    df_forecast['district_no'] = df_forecast['district_no'].astype(str).str.zfill(5)
    df_truth['district_no'] = df_truth['district_no'].astype(str).str.zfill(5)
    # =================================================================

    # Part 1: Real forecasts from SEAS5 (1981 onwards)
    df_forecast_real = df_forecast[df_forecast['year'] > 1980].copy()

    # Part 2: Ground truth from AGERA5 to fill missing years (1979, 1980)
    df_truth_fill = df_truth[df_truth['year'] <= 1980].copy()

    rename_dict = {
        'spring_temp_anomaly_actual': 'spring_temp_anomaly_forecast',
        'spring_precip_anomaly_actual': 'spring_precip_anomaly_forecast',
        'summer_temp_anomaly_actual': 'summer_temp_anomaly_forecast',
        'summer_precip_anomaly_actual': 'summer_precip_anomaly_forecast',
    }
    df_truth_fill.rename(columns=rename_dict, inplace=True)

    neutral_prob = 0.5
    df_truth_fill['spring_temp_prob_warm_forecast'] = neutral_prob
    df_truth_fill['spring_precip_prob_wet_forecast'] = neutral_prob
    df_truth_fill['summer_temp_prob_warm_forecast'] = neutral_prob
    df_truth_fill['summer_precip_prob_wet_forecast'] = neutral_prob

    final_columns = df_forecast_real.columns
    df_truth_fill = df_truth_fill[final_columns]

    df_hybrid_forecast = pd.concat([df_truth_fill, df_forecast_real], ignore_index=True)
    df_hybrid_forecast.sort_values(by=['year', 'district_no'], inplace=True)
    logging.info("Hybrid forecast feature set created successfully.")
    return df_hybrid_forecast


def main():
    """Main orchestrator for creating the crop dataset with valid weather features."""
    logging.info("--- Starting Final Dataset Creation with CORRECTED Weather Features ---")

    # --- 1. Define Paths ---
    path_yield_data = Path("data/02_intermediate/sugarbeet_yield.csv")
    path_districts_geo = Path("data/01_raw/districts_official.geojson")
    path_weather_data = Path("data/02_intermediate/agera5_germany_merged.nc")
    path_seas5_forecast = Path("data/02_intermediate/seas5_forecast_features_1979_2021.csv")
    path_agera5_truth = Path("data/02_intermediate/agera5_ground_truth_1979_2021.csv")
    output_dir = Path("data/03_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / "final_dataset_with_advanced_features.csv"

    # --- 2. Load Base Data ---
    df_yield, gdf_districts, ds_weather = load_and_prepare_data(
        path_yield_data, path_districts_geo, path_weather_data
    )

    # ============================ THE FIX ============================
    # ### CRITICAL FIX: Remove the non-feature 'crs' column if it exists ###
    # This prevents the `calculate_anomalies` function from creating a junk 'crs_anomaly' column.
    if 'crs' in df_yield.columns:
        logging.info("Removing non-feature 'crs' column from yield data.")
        df_yield = df_yield.drop(columns=['crs'])
    # =================================================================

    # --- (Rest of the script is unchanged) ---
    if output_filepath.exists():
        logging.info(f"Dataset at '{output_filepath}' already exists. Skipping process.")
        # If the file exists, we still need to run the rest of the script for subsequent steps.
        # This line is removed so that the script can proceed if the file exists but needs to be regenerated.

    ds_weather_rasterized = rasterize_districts(gdf_districts, ds_weather)
    df_antecedent_raw = calculate_agera5_antecedent_features_raw(ds_weather_rasterized)
    df_antecedent_anomalies = calculate_anomalies(df_antecedent_raw)
    df_hybrid_forecast = create_hybrid_forecast_features(path_seas5_forecast, path_agera5_truth)

    logging.info("Assembling final dataset by merging antecedent and forecast features...")
    df_weather_features = pd.merge(df_antecedent_anomalies, df_hybrid_forecast, on=['year', 'district_no'], how='inner')
    final_df = pd.merge(df_yield, df_weather_features, on=['district_no', 'year'], how='left')

    missing_count = final_df.isnull().sum().sum()
    if missing_count > 0:
        logging.warning(f"Found {missing_count} missing values after final merge. Dropping affected rows.")
        final_df.dropna(inplace=True)

    logging.info(f"Saving final dataset with {len(final_df)} rows to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False)

    logging.info("--- SUCCESS: Corrected feature dataset created! ---")
    print("\n--- Final Dataset Preview ---")
    print(final_df.head())
    print("\n--- Final Dataset Columns ---")
    print(final_df.columns.tolist())


if __name__ == "__main__":
    main()
