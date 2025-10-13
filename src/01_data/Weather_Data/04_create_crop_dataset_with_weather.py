# src/01_data/Weather_Data/04_create_crop_dataset_with_weather.py

import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import logging
from pathlib import Path
from tqdm import tqdm
import dask
import numpy as np

# --- CRITICAL FIX IMPORTS for Rasterization ---
import rasterio.features
import shapely.geometry

# ---------------------------------------------

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_data(path_yield_data, path_districts_geo, path_weather_data):
    """
    Loads and prepares all necessary input files for processing.
    (This function is unchanged)
    """
    logging.info("Loading and preparing input files...")
    df_yield = pd.read_csv(path_yield_data)
    df_yield = df_yield[
        df_yield['year'].between(1980, 2024)].copy()  # Start from 1980 to have antecedent data for the first year
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


# ==============================================================================
# === CORE LOGIC REPLACEMENT: NEW FEATURE ENGINEERING FUNCTION ===
# ==============================================================================
def calculate_agera5_features_for_districts(ds_weather):
    """
    Calculates district-level weather features using a two-tier strategy.
    Tier 1 (Antecedent: Oct-Feb): Detailed daily indices.
    Tier 2 (Forecast Period: Mar-Jul): Monthly averages to match SEAS5.
    """
    logging.info("--- Starting Two-Tier Zonal Statistics for AgERA5 Features ---")

    # --- Step 1: Calculate daily spatial means for each district ---
    logging.info("Calculating daily spatial mean weather for each district...")
    grouped_by_district = ds_weather.groupby('district_id_raster')

    # Dynamically find the stacked spatial dimension created by groupby
    spatial_dims = [d for d in grouped_by_district.dims if d not in ['time', 'district_id_raster']]
    if not spatial_dims:
        raise ValueError("Could not determine the stacked spatial dimension for reduction.")
    reduction_dim = spatial_dims[0]

    daily_mean_weather = grouped_by_district.mean(dim=reduction_dim)

    # Drop the background (fill=0) group which represents areas outside Germany
    if 0 in daily_mean_weather.district_id_raster.values:
        daily_mean_weather = daily_mean_weather.sel(
            district_id_raster=daily_mean_weather.district_id_raster != 0)
    logging.info("Daily spatial means calculated successfully.")

    # CRITICAL FIX: Explicitly sort the data by time to prevent non-monotonic index errors.
    logging.info("Ensuring the daily weather data is sorted chronologically...")
    daily_mean_weather = daily_mean_weather.sortby('time')
    logging.info("Time sorting complete. Proceeding to feature calculation.")

    results = []
    # Loop over each growing season year. We start in 1980 to have full antecedent data from Oct 1979.
    for year in tqdm(range(1980, 2025), desc="Processing Years for Feature Engineering"):

        # --- TIER 1: Antecedent Period (Oct-Feb) - Full Daily Detail ---
        # Select data from October of the previous year to the end of February of the current year.
        antecedent_data = daily_mean_weather.sel(time=slice(f'{year - 1}-10-01', f'{year}-02-28'))

        # Calculate detailed agronomic indices
        frost_days = (antecedent_data['Temperature_Air_2m_Min_24h'] < 0).sum(dim='time')
        heavy_precip_days = (antecedent_data['Precipitation_Flux'] > 10).sum(dim='time')

        # Growing Degree Days (GDD) for winter
        gdd_base = 5.0
        daily_gdd = (antecedent_data['Temperature_Air_2m_Mean_24h'] - gdd_base).clip(min=0)
        gdd_sum_winter = daily_gdd.sum(dim='time')

        # --- TIER 2: Forecast Period (Mar-Jul) - Monthly Averages Only ---
        # Select data for the period for which we will have SEAS5 forecasts
        forecast_period_data = daily_mean_weather.sel(time=slice(f'{year}-03-01', f'{year}-07-31'))

        # Resample the daily data into monthly bins and calculate aggregates
        # Using .mean() and .sum() separately is more explicit and less prone to multi-index issues
        monthly_mean = forecast_period_data.resample(time='1M').mean()
        monthly_sum = forecast_period_data.resample(time='1M').sum()

        # Extract the required monthly features
        monthly_features = {}
        month_map = {3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 7: 'jul'}
        for month_idx, month_name in month_map.items():
            data_for_month_mean = monthly_mean.sel(time=monthly_mean.time.dt.month == month_idx)
            data_for_month_sum = monthly_sum.sel(time=monthly_sum.time.dt.month == month_idx)

            monthly_features[f'temp_mean_{month_name}'] = data_for_month_mean['Temperature_Air_2m_Mean_24h'].squeeze(
                drop=True)
            monthly_features[f'precip_sum_{month_name}'] = data_for_month_sum['Precipitation_Flux'].squeeze(drop=True)
            monthly_features[f'srad_mean_{month_name}'] = data_for_month_mean['Solar_Radiation_Flux'].squeeze(drop=True)

        # Combine all features for the year into a single xarray Dataset
        year_ds = xr.Dataset({
            'antecedent_frost_days': frost_days,
            'antecedent_heavy_precip_days': heavy_precip_days,
            'antecedent_gdd_sum': gdd_sum_winter,
            **monthly_features
        })

        # Add year as a coordinate and convert to a DataFrame
        year_ds = year_ds.assign_coords(year=year)
        year_df = year_ds.to_dataframe()
        results.append(year_df)

    # Combine DataFrames from all years
    final_df = pd.concat(results).reset_index()
    # Clean up column names and types
    final_df.rename(columns={'district_id_raster': 'district_no_int'}, inplace=True)
    final_df['district_no'] = final_df['district_no_int'].astype(str).str.zfill(5)
    final_df.drop(columns=['district_no_int'], inplace=True)

    logging.info("--- Finished Two-Tier Feature Calculation ---")
    return final_df

def calculate_anomalies(df, base_period=(1991, 2020)):
    """Calculates feature anomalies relative to a base period."""
    logging.info(f"Calculating anomalies relative to {base_period[0]}-{base_period[1]} climatology...")

    feature_cols = [col for col in df.columns if col not in ['district_no', 'year']]

    # Filter the DataFrame to the base period to calculate the mean
    climatology_df = df[df['year'].between(base_period[0], base_period[1])]

    # Calculate the long-term mean for each feature for each district
    district_climatology = climatology_df.groupby('district_no')[feature_cols].mean().reset_index()

    # Merge the climatology back into the main DataFrame
    df_with_climatology = pd.merge(df, district_climatology, on='district_no', suffixes=('', '_clim'))

    # Calculate the anomalies
    for col in feature_cols:
        df_with_climatology[f'{col}_anomaly'] = df_with_climatology[col] - df_with_climatology[f'{col}_clim']

    # Keep only the year, district_no, and the new anomaly columns
    anomaly_cols = ['year', 'district_no'] + [f'{col}_anomaly' for col in feature_cols]
    final_anomaly_df = df_with_climatology[anomaly_cols]

    logging.info("Anomaly calculation complete.")
    return final_anomaly_df


def main():
    """Main orchestrator for creating the crop dataset with weather features."""
    logging.info("--- Starting Final Dataset Creation with ADVANCED Weather Features ---")

    # --- 1. Define Paths ---
    path_yield_data = Path("data/02_intermediate/sugarbeet_yield.csv")
    path_districts_geo = Path("data/01_raw/districts_official.geojson")
    path_weather_data = Path("data/02_intermediate/agera5_germany_merged.nc")
    output_dir = Path("data/03_processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================ THE FIX ============================
    # ### REVERTED TO ORIGINAL FILENAME ###
    # The output path is restored to ensure the existing pipeline is not broken.
    # This file will now contain the new, more powerful features.
    output_filepath = output_dir / "final_dataset_with_advanced_features.csv"
    # ===============================================================

    if output_filepath.exists():
        logging.info(f"Dataset at '{output_filepath}' already exists. Skipping process.")
        return

    # --- 2. Load and Pre-filter Data ---
    df_yield, gdf_districts, ds_weather = load_and_prepare_data(
        path_yield_data, path_districts_geo, path_weather_data
    )

    # --- 3. Rasterize Districts and Calculate All New AgERA5 Features ---
    ds_weather = rasterize_districts(gdf_districts, ds_weather)
    df_weather_features_raw = calculate_agera5_features_for_districts(ds_weather)
    df_weather_anomalies = calculate_anomalies(df_weather_features_raw)

    # --- 4. Assemble Final Dataset by Merging Yield with NEW Weather Anomalies ---
    # This step now replaces the logic of merging multiple old feature files.
    logging.info("Assembling final dataset by merging yield data with new weather anomalies...")
    final_df = pd.merge(df_yield, df_weather_anomalies, on=['district_no', 'year'], how='left')

    # --- 5. Handle Missing Data ---
    missing_count = final_df.isnull().sum().sum()
    if missing_count > 0:
        logging.warning(f"Found {missing_count} missing values after final merge. This can occur if yield data exists for years where weather data is incomplete. Dropping affected rows.")
        final_df.dropna(inplace=True)

    # --- 6. Save Final Dataset ---
    logging.info(f"Saving final dataset with {len(final_df)} rows to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False)

    logging.info("--- SUCCESS: Advanced feature dataset created! ---")
    print("\n--- Final Dataset Preview ---")
    print(final_df.head())
    print("\n--- Final Dataset Columns ---")
    print(final_df.columns.tolist())

if __name__ == "__main__":
    main()