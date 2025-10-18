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

def load_and_prepare_data(path_yield_data, path_districts_geo, path_weather_data):
    """
    Loads and prepares all necessary input files for processing.
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
    """
    logging.info("--- Starting calculation of ANTECEDENT (Oct-Feb) AgERA5 Features ---")
    grouped_by_district = ds_weather.groupby('district_id_raster').mean()
    if 0 in grouped_by_district.district_id_raster.values:
        grouped_by_district = grouped_by_district.sel(
            district_id_raster=grouped_by_district.district_id_raster != 0)

    daily_mean_weather = grouped_by_district
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
    return final_df


def calculate_anomalies(df, base_period=(1991, 2023)):
    """
    Calculates feature anomalies relative to a base period.
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
    return final_anomaly_df


def load_forecast_features(path_forecast):
    """
    Loads the pure forecast data and ensures 'district_no' is correctly formatted.
    """
    logging.info("Loading PURE forecast features (no backfilling)...")
    df_forecast = pd.read_csv(path_forecast)
    # Enforce string type on 'district_no' for a robust merge
    df_forecast['district_no'] = df_forecast['district_no'].astype(str).str.zfill(5)
    logging.info("Forecast features loaded successfully.")
    return df_forecast

def main():
    """Main orchestrator for creating the final master dataset."""
    logging.info("--- Starting Final Dataset Creation with PURE Forecast Features ---")

    # --- 1. Define Paths ---
    path_yield_data = Path("data/02_intermediate/sugarbeet_yield.csv")
    path_districts_geo = Path("data/01_raw/districts_official.geojson")
    path_weather_data = Path("data/02_intermediate/agera5_germany_merged.nc")
    path_forecast = Path("data/02_intermediate/ecmwf51_forecast_features_FINAL.csv")
    output_dir = Path("data/03_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = "data/03_processed/final_dataset_with_advanced_features.csv"

    # --- 2. Load Base Data ---
    df_yield, gdf_districts, ds_weather = load_and_prepare_data(
        path_yield_data, path_districts_geo, path_weather_data
    )

    if 'crs' in df_yield.columns:
        df_yield = df_yield.drop(columns=['crs'])

    # --- 3. Build Feature Sets ---
    ds_weather_rasterized = rasterize_districts(gdf_districts, ds_weather)
    df_antecedent_raw = calculate_agera5_antecedent_features_raw(ds_weather_rasterized)
    df_antecedent_anomalies = calculate_anomalies(df_antecedent_raw)

    # ============================ THE MINIMAL FIX (PART 2) ============================
    # ### Use the new, simpler function ###
    df_forecast = load_forecast_features(path_forecast)

    # --- 4. Assemble Final Dataset ---
    logging.info("Assembling final dataset by merging antecedent and PURE forecast features...")
    df_weather_features = pd.merge(df_antecedent_anomalies, df_forecast, on=['year', 'district_no'], how='inner')

    # ### Use an 'inner' merge to only keep years with yield AND all weather features ###
    final_df = pd.merge(df_yield, df_weather_features, on=['district_no', 'year'], how='inner')
    # =====================================================================================

    missing_count = final_df.isnull().sum().sum()
    if missing_count > 0:
        logging.warning(f"Found {missing_count} unexpected missing values after final merge. Check input data.")
        final_df.dropna(inplace=True)

    logging.info(f"Saving final master dataset with {len(final_df)} rows to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False, float_format='%.6f')

    logging.info("--- SUCCESS: Final master dataset created! ---")
    print("\n--- Final Dataset Preview ---")
    print(final_df.head())
    print(f"\n--- Final Dataset has {len(final_df.columns)} columns ---")
    print(final_df.columns.tolist())


if __name__ == "__main__":
    main()