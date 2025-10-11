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
    """
    logging.info("Loading and preparing input files...")

    # --- Load Agronomic (Yield) Data ---
    df_yield = pd.read_csv(path_yield_data)
    df_yield = df_yield[df_yield['year'].between(1979, 2024)].copy()
    initial_rows = len(df_yield)
    df_yield.dropna(subset=['yield'], inplace=True)
    rows_dropped = initial_rows - len(df_yield)
    if rows_dropped > 0:
        logging.info(f"Dropped {rows_dropped} rows with missing 'yield' values.")
    df_yield['district_no'] = df_yield['district_no'].astype(str).str.zfill(5)

    # --- Load Geospatial Data ---
    gdf_districts = gpd.read_file(path_districts_geo)

    if gdf_districts.index.name is not None:
        gdf_districts.reset_index(inplace=True)
    elif 'id' not in gdf_districts.columns and 'district_no' not in gdf_districts.columns:
        gdf_districts.reset_index(inplace=True)

    rename_map = {}
    if 'id' in gdf_districts.columns:
        rename_map['id'] = 'district_no'
    elif 'index' in gdf_districts.columns and 'district_no' not in gdf_districts.columns:
        rename_map['index'] = 'district_no'

    if rename_map:
        gdf_districts.rename(columns=rename_map, inplace=True)
        logging.info(f"Renamed column(s): {rename_map}")
    else:
        logging.error(
            f"GeoJSON does not contain 'id' or 'district_no' as a column/index after reset. Available columns: {gdf_districts.columns.tolist()}")
        raise KeyError("Could not find district identification column ('id' or 'district_no') in GeoJSON.")

    # Prepare district_no for merge (string) and rasterization (numeric)
    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    gdf_districts['district_no_int'] = pd.to_numeric(gdf_districts['district_no'])

    logging.info(f"Final GeoDataFrame columns: {gdf_districts.columns.tolist()}")

    # --- Load Weather Data ---
    ds_weather = xr.open_dataset(path_weather_data)
    ds_weather.rio.write_crs("EPSG:4326", inplace=True)

    # Convert temperatures to Celsius
    ds_weather['Temperature_Air_2m_Mean_24h'] -= 273.15
    ds_weather['Temperature_Air_2m_Max_24h'] -= 273.15
    ds_weather['Temperature_Air_2m_Min_24h'] -= 273.15

    return df_yield, gdf_districts, ds_weather


def rasterize_districts(gdf_districts, ds_weather):
    """
    Creates a raster layer where each cell contains the district ID,
    aligned to the weather dataset's grid using direct rasterio/shapely calls.
    """
    logging.info("Rasterizing district geometries onto weather grid using rasterio...")

    # 1. Define the geometric context
    transform = ds_weather.rio.transform()
    out_shape = (ds_weather.rio.height, ds_weather.rio.width)

    # 2. Prepare geometries and values: (geometry_in_geojson_format, value)
    geometries_and_values = [
        (shapely.geometry.mapping(geom), val)
        for geom, val in zip(gdf_districts.geometry, gdf_districts['district_no_int'])
    ]

    # 3. Perform Rasterization
    district_data = rasterio.features.rasterize(
        shapes=geometries_and_values,
        out_shape=out_shape,
        transform=transform,
        fill=0,  # Background value (outside of Germany)
        all_touched=True,
        dtype=np.int32
    )

    # 4. Create the final Xarray DataArray
    district_raster = xr.DataArray(
        district_data,
        coords={'lat': ds_weather.lat, 'lon': ds_weather.lon},
        dims=['lat', 'lon'],
        name='district_id_raster',
        attrs={'grid_mapping': ds_weather.rio.grid_mapping}
    )

    # Assign the raster as a coordinate to the weather dataset
    # This may implicitly trigger dimension stacking (e.g., to 'stacked_lat_lon')
    ds_weather = ds_weather.assign_coords(district_id_raster=district_raster)

    logging.info("District rasterization complete. Ready for GroupBy operation.")
    return ds_weather


def calculate_weather_features_for_districts_optimized(ds_weather):
    """
    Calculates weather features for all districts simultaneously using Xarray GroupBy.
    (FIXED: Dynamically identifies the spatial reduction dimension to account for stacking.)
    """
    logging.info("Starting optimized GroupBy zonal statistics for weather features...")

    results = []
    ACCUMULATION_MONTHS = [7, 8, 9]

    # --- CRITICAL FIX: Identify the correct spatial dimension for reduction ---
    # The spatial dimensions are the ones that are NOT 'time' and NOT the grouping coordinate.
    # We check the dimensions of a data variable (e.g., Precipitation_Flux) for the check.
    spatial_dims = [d for d in ds_weather['Precipitation_Flux'].dims if d not in ('time', 'district_id_raster')]

    if len(spatial_dims) == 1:
        spatial_dim_for_mean = spatial_dims[0]
        logging.info(f"Reducing spatial dimensions using dynamically identified dimension: '{spatial_dim_for_mean}'.")
    else:
        # Fallback to the dimension name indicated in the traceback: 'stacked_lat_lon'
        spatial_dim_for_mean = 'stacked_lat_lon'
        logging.warning(
            f"Could not dynamically identify single spatial dimension ({spatial_dims}). Falling back to '{spatial_dim_for_mean}'.")
    # -------------------------------------------------------------------------

    # Step 1: Group and calculate the spatial mean
    daily_mean_weather_by_district = ds_weather.groupby('district_id_raster').mean(dim=spatial_dim_for_mean)

    # Drop the background (fill=0) group
    if 0 in daily_mean_weather_by_district.district_id_raster.values:
        daily_mean_weather_by_district = daily_mean_weather_by_district.sel(
            district_id_raster=daily_mean_weather_by_district.district_id_raster != 0)

    # Step 2: Iterate over years and calculate time-based features
    for year in tqdm(range(1979, 2025), desc="Processing Years"):
        year_data = daily_mean_weather_by_district.sel(time=str(year))

        # --- A. Accumulation Phase (July, Aug, Sep) ---
        peak_growth = year_data.sel(time=year_data.time.dt.month.isin(ACCUMULATION_MONTHS))

        # Perform all aggregations simultaneously across all districts for the time period
        temp_max_peak = peak_growth['Temperature_Air_2m_Max_24h']
        temp_min_peak = peak_growth['Temperature_Air_2m_Min_24h']

        precip_total_peak_growth = peak_growth['Precipitation_Flux'].sum(dim='time')
        temp_mean_peak_growth = peak_growth['Temperature_Air_2m_Mean_24h'].mean(dim='time')
        heat_stress_days_peak_growth = (temp_max_peak > 30).sum(dim='time').values.astype(int)
        solar_rad_peak_growth = peak_growth['Solar_Radiation_Flux'].mean(dim='time')

        mean_t_max = temp_max_peak.mean(dim='time')
        mean_t_min = temp_min_peak.mean(dim='time')
        DTR_accumulation_phase = mean_t_max - mean_t_min

        # --- B. Early Spring Period (Mar 1 to Apr 15) ---
        early_spring_filter = ((year_data.time.dt.month == 3) |
                               ((year_data.time.dt.month == 4) & (year_data.time.dt.day <= 15)))

        early_spring_data = year_data.sel(time=early_spring_filter)
        freezing_days = (early_spring_data['Temperature_Air_2m_Min_24h'] < 0).sum(dim='time').values.astype(int)

        # Step 3: Combine results into a DataFrame
        district_ids = [str(int(d)).zfill(5) for d in precip_total_peak_growth.district_id_raster.values]

        year_results = pd.DataFrame({
            'district_no': district_ids,
            'year': year,
            'precip_total_peak_growth': precip_total_peak_growth.values,
            'temp_mean_peak_growth': temp_mean_peak_growth.values,
            'heat_stress_days_peak_growth': heat_stress_days_peak_growth,
            'solar_rad_peak_growth': solar_rad_peak_growth.values,
            'DTR_accumulation_phase': DTR_accumulation_phase.values,
            'temp_min_peak_growth': mean_t_min.values,
            'temp_max_peak_growth': mean_t_max.values,
            'spring_freezing_days': freezing_days,
        })
        results.append(year_results)

    # Final concatenation of all yearly results
    return pd.concat(results, ignore_index=True)


def main():
    """Main orchestrator for creating the crop dataset with weather features."""
    logging.info("--- Starting Final Dataset Creation with Weather Features ---")

    # --- 1. Define Paths ---
    path_yield_data = Path("data/02_intermediate/sugarbeet_yield.csv")
    path_districts_geo = Path("data/01_raw/districts_official.geojson")
    path_weather_data = Path("data/02_intermediate/agera5_germany_merged.nc")
    output_dir = Path("data/03_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / "final_dataset_with_advanced_features.csv"

    if output_filepath.exists():
        logging.info(f"Dataset at '{output_filepath}' already exists. Skipping process.")
        return

    # --- 2. Load and Pre-filter Data ---
    df_yield, gdf_districts, ds_weather = load_and_prepare_data(
        path_yield_data, path_districts_geo, path_weather_data
    )

    # --- NEW STEP: Rasterize Districts (Performance Critical) ---
    ds_weather = rasterize_districts(gdf_districts, ds_weather)

    # --- 3. Perform Zonal Statistics (Optimized) ---
    df_weather_features = calculate_weather_features_for_districts_optimized(ds_weather)

    # --- 4. Assemble, Merge, and Impute Missing Weather Data ---
    logging.info("Assembling final dataset...")
    final_df = pd.merge(df_yield, df_weather_features, on=['district_no', 'year'], how='left')

    # --- 4a. State-Level Imputation ---
    weather_cols = [
        'precip_total_peak_growth',
        'temp_mean_peak_growth',
        'heat_stress_days_peak_growth',
        'solar_rad_peak_growth',
        'DTR_accumulation_phase',
        'temp_min_peak_growth',
        'temp_max_peak_growth',
        'spring_freezing_days'
    ]

    missing_before = final_df[weather_cols].isnull().sum().sum()

    if missing_before > 0:
        logging.info(f"Found {missing_before} missing weather data points. Starting state-level imputation...")

        districts_to_state = gdf_districts[['district_no', 'state']].drop_duplicates()
        initial_merge_rows = len(final_df)
        final_df = pd.merge(final_df, districts_to_state, on='district_no', how='left')

        if len(final_df) != initial_merge_rows:
            logging.warning("Merge with state info changed the row count. Check GeoJSON for duplicate districts.")

        state_yearly_means = final_df.groupby(['state', 'year'])[weather_cols].transform('mean')
        final_df[weather_cols] = final_df[weather_cols].fillna(state_yearly_means)

        final_df.drop(columns=['state'], inplace=True)

        missing_after = final_df[weather_cols].isnull().sum().sum()
        logging.info(f"Imputed {missing_before - missing_after} values. {missing_after} remain.")

    # Drop any remaining rows with missing weather data
    initial_rows = len(final_df)
    final_df.dropna(subset=weather_cols, inplace=True)
    rows_dropped = initial_rows - len(final_df)
    logging.info(f"Dropped {rows_dropped} rows where weather data could not be imputed.")

    # --- 4b. Format Columns ---
    logging.info("Formatting numeric columns...")
    for col in weather_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce').round(5)

    # --- 5. Save Final Dataset ---
    logging.info(f"Saving final dataset with {len(final_df)} rows to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False)

    logging.info("--- SUCCESS: Dataset created! ---")
    print("\n--- Final Dataset Preview ---")
    print(final_df.head())


if __name__ == "__main__":
    main()