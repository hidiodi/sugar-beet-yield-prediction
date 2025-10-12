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
    ds_weather = ds_weather.assign_coords(district_id_raster=district_raster)

    logging.info("District rasterization complete. Ready for GroupBy operation.")
    return ds_weather


def calculate_weather_features_for_districts_optimized(ds_weather):
    """
    Calculates weather features for all districts simultaneously using Xarray GroupBy.
    (FIXED: Dynamically identifies the internal 'stacked' dimension created by groupby for reduction.)
    """
    logging.info("--- Starting Optimized Zonal Statistics ---")

    # --- Step 1: In-depth logging of the initial state ---
    logging.info(f"[PRE-GROUPBY] Initial ds_weather object:\n{ds_weather}")
    logging.info(f"[PRE-GROUPBY] Dimensions of ds_weather: {ds_weather.dims}")
    logging.info(
        f"[PRE-GROUPBY] Dimensions of a data variable (Precipitation_Flux): {ds_weather['Precipitation_Flux'].dims}")
    logging.info(f"[PRE-GROUPBY] List of coordinates: {list(ds_weather.coords.keys())}")

    results = []
    ACCUMULATION_MONTHS = [7, 8, 9]

    # --- Step 2: Create the GroupBy object separately to inspect it ---
    logging.info("Creating the GroupBy object...")
    grouped_by_district = ds_weather.groupby('district_id_raster')
    logging.info(f"[POST-GROUPBY] Type of grouped object: {type(grouped_by_district)}")
    logging.info(f"[POST-GROUPBY] Dimensions available in the grouped object context: {grouped_by_district.dims}")

    # --- Step 3: CRITICAL FIX - Dynamically find the stacked dimension ---
    # The dimensions available for reduction are those in the grouped object that are NOT the grouping coordinate.
    available_reduce_dims = {k: v for k, v in grouped_by_district.dims.items() if k != 'district_id_raster'}
    # From this, we want the spatial dimension, which is the one that is NOT 'time'.
    spatial_dims_in_grouped_context = [d for d in available_reduce_dims if d != 'time']

    if len(spatial_dims_in_grouped_context) != 1:
        logging.error(
            f"FATAL: Expected to find exactly one stacked spatial dimension, but found: {spatial_dims_in_grouped_context}")
        raise ValueError("Could not determine the correct stacked dimension for reduction.")

    reduction_dim = spatial_dims_in_grouped_context[0]
    logging.info(f"SUCCESS: Dynamically identified the correct reduction dimension: '{reduction_dim}'")

    # --- Step 4: Perform the mean calculation using the correct dimension name ---
    logging.info(f"Calculating spatial mean for each district by reducing over dimension '{reduction_dim}'...")
    daily_mean_weather_by_district = grouped_by_district.mean(dim=reduction_dim)
    logging.info("Spatial mean calculation complete.")
    logging.info(f"Resulting dimensions after mean: {daily_mean_weather_by_district.dims}")

    # Drop the background (fill=0) group
    if 0 in daily_mean_weather_by_district.district_id_raster.values:
        daily_mean_weather_by_district = daily_mean_weather_by_district.sel(
            district_id_raster=daily_mean_weather_by_district.district_id_raster != 0)

    # --- Step 5: Iterate over years and calculate time-based features (unchanged) ---
    for year in tqdm(range(1979, 2025), desc="Processing Years"):
        year_data = daily_mean_weather_by_district.sel(time=str(year))

        # --- A. Accumulation Phase (July, Aug, Sep) ---
        peak_growth = year_data.sel(time=year_data.time.dt.month.isin(ACCUMULATION_MONTHS))

        # Perform all aggregations simultaneously
        precip_total_peak_growth = peak_growth['Precipitation_Flux'].sum(dim='time')
        temp_mean_peak_growth = peak_growth['Temperature_Air_2m_Mean_24h'].mean(dim='time')
        heat_stress_days_peak_growth = (peak_growth['Temperature_Air_2m_Max_24h'] > 30).sum(dim='time').astype(int)
        solar_rad_peak_growth = peak_growth['Solar_Radiation_Flux'].mean(dim='time')
        mean_t_max = peak_growth['Temperature_Air_2m_Max_24h'].mean(dim='time')
        mean_t_min = peak_growth['Temperature_Air_2m_Min_24h'].mean(dim='time')
        DTR_accumulation_phase = mean_t_max - mean_t_min

        # --- B. Early Spring Period (Mar 1 to Apr 15) ---
        early_spring_filter = ((year_data.time.dt.month == 3) |
                               ((year_data.time.dt.month == 4) & (year_data.time.dt.day <= 15)))
        early_spring_data = year_data.sel(time=early_spring_filter)
        freezing_days = (early_spring_data['Temperature_Air_2m_Min_24h'] < 0).sum(dim='time').astype(int)

        # Step 3: Combine results into a DataFrame
        district_ids = [str(int(d)).zfill(5) for d in precip_total_peak_growth.district_id_raster.values]
        year_results = pd.DataFrame({
            'district_no': district_ids, 'year': year,
            'precip_total_peak_growth': precip_total_peak_growth.values,
            'temp_mean_peak_growth': temp_mean_peak_growth.values,
            'heat_stress_days_peak_growth': heat_stress_days_peak_growth.values,
            'solar_rad_peak_growth': solar_rad_peak_growth.values,
            'DTR_accumulation_phase': DTR_accumulation_phase.values,
            'temp_min_peak_growth': mean_t_min.values,
            'temp_max_peak_growth': mean_t_max.values,
            'spring_freezing_days': freezing_days.values,
        })
        results.append(year_results)

    logging.info("--- Finished Optimized Zonal Statistics ---")
    return pd.concat(results, ignore_index=True)

def main():
    """Main orchestrator for creating the crop dataset with weather features."""
    logging.info("--- Starting Final Dataset Creation with Weather Features ---")

    # --- 1. Define Paths ---
    path_yield_data = Path("data/02_intermediate/sugarbeet_yield.csv")
    path_districts_geo = Path("data/01_raw/districts_official.geojson")
    path_weather_data = Path("data/02_intermediate/agera5_germany_merged.nc")

    # --- NEW: Add path to the pre-processed forecast features ---
    path_forecast_features = Path("data/03_processed/final_hybrid_features_only.csv")

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

    # --- 3. Rasterize Districts and Perform Zonal Statistics ---
    ds_weather = rasterize_districts(gdf_districts, ds_weather)
    df_weather_features = calculate_weather_features_for_districts_optimized(ds_weather)

    # --- 4. Assemble and Merge All Datasets ---
    logging.info("Assembling final dataset...")
    # Merge yield data with the calculated ground-truth weather features
    final_df = pd.merge(df_yield, df_weather_features, on=['district_no', 'year'], how='left')

    # --- NEW STEP: Load and merge the forecast features ---
    logging.info(f"Loading pre-processed forecast features from '{path_forecast_features}'...")
    if not path_forecast_features.exists():
        logging.error(
            f"FATAL: Forecast features file not found at '{path_forecast_features}'. Please run the forecast processing scripts first.")
        return

    df_forecast = pd.read_csv(path_forecast_features)
    # Ensure district_no is a zero-padded string for a reliable merge
    df_forecast['district_no'] = df_forecast['district_no'].astype(str).str.zfill(5)

    # Merge forecast data into the main dataframe
    final_df = pd.merge(final_df, df_forecast, on=['district_no', 'year'], how='left')
    logging.info("Successfully merged forecast features.")

    # --- NEW: Check for missingness after forecast merge, as these cannot be imputed ---
    forecast_cols = [col for col in df_forecast.columns if col not in ['year', 'district_no']]
    missing_forecasts = final_df[forecast_cols].isnull().sum().sum()
    if missing_forecasts > 0:
        logging.warning(
            f"Found {missing_forecasts} missing forecast data points after merging (this can happen for years without forecasts).")
        initial_rows = len(final_df)
        final_df.dropna(subset=forecast_cols, inplace=True)
        rows_dropped = initial_rows - len(final_df)
        logging.info(f"Dropped {rows_dropped} rows due to missing forecast features.")

    # --- 5. Impute Missing Ground-Truth Weather Data ---
    weather_cols = [col for col in df_weather_features.columns if col not in ['year', 'district_no']]
    missing_before = final_df[weather_cols].isnull().sum().sum()

    if missing_before > 0:
        logging.info(
            f"Found {missing_before} missing ground-truth weather data points. Starting state-level imputation...")
        districts_to_state = gdf_districts[['district_no', 'state']].drop_duplicates()
        final_df = pd.merge(final_df, districts_to_state, on='district_no', how='left')

        state_yearly_means = final_df.groupby(['state', 'year'])[weather_cols].transform('mean')
        final_df[weather_cols] = final_df[weather_cols].fillna(state_yearly_means)
        final_df.drop(columns=['state'], inplace=True)

        missing_after = final_df[weather_cols].isnull().sum().sum()
        logging.info(f"Imputed {missing_before - missing_after} values. {missing_after} remain.")

    # Drop any remaining rows with missing weather data
    initial_rows = len(final_df)
    final_df.dropna(subset=weather_cols, inplace=True)
    rows_dropped = initial_rows - len(final_df)
    if rows_dropped > 0:
        logging.info(f"Dropped {rows_dropped} rows where ground-truth weather data could not be imputed.")

    # --- 6. Format and Save Final Dataset ---
    logging.info("Formatting numeric columns...")
    all_feature_cols = weather_cols + forecast_cols
    for col in all_feature_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').round(5)

    logging.info(f"Saving final dataset with {len(final_df)} rows to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False)

    logging.info("--- SUCCESS: Dataset created! ---")
    print("\n--- Final Dataset Preview ---")
    print(final_df.head())
    print("\n--- Final Dataset Columns ---")
    print(final_df.columns.tolist())


if __name__ == "__main__":
    main()