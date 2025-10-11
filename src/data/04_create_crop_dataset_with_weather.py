# src/data/04_create_final_dataset.py (or whatever the file is named)

import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import logging
from pathlib import Path
from tqdm import tqdm
# Import Dask for robust parallel execution (optional, but good practice with large Geo/Xarray operations)
import dask

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_data(path_yield_data, path_districts_geo, path_weather_data):
    """
    Loads and prepares all necessary input files for processing.
    (No changes needed here - the ID fix and unit conversion are correct)
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

    # --- Load Geospatial Data (GeoJSON ID FIX is preserved) ---
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

    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    logging.info(f"Final GeoDataFrame columns: {gdf_districts.columns.tolist()}")

    # --- Load Weather Data ---
    ds_weather = xr.open_dataset(path_weather_data)
    ds_weather.rio.write_crs("EPSG:4326", inplace=True)

    # Convert temperatures to Celsius for all calculations
    ds_weather['Temperature_Air_2m_Mean_24h'] -= 273.15
    ds_weather['Temperature_Air_2m_Max_24h'] -= 273.15
    ds_weather['Temperature_Air_2m_Min_24h'] -= 273.15

    return df_yield, gdf_districts, ds_weather


def calculate_weather_features_for_districts(gdf_districts, ds_weather):
    """
    Performs zonal statistics to calculate weather features for each district.
    """
    logging.info("Starting advanced zonal statistics for weather features...")

    results = []

    # Define time windows using explicit month/day ranges where possible
    ACCUMULATION_MONTHS = [7, 8, 9]  # July, August, September

    # Define a robust date range for early spring (March 1 to April 15)
    # This requires careful filtering inside the loop.

    # Use dask.config.set to increase the default timeout for the clip operation
    # This helps with large Xarray/GeoPandas operations
    with dask.config.set({'array.chunk-size': '256MiB', 'distributed.comm.timeouts': '60s'}):

        for district in tqdm(gdf_districts.itertuples(), total=len(gdf_districts), desc="Processing Districts"):
            try:
                if district.geometry is None or district.geometry.is_empty:
                    logging.warning(f"Skipping district {district.district_no} due to empty geometry.")
                    continue

                # 1. Clip the weather data
                # Use a larger buffer or skip the clip if the geometry is simple and small to avoid memory issues
                clipped_ds = ds_weather.rio.clip([district.geometry], gdf_districts.crs, drop=True)

                if clipped_ds.rio.width == 0 or clipped_ds.rio.height == 0:
                    raise ValueError("Clipping resulted in an empty dataset.")

                # 2. Calculate the spatial mean for daily data
                daily_mean_weather = clipped_ds.mean(dim=['lat', 'lon'])

                for year in range(1979, 2025):
                    year_data = daily_mean_weather.sel(time=str(year))

                    # --- A. Accumulation Phase (July, Aug, Sep) ---
                    peak_growth = year_data.sel(time=year_data.time.dt.month.isin(ACCUMULATION_MONTHS))

                    # 1. Calculate DTR components
                    mean_t_max = peak_growth['Temperature_Air_2m_Max_24h'].mean().values.item()
                    mean_t_min = peak_growth['Temperature_Air_2m_Min_24h'].mean().values.item()
                    DTR = mean_t_max - mean_t_min

                    # --- B. Early Spring Period (Mar 1 to Apr 15) ---
                    # CRITICAL FIX 1: Robust date filtering for the specific period
                    # Filter for March (month=3) OR April 1st to 15th (month=4 AND day<=15)
                    early_spring_filter = ((year_data.time.dt.month == 3) |
                                           ((year_data.time.dt.month == 4) & (year_data.time.dt.day <= 15)))

                    early_spring_data = year_data.sel(time=early_spring_filter)

                    # Count days when minimum temperature was below freezing
                    freezing_days = (early_spring_data['Temperature_Air_2m_Min_24h'] < 0).sum().values.item()

                    # --- 3. Append Results (CRITICAL FIX 2: Ensure all keys are present) ---
                    results.append({
                        'district_no': district.district_no,
                        'year': year,
                        'precip_total_peak_growth': peak_growth['Precipitation_Flux'].sum().values.item(),
                        'temp_mean_peak_growth': peak_growth['Temperature_Air_2m_Mean_24h'].mean().values.item(),
                        'heat_stress_days_peak_growth': (
                                peak_growth['Temperature_Air_2m_Max_24h'] > 30).sum().values.item(),
                        # CRITICAL FIX 2: Include the correct Solar variable name
                        'solar_rad_peak_growth': peak_growth['Solar_Radiation_Flux'].mean().values.item(),
                        'DTR_accumulation_phase': DTR,
                        'temp_min_peak_growth': mean_t_min,
                        'temp_max_peak_growth': mean_t_max,
                        'spring_freezing_days': freezing_days,
                    })

            except Exception as e:
                logging.error(f"FATAL ERROR processing district {district.district_no} in year {year}. Error: {e}",
                              exc_info=True)
                continue

    return pd.DataFrame(results)


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

    # --- 3. Perform Zonal Statistics ---
    # Temporarily set to a smaller number of districts for testing if the full run is too slow
    # gdf_districts = gdf_districts.head(10)

    df_weather_features = calculate_weather_features_for_districts(gdf_districts, ds_weather)
    if 'district_no' not in df_weather_features.columns:
        logging.error(
            f"Weather features DataFrame is missing 'district_no'. Columns found: {df_weather_features.columns.tolist()}")
        raise KeyError("Fatal Error: 'district_no' is missing from the calculated weather features.")

    # --- 4. Assemble, Merge, and Impute Missing Weather Data ---
    logging.info("Assembling final dataset...")
    final_df = pd.merge(df_yield, df_weather_features, on=['district_no', 'year'], how='left')

    # --- 4a. NEW: State-Level Imputation ---
    # CRITICAL FIX 3: Update weather_cols to include all new features
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

        # Add state information for grouping (only include necessary columns)
        districts_to_state = gdf_districts[['district_no', 'state']].drop_duplicates()
        initial_merge_rows = len(final_df)
        final_df = pd.merge(final_df, districts_to_state, on='district_no', how='left')

        if len(final_df) != initial_merge_rows:
            logging.warning(
                "Merge with state info changed the row count. Check for duplicate district_no entries in GeoJSON.")

        # Calculate state-level means for each weather column, grouped by year
        # Use .groupby(...).transform('mean') which is the correct pattern for imputation
        state_yearly_means = final_df.groupby(['state', 'year'])[weather_cols].transform('mean')

        # Fill missing values with the calculated state-level means
        final_df[weather_cols] = final_df[weather_cols].fillna(state_yearly_means)

        # Clean up the temporary state column
        final_df.drop(columns=['state'], inplace=True)

        missing_after = final_df[weather_cols].isnull().sum().sum()
        logging.info(f"Imputed {missing_before - missing_after} values. {missing_after} remain.")

    # Drop any rows that still have missing data (e.g., if a whole state failed or district had no geometry)
    initial_rows = len(final_df)
    final_df.dropna(subset=weather_cols, inplace=True)
    rows_dropped = initial_rows - len(final_df)
    logging.info(f"Dropped {rows_dropped} rows where weather data could not be imputed.")

    # --- 4b. Format Columns ---
    logging.info("Formatting numeric columns...")
    # Since all the calculated features are numeric, ensure they are float types and round them.
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