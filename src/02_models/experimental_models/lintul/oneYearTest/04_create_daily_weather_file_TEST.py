# File: src/data/04_create_daily_weather_file_ONE_YEAR_TEST.py
# Description: A FAST-RUNNING and ROBUST test version of the parallel script.
#              Processes a single year to generate test data and includes
#              a pre-flight check to validate NetCDF variable names.

import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import logging
from rasterstats import zonal_stats
from tqdm import tqdm
import rioxarray
import multiprocessing
import os
import sys

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')

# --- TEST CONFIGURATION ---
TEST_YEAR = 2018

# --- Path Configuration ---
BASE_DIR = Path.cwd()
AGERA5_INPUT_PATH = BASE_DIR / "data/02_intermediate/agera5_germany_merged.nc"
GEOJSON_PATH = BASE_DIR / "data/01_raw/districts_official.geojson"
OUTPUT_CSV_PATH = BASE_DIR / f"data/02_intermediate/historical_daily_weather_era5_{TEST_YEAR}_TEST.csv"
TEMP_RASTER_DIR = BASE_DIR / "data/02_intermediate/temp_rasters"

# --- IMPORTANT: VARIABLE MAPPING ---
# You MUST verify that the keys in this dictionary (e.g., 'Temperature_Air_2m_Min_24h')
# EXACTLY MATCH the variable names in your NetCDF file.
# Run the small verification script from the previous step if you are unsure.
VARIABLE_MAP = {
    'Temperature_Air_2m_Min_24h': 'tmin',
    'Temperature_Air_2m_Max_24h': 'tmax',
    'Precipitation_Flux': 'precip',
    'Solar_Radiation_Flux': 'srad',
}

# --- Global variables for multiprocessing pool ---
# These are initialized once per worker process to avoid repeatedly loading large files.
ds = None
gdf = None


def init_worker(dataset_path, geojson_path):
    """Initializer for each worker in the multiprocessing pool."""
    global ds, gdf
    try:
        ds = xr.open_dataset(dataset_path)
        gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    except Exception as e:
        logging.error(f"Worker initialization failed: {e}")
        # Propagate the error to stop the main process
        raise


def process_day(day):
    """
    Processes a single day of data: extracts each variable, calculates the zonal mean
    for each district, applies unit conversions, and returns a DataFrame.
    """
    pid = os.getpid()
    temp_raster_path = TEMP_RASTER_DIR / f"temp_raster_{pid}.tif"
    try:
        # Select the data for the given day
        daily_slice = ds.sel(time=day)

        # Prepare a dictionary to hold the results for this day
        district_daily_stats = {
            'date': pd.to_datetime(day),
            'district_no': gdf['id'].tolist()
        }

        # Loop through each variable defined in the map
        for nc_var, csv_col in VARIABLE_MAP.items():
            # This check is redundant due to the main pre-flight check, but good for safety
            if nc_var not in daily_slice:
                logging.warning(f"Variable '{nc_var}' not found for day {day}. Skipping.")
                continue

            data_array = daily_slice[nc_var]

            # Set spatial information for raster processing
            data_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            data_array.rio.write_crs("EPSG:4326", inplace=True)

            # Write to a temporary raster file for zonal_stats
            data_array.rio.to_raster(temp_raster_path, compress='LZW', driver='GTiff')

            # Calculate mean value for each district polygon
            stats = zonal_stats(gdf, str(temp_raster_path), stats="mean", all_touched=True)
            mean_values = [s['mean'] for s in stats]

            # --- Apply Unit Conversions ---
            if csv_col in ['tmin', 'tmax']:
                # Kelvin to Celsius
                converted_values = [v - 273.15 if v is not None else None for v in mean_values]
            elif csv_col == 'precip':
                # kg m-2 s-1 to mm/day (Flux to Amount)
                # 1 kg m-2 is 1 mm of water. 86400 seconds in a day.
                converted_values = [v * 86400 if v is not None else None for v in mean_values]
            elif csv_col == 'srad':
                # W m-2 to MJ m-2 day-1 (Flux to Amount)
                # J s-1 m-2 * 86400 s day-1 / 1,000,000 J MJ-1
                converted_values = [(v * 86400) / 1_000_000 if v is not None else None for v in mean_values]
            else:
                converted_values = mean_values

            district_daily_stats[csv_col] = converted_values

        return pd.DataFrame(district_daily_stats)

    except Exception as e:
        logging.error(f"Error processing day {day} in worker {pid}: {e}", exc_info=True)
        return None
    finally:
        # Clean up the temporary file
        if temp_raster_path.exists():
            temp_raster_path.unlink()


def create_daily_weather_file_parallel_test():
    """
    Main orchestrator function. Validates inputs, sets up a multiprocessing pool,
    processes the data for the test year, and saves the result.
    """
    logging.info(f"--- Starting: ONE-YEAR TEST for Daily Weather Creation (Year: {TEST_YEAR}) ---")

    # --- 1. PRE-FLIGHT CHECKS ---
    logging.info("Performing pre-flight checks...")
    for path in [AGERA5_INPUT_PATH, GEOJSON_PATH]:
        if not path.exists():
            logging.error(f"FATAL: Required input file not found at '{path}'. Exiting.")
            sys.exit(1)

    # **CRITICAL**: Validate that variable names in the map exist in the NetCDF file
    with xr.open_dataset(AGERA5_INPUT_PATH) as temp_ds:
        available_vars = list(temp_ds.data_vars)
        for required_var in VARIABLE_MAP.keys():
            if required_var not in available_vars:
                logging.error("=" * 80)
                logging.error(f"FATAL: Variable '{required_var}' from VARIABLE_MAP was not found in the NetCDF file.")
                logging.error(f"Please correct the VARIABLE_MAP in the script.")
                logging.error(f"Available variables in '{AGERA5_INPUT_PATH}' are: {available_vars}")
                logging.error("=" * 80)
                sys.exit(1)
    logging.info("✓ Pre-flight checks passed. All required variables found.")

    # Check if output already exists
    if OUTPUT_CSV_PATH.exists():
        logging.warning(f"Test output file {OUTPUT_CSV_PATH} already exists. Skipping recreation.")
        return

    # --- 2. SETUP PARALLEL PROCESSING ---
    TEMP_RASTER_DIR.mkdir(exist_ok=True)
    # Use one less than the total number of CPUs to leave resources for the OS
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Initializing a pool of {num_workers} worker processes.")

    # --- 3. FILTER DATES FOR THE TEST YEAR ---
    with xr.open_dataset(AGERA5_INPUT_PATH) as temp_ds:
        all_time_steps = temp_ds['time'].values
        # This is a more efficient way to filter dates within a specific year
        time_steps_for_year = pd.to_datetime(all_time_steps)
        time_steps_for_year = time_steps_for_year[time_steps_for_year.year == TEST_YEAR].to_numpy()

        if len(time_steps_for_year) == 0:
            logging.error(f"FATAL: No time steps found for the year {TEST_YEAR} in the NetCDF file. Exiting.")
            sys.exit(1)
        logging.info(
            f"Filtered {len(all_time_steps)} total timesteps down to {len(time_steps_for_year)} for {TEST_YEAR}.")

    # --- 4. EXECUTE PROCESSING POOL ---
    all_results = []
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker,
                              initargs=(AGERA5_INPUT_PATH, GEOJSON_PATH)) as pool:
        # Use imap_unordered for efficiency, as the order of days doesn't matter until the end.
        results_iterator = pool.imap_unordered(process_day, time_steps_for_year)

        # Wrap the iterator with tqdm for a progress bar
        for result_df in tqdm(results_iterator, total=len(time_steps_for_year), desc=f"Processing {TEST_YEAR} Data"):
            if result_df is not None:
                all_results.append(result_df)

    # --- 5. FINALIZE AND SAVE ---
    if not all_results:
        logging.error("FATAL: No results were generated by the worker processes. Check worker logs for errors.")
        sys.exit(1)

    logging.info(f"Concatenating {len(all_results)} daily results...")
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.sort_values(by=['date', 'district_no'], inplace=True)

    logging.info(f"Saving ONE-YEAR test data to {OUTPUT_CSV_PATH}")
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.4f')

    # --- 6. CLEANUP ---
    try:
        if TEMP_RASTER_DIR.exists():
            # Check if the directory is empty before removing
            if not os.listdir(TEMP_RASTER_DIR):
                os.rmdir(TEMP_RASTER_DIR)
            else:
                logging.warning(
                    f"Temp raster directory {TEMP_RASTER_DIR} is not empty. Manual cleanup may be required.")
    except OSError as e:
        logging.error(f"Error removing temporary directory {TEMP_RASTER_DIR}: {e}")

    logging.info(f"--- SUCCESS: ONE-YEAR test file created for {TEST_YEAR}! ---")


if __name__ == "__main__":
    # Set start method for multiprocessing, crucial for macOS and Windows
    # 'fork' is default on Linux, 'spawn' is default/required for others.
    # 'spawn' is safer across all platforms.
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # It can only be set once.
        pass

    create_daily_weather_file_parallel_test()