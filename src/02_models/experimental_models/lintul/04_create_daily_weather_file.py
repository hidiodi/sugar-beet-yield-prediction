# File: src/data/04_create_daily_weather_file.py
# Description: HIGH-PERFORMANCE, PARALLELIZED script to create the daily weather CSV.
#              Uses all available CPU cores to drastically reduce processing time from days to hours.

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

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path.cwd()
AGERA5_INPUT_PATH = BASE_DIR / "data/02_intermediate/agera5_germany_merged.nc"
GEOJSON_PATH = BASE_DIR / "data/01_raw/districts_official.geojson"
OUTPUT_CSV_PATH = BASE_DIR / "data/02_intermediate/historical_daily_weather_era5.csv"
TEMP_RASTER_DIR = BASE_DIR / "data/02_intermediate/temp_rasters"

VARIABLE_MAP = {
    'Temperature_Air_2m_Min_24h': 'tmin',
    'Temperature_Air_2m_Max_24h': 'tmax',
    'Precipitation_Flux': 'precip',
    'Solar_Radiation_Flux': 'srad',
}

# --- Global objects for workers ---
# These will be loaded once and inherited by the child processes, saving memory and time.
ds = None
gdf = None


def init_worker(dataset_path, geojson_path):
    """Initializer for each worker process. Loads large objects into the process's memory."""
    global ds, gdf
    ds = xr.open_dataset(dataset_path)
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")


def process_day(day):
    """
    The core logic for processing a single day. This function is executed by each worker process.
    """
    # Each worker needs its own unique temp file to avoid race conditions
    pid = os.getpid()
    temp_raster_path = TEMP_RASTER_DIR / f"temp_raster_{pid}.tif"

    try:
        daily_slice = ds.sel(time=day)

        district_daily_stats = {
            'date': pd.to_datetime(day),
            'district_no': gdf['id'].tolist()
        }

        for nc_var, csv_col in VARIABLE_MAP.items():
            if nc_var not in daily_slice:
                continue

            data_array = daily_slice[nc_var]
            data_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            data_array.rio.write_crs("EPSG:4326", inplace=True)
            data_array.rio.to_raster(temp_raster_path, compress='LZW')

            stats = zonal_stats(gdf, str(temp_raster_path), stats="mean")
            mean_values = [s['mean'] for s in stats]

            if csv_col in ['tmin', 'tmax']:
                converted_values = [v - 273.15 if v is not None else None for v in mean_values]
            elif csv_col == 'precip':
                converted_values = [v * 86400 if v is not None else None for v in mean_values]
            elif csv_col == 'srad':
                converted_values = [(v * 86400) / 1_000_000 if v is not None else None for v in mean_values]
            else:
                converted_values = mean_values

            district_daily_stats[csv_col] = converted_values

        return pd.DataFrame(district_daily_stats)
    except Exception as e:
        # Log the error but return None so the process doesn't crash
        logging.error(f"Error processing day {day} in worker {pid}: {e}")
        return None
    finally:
        # Clean up the unique temp file
        if temp_raster_path.exists():
            temp_raster_path.unlink()


def create_daily_weather_file_parallel():
    """
    Main function to orchestrate the parallel processing of AgERA5 data.
    """
    logging.info("--- Starting: HIGH-PERFORMANCE Parallel Daily Weather CSV Creation ---")

    if OUTPUT_CSV_PATH.exists():
        logging.info(f"Output file {OUTPUT_CSV_PATH} already exists. Skipping.")
        return

    # Create a directory for temporary raster files
    TEMP_RASTER_DIR.mkdir(exist_ok=True)

    # Use all available CPU cores, but leave one free for system stability
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Initializing a pool of {num_workers} worker processes.")

    # Get the list of all dates to process
    with xr.open_dataset(AGERA5_INPUT_PATH) as temp_ds:
        time_steps = temp_ds['time'].values

    all_results = []

    # Create the pool of processes
    # The `initializer` function loads the large data files into each worker ONCE.
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker,
                              initargs=(AGERA5_INPUT_PATH, GEOJSON_PATH)) as pool:

        # Use imap_unordered for efficiency and wrap with tqdm for a progress bar
        # It will process the list of dates and feed them to the worker function
        results_iterator = pool.imap_unordered(process_day, time_steps)

        for result_df in tqdm(results_iterator, total=len(time_steps), desc="Processing Daily Weather in Parallel"):
            if result_df is not None:
                all_results.append(result_df)

    # --- Finalize and Save ---
    if not all_results:
        logging.error("No data was processed. Output file will not be created.")
        return

    logging.info("Concatenating results from all workers...")
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.sort_values(by=['date', 'district_no'], inplace=True)

    logging.info(f"Saving final daily weather data to {OUTPUT_CSV_PATH}")
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.4f')

    # Clean up the temp directory
    if TEMP_RASTER_DIR.exists():
        os.rmdir(TEMP_RASTER_DIR)

    logging.info("--- SUCCESS: Parallel processing complete! Daily weather file created. ---")


if __name__ == "__main__":
    create_daily_weather_file_parallel()