# File: src/data/04_create_daily_weather_file_ONE_YEAR_TEST.py (Corrected Version)
# Description: This version reads directly from the "consolidated" intermediate files
#              produced by the unchanged '03_process...' script, bypassing the flawed
#              final merged file.

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- TEST CONFIGURATION ---
TEST_YEAR = 2018

# --- Configuration ---
BASE_DIR = Path.cwd()
# *** KEY CHANGE ***: Point to the directory of good, intermediate files.
AGERA5_CONSOLIDATED_DIR = BASE_DIR / "data/02_intermediate/consolidated_agera5"
GEOJSON_PATH = BASE_DIR / "data/01_raw/districts_official.geojson"
OUTPUT_CSV_PATH = BASE_DIR / f"data/02_intermediate/historical_daily_weather_era5_{TEST_YEAR}_TEST.csv"
TEMP_RASTER_DIR = BASE_DIR / "data/02_intermediate/temp_rasters"

# *** KEY CHANGE ***: Map the final CSV column to the FILENAME SUFFIX and the NC VARIABLE name
VARIABLE_MAP = {
    # csv_col: (filename_suffix, nc_variable_name)
    'tmin':   ('temp_minimum', 'Temperature_Air_2m_Min_24h'),
    'tmax':   ('temp_maximum', 'Temperature_Air_2m_Max_24h'),
    'precip': ('precipitation_flux', 'Precipitation_Flux'),
    'srad':   ('solar_radiation_flux', 'Solar_Radiation_Flux'),
}

# --- Worker Functions ---
gdf = None

def init_worker(geojson_path):
    """Initializes the worker with the GeoDataFrame."""
    global gdf
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")

def process_day(day_str):
    """
    Processes a single day by opening the FOUR separate NetCDF files for that year,
    extracting the data for the day, and calculating zonal stats.
    """
    day = pd.to_datetime(day_str)
    year = day.year
    pid = os.getpid()
    temp_raster_path = TEMP_RASTER_DIR / f"temp_raster_{pid}.tif"

    try:
        # This will hold the results for all variables for this one day
        district_daily_stats = {'date': day, 'district_no': gdf['id'].tolist()}

        # Loop through the four variables we need for the crop model
        for csv_col, (filename_suffix, nc_var) in VARIABLE_MAP.items():
            # Construct the path to the correct file for this variable and year
            nc_filepath = AGERA5_CONSOLIDATED_DIR / f"agera5_germany_{year}_{filename_suffix}.nc.nc"

            if not nc_filepath.exists():
                logging.warning(f"File not found for {csv_col} on {day_str}: {nc_filepath}")
                district_daily_stats[csv_col] = [None] * len(gdf)
                continue

            # Open the single-variable file and select the data for the current day
            with xr.open_dataset(nc_filepath) as ds:
                daily_slice = ds.sel(time=day, method='nearest')
                data_array = daily_slice[nc_var]

                # --- The rest of the logic is the same as before ---
                data_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True).rio.write_crs("EPSG:4326", inplace=True)
                data_array.rio.to_raster(temp_raster_path, compress='LZW')
                stats = zonal_stats(gdf, str(temp_raster_path), stats="mean")

                mean_values = [s['mean'] for s in stats]
                if csv_col in ['tmin', 'tmax']:
                    converted_values = [v - 273.15 if v is not None else None for v in mean_values]
                elif csv_col == 'precip':
                    converted_values = [v * 86400 if v is not None else None for v in mean_values]
                elif csv_col == 'srad':
                    # Assuming original is W/m^2, convert to MJ/m^2/day
                    converted_values = [(v * 86400) / 1_000_000 if v is not None else None for v in mean_values]
                else:
                    converted_values = mean_values
                district_daily_stats[csv_col] = converted_values

        return pd.DataFrame(district_daily_stats)
    except Exception as e:
        logging.error(f"Failed to process day {day_str}: {e}", exc_info=True)
        return None
    finally:
        if temp_raster_path.exists(): temp_raster_path.unlink()


def create_daily_weather_file_parallel_test():
    logging.info(f"--- Starting: Daily Weather Creation (Year: {TEST_YEAR}) from Consolidated Files ---")
    if OUTPUT_CSV_PATH.exists():
        logging.info(f"Test output file {OUTPUT_CSV_PATH} already exists. Skipping.")
        return

    TEMP_RASTER_DIR.mkdir(exist_ok=True)
    num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Generate the list of days to process for the test year
    time_steps_for_year = pd.to_datetime(pd.date_range(start=f'{TEST_YEAR}-01-01', end=f'{TEST_YEAR}-12-31', freq='D')).strftime('%Y-%m-%d')
    logging.info(f"Found {len(time_steps_for_year)} days to process for {TEST_YEAR}.")

    all_results = []
    # Initialize workers with just the GeoJSON path
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(GEOJSON_PATH,)) as pool:
        results_iterator = pool.imap_unordered(process_day, time_steps_for_year)
        for result_df in tqdm(results_iterator, total=len(time_steps_for_year), desc=f"Processing {TEST_YEAR} Data"):
            if result_df is not None:
                all_results.append(result_df)

    if not all_results:
        logging.error("FATAL: No data was processed. Check input files and paths.")
        return

    logging.info("Concatenating results...")
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.sort_values(by=['date', 'district_no'], inplace=True)
    logging.info(f"Saving ONE-YEAR test data to {OUTPUT_CSV_PATH}")
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.4f')
    if TEMP_RASTER_DIR.exists(): os.rmdir(TEMP_RASTER_DIR)
    logging.info(f"--- SUCCESS: ONE-YEAR test file created for {TEST_YEAR}! ---")


if __name__ == "__main__":
    create_daily_weather_file_parallel_test()