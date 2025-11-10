# File: build_weather_data.py
# Description: A robust module to generate daily historical weather for all districts.
# Version 2: Uses a nearest-neighbor (centroid) approach instead of zonal_stats
#            to correctly handle small districts (like Herne, 05916).
# - Refactored to use central configuration.

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import logging
from tqdm import tqdm
import multiprocessing
import os
import sys

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global variable for worker processes ---
worker_data = None


def init_worker(geojson_path):
    """
    Initializes each worker process by loading the GeoDataFrame
    and calculating the district centroids for nearest-neighbor lookup.
    """
    global worker_data
    try:
        gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
        gdf['centroid'] = gdf.geometry.centroid
        district_centroids = [{'id': row['id'], 'lon': row['centroid'].x, 'lat': row['centroid'].y} for _, row in gdf.iterrows()]
        worker_data = {'districts': district_centroids}
        logging.info(f"Worker {os.getpid()} initialized with {len(district_centroids)} districts.")
    except Exception as e:
        logging.error(f"Worker {os.getpid()} failed to initialize: {e}", exc_info=True)
        worker_data = None


def process_day(day_str):
    """
    Processes a single day for all districts using the nearest-neighbor method.
    """
    global worker_data
    if worker_data is None:
        logging.warning(f"Worker {os.getpid()} not initialized. Skipping day {day_str}.")
        return None

    day = pd.to_datetime(day_str)
    year = day.year
    day_results_list = []

    try:
        daily_data_slices = {}
        for csv_col, (filename_suffix, nc_var) in config.WEATHER_VARIABLE_MAP.items():
            nc_filepath = config.AGERA5_CONSOLIDATED_DIR / f"agera5_germany_{year}_{filename_suffix}.nc.nc"
            if not nc_filepath.exists():
                logging.warning(f"Missing file for {csv_col} on {day_str}. Will be empty.")
                daily_data_slices[csv_col] = None
                continue
            with xr.open_dataset(nc_filepath, use_cftime=True) as ds:
                daily_data_slices[csv_col] = ds.sel(time=day, method='nearest')[nc_var]

        for district in worker_data['districts']:
            district_no = district['id']
            lon = district['lon']
            lat = district['lat']
            district_stats = {'date': day, 'district_no': district_no}
            for csv_col, data_array in daily_data_slices.items():
                if data_array is None:
                    district_stats[csv_col] = None
                    continue
                try:
                    value = data_array.sel(lon=lon, lat=lat, method='nearest').item()
                except Exception:
                    value = None
                if csv_col in ['tmin', 'tmax']:
                    district_stats[csv_col] = value - 273.15 if value is not None else None
                elif csv_col == 'precip':
                    district_stats[csv_col] = value
                elif csv_col == 'srad':
                    district_stats[csv_col] = value / 1_000_000 if value is not None else None
                elif csv_col == 'wind':
                    district_stats[csv_col] = value
                elif csv_col == 'vap':
                    if value is not None:
                        dewpoint_c = value - 273.15
                        vap_kpa = 0.61094 * np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
                        district_stats[csv_col] = vap_kpa
                    else:
                        district_stats[csv_col] = None
                else:
                    district_stats[csv_col] = value
            day_results_list.append(district_stats)
        return pd.DataFrame(day_results_list)
    except Exception as e:
        logging.error(f"Failed to process day {day_str}: {e}", exc_info=True)
        return None


def create_daily_weather_file_for_year(year_to_process):
    """Main controlling function to generate the daily weather CSV for a specific year."""
    output_csv_path = config.DAILY_WEATHER_DIR / f"historical_daily_weather_era5_{year_to_process}.csv"
    logging.info(f"--- Starting: Daily Weather Creation for Year: {year_to_process} ---")
    if output_csv_path.exists():
        logging.info(f"Output file {output_csv_path.name} already exists. Skipping generation.")
        return output_csv_path

    config.DAILY_WEATHER_DIR.mkdir(exist_ok=True)
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Using {num_workers} worker processes.")
    time_steps_for_year = pd.to_datetime(pd.date_range(start=f'{year_to_process}-01-01', end=f'{year_to_process}-12-31', freq='D')).strftime('%Y-%m-%d')
    all_results = []
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(config.DISTRICTS_GEOJSON_PATH,)) as pool:
        results_iterator = pool.imap_unordered(process_day, time_steps_for_year)
        for result_df in tqdm(results_iterator, total=len(time_steps_for_year), desc=f"Processing {year_to_process} Weather Data"):
            if result_df is not None:
                all_results.append(result_df)
    if not all_results:
        logging.error(f"FATAL: No data was processed for {year_to_process}.")
        return None

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.sort_values(by=['date', 'district_no'], inplace=True)
    null_rows = final_df[final_df['tmax'].isnull()].shape[0]
    if null_rows > 0:
        logging.warning(f"Found {null_rows} rows with null data. This might be due to missing .nc files.")
    logging.info(f"Saving data for {year_to_process} to {output_csv_path.name}")
    final_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    logging.info(f"--- SUCCESS: Weather file created for {year_to_process}! ---")
    return output_csv_path


if __name__ == "__main__":
    logging.info("Running in standalone mode to generate data for a defined year range.")
    for year in range(config.WEATHER_START_YEAR, config.WEATHER_END_YEAR + 1):
        try:
            create_daily_weather_file_for_year(year)
        except Exception as e:
            logging.error(f"--- FAILED to process year {year}. Error: {e} ---")
            logging.info("Continuing to the next year.")
            continue
    logging.info("--- Standalone data generation complete. ---")
