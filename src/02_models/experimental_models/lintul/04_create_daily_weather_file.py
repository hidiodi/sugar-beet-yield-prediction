# File: build_weather_data.py
# Description: A reusable module to generate the daily historical weather file for a given year.
#              Refactored from the original standalone script to be callable from a master pipeline.

import numpy as np
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

# --- Configuration ---
BASE_DIR = Path.cwd()
AGERA5_CONSOLIDATED_DIR = BASE_DIR / "data/02_intermediate/consolidated_agera5"
GEOJSON_PATH = BASE_DIR / "data/01_raw/districts_official.geojson"
OUTPUT_DIR = BASE_DIR / "data/02_intermediate/daily_weather"
TEMP_RASTER_DIR = BASE_DIR / "data/02_intermediate/temp_rasters"

VARIABLE_MAP = {
    'tmin': ('temp_minimum', 'Temperature_Air_2m_Min_24h'),
    'tmax': ('temp_maximum', 'Temperature_Air_2m_Max_24h'),
    'precip': ('precipitation_flux', 'Precipitation_Flux'),
    'srad': ('solar_radiation_flux', 'Solar_Radiation_Flux'),
    'wind': ('wind_speed_mean', 'Wind_Speed_10m_Mean_24h'),
    'vap': ('dewpoint_temp_mean', 'Dew_Point_Temperature_2m_Mean_24h'),
}

gdf = None


def init_worker(geojson_path):
    global gdf
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")


def process_day(day_str):
    day = pd.to_datetime(day_str)
    year = day.year
    pid = os.getpid()
    temp_raster_path = TEMP_RASTER_DIR / f"temp_raster_{pid}.tif"

    try:
        district_daily_stats = {'date': day, 'district_no': gdf['id'].tolist()}
        for csv_col, (filename_suffix, nc_var) in VARIABLE_MAP.items():
            nc_filepath = AGERA5_CONSOLIDATED_DIR / f"agera5_germany_{year}_{filename_suffix}.nc.nc"
            if not nc_filepath.exists():
                district_daily_stats[csv_col] = [None] * len(gdf)
                continue

            with xr.open_dataset(nc_filepath) as ds:
                daily_slice = ds.sel(time=day, method='nearest')
                data_array = daily_slice[nc_var]
                data_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True).rio.write_crs("EPSG:4326",
                                                                                                      inplace=True)
                data_array.rio.to_raster(temp_raster_path, compress='LZW')
                stats = zonal_stats(gdf, str(temp_raster_path), stats="mean")
                mean_values = [s['mean'] for s in stats]

                if csv_col in ['tmin', 'tmax']:
                    converted_values = [v - 273.15 if v is not None else None for v in mean_values]
                elif csv_col == 'precip':
                    converted_values = [v * 86400 if v is not None else None for v in mean_values]
                elif csv_col == 'srad':
                    converted_values = [(v * 86400) / 1_000_000 if v is not None else None for v in mean_values]
                elif csv_col == 'wind':
                    converted_values = mean_values
                elif csv_col == 'vap':
                    converted_values = []
                    for v in mean_values:
                        if v is not None:
                            dewpoint_c = v - 273.15
                            vap_kpa = 0.61094 * np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
                            converted_values.append(vap_kpa)
                        else:
                            converted_values.append(None)
                else:
                    converted_values = mean_values
                district_daily_stats[csv_col] = converted_values
        return pd.DataFrame(district_daily_stats)
    except Exception as e:
        logging.error(f"Failed to process day {day_str}: {e}", exc_info=True)
        return None
    finally:
        if temp_raster_path.exists(): temp_raster_path.unlink()


def create_daily_weather_file_for_year(year_to_process):
    """Main controlling function to generate the daily weather CSV for a specific year."""
    output_csv_path = OUTPUT_DIR / f"historical_daily_weather_era5_{year_to_process}.csv"

    logging.info(f"--- Starting: Daily Weather Creation for Year: {year_to_process} ---")
    if output_csv_path.exists():
        logging.info(f"Output file {output_csv_path.name} already exists. Skipping generation.")
        return output_csv_path

    TEMP_RASTER_DIR.mkdir(exist_ok=True)
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    time_steps_for_year = pd.to_datetime(
        pd.date_range(start=f'{year_to_process}-01-01', end=f'{year_to_process}-12-31', freq='D')).strftime('%Y-%m-%d')

    all_results = []
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(GEOJSON_PATH,)) as pool:
        results_iterator = pool.imap_unordered(process_day, time_steps_for_year)
        for result_df in tqdm(results_iterator, total=len(time_steps_for_year),
                              desc=f"Processing {year_to_process} Weather Data"):
            if result_df is not None:
                all_results.append(result_df)

    if not all_results:
        logging.error(f"FATAL: No data was processed for {year_to_process}.")
        return None

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.sort_values(by=['date', 'district_no'], inplace=True)
    logging.info(f"Saving data for {year_to_process} to {output_csv_path.name}")
    final_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    if TEMP_RASTER_DIR.exists():
        try:
            os.rmdir(TEMP_RASTER_DIR)
        except OSError:
            pass  # Directory not empty, which is fine

    logging.info(f"--- SUCCESS: Weather file created for {year_to_process}! ---")
    return output_csv_path


if __name__ == "__main__":
    # This allows the script to be run standalone to pre-generate data for a range of years.
    logging.info("Running in standalone mode to generate data for a defined year range.")

    # --- DEFINE THE RANGE OF YEARS YOU WANT TO PROCESS ~10 minutes per year ---
    START_YEAR_TO_RUN = 1981
    END_YEAR_TO_RUN = 1999  # Inclusive

    for year in range(START_YEAR_TO_RUN, END_YEAR_TO_RUN + 1):
        try:
            create_daily_weather_file_for_year(year)
        except Exception as e:
            logging.error(f"--- FAILED to process year {year}. Error: {e} ---")
            logging.info("Continuing to the next year.")
            continue

    logging.info("--- Standalone data generation complete. ---")
