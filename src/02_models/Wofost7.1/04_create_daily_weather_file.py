# File: build_weather_data_v3.py
# Description: V3.3 - Generates daily historical weather using zonal statistics.
#              FIX: Cleans up logging by reporting fallback usage only once per district
#              to reduce repetitive and unhelpful log spam.

import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
from pathlib import Path
import logging
from tqdm import tqdm
import multiprocessing
import sys
import numpy as np

# --- Setup Project Root and Logging ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
DISTRICTS_GEOJSON_PATH = config.DISTRICTS_GEOJSON_PATH
AGERA5_CONSOLIDATED_DIR = config.AGERA5_CONSOLIDATED_DIR
OUTPUT_WEATHER_DIR = project_root / "data" / "04_feature" / "weather_district_daily"


def process_district(district_info):
    """
    Worker function to process all years of weather data for a single district.
    """
    district_id = district_info['id']
    district_geom = district_info['geometry']
    log_prefix = f"[District {district_id}]"

    processed_count = 0
    fallback_count = 0  # Counter for logging
    total_years = len(range(config.WEATHER_START_YEAR, config.WEATHER_END_YEAR + 1))

    for year in range(config.WEATHER_START_YEAR, config.WEATHER_END_YEAR + 1):
        try:
            output_dir_year = OUTPUT_WEATHER_DIR / str(year)
            output_dir_year.mkdir(parents=True, exist_ok=True)
            output_csv_path = output_dir_year / f"{district_id}_{year}.csv"

            if output_csv_path.exists():
                continue

            yearly_datasets = []
            all_files_exist = True
            for csv_col, (filename_suffix, nc_var) in config.WEATHER_VARIABLE_MAP.items():
                nc_filepath = AGERA5_CONSOLIDATED_DIR / f"agera5_germany_{year}_{filename_suffix}.nc.nc"
                if not nc_filepath.exists():
                    all_files_exist = False
                    break
                yearly_datasets.append(xr.open_dataset(nc_filepath, engine='netcdf4'))

            if not all_files_exist:
                continue

            with xr.merge(yearly_datasets) as ds:
                ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")

                try:
                    clipped_ds = ds.rio.clip([district_geom], ds.rio.crs, drop=True)
                    district_daily_mean = clipped_ds.mean(dim=["lat", "lon"], keep_attrs=True)
                except rioxarray.exceptions.NoDataInBounds:
                    # Increment counter instead of logging immediately
                    fallback_count += 1
                    centroid = district_geom.centroid
                    district_daily_mean = ds.sel(lon=centroid.x, lat=centroid.y, method='nearest')

                df = district_daily_mean.to_dataframe()
                df.reset_index(inplace=True)

                rename_map = {v[1]: k for k, v in config.WEATHER_VARIABLE_MAP.items()}
                df.rename(columns=rename_map, inplace=True)

                if 'tmin' in df.columns: df['tmin'] -= 273.15
                if 'tmax' in df.columns: df['tmax'] -= 273.15
                if 'precip' in df.columns: df['precip'] *= 86400
                if 'srad' in df.columns: df['srad'] = (df['srad'] * 86400) / 1_000_000
                if 'vap' in df.columns:
                    dewpoint_c = df['vap'] - 273.15
                    df['vap'] = 0.61094 * np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))

                df['district_no'] = district_id
                df.rename(columns={'time': 'date'}, inplace=True)
                df = df.drop(columns=[col for col in ['lon', 'lat', 'spatial_ref'] if col in df.columns],
                             errors='ignore')

                final_cols = ['date', 'district_no'] + [key for key in config.WEATHER_VARIABLE_MAP.keys() if
                                                        key in df.columns]
                df = df[final_cols]

                df.to_csv(output_csv_path, index=False, float_format='%.4f')

            processed_count += 1

        except Exception as e:
            logging.error(f"{log_prefix} FAILED to process year {year}. Error: {e}", exc_info=True)
            continue

    # --- CLEAN LOGGING: Report summary after all years are processed for this district ---
    if fallback_count > 0:
        logging.warning(
            f"{log_prefix} Used nearest neighbor fallback for {fallback_count}/{total_years} years due to small geometry.")

    return processed_count


def main():
    """Main controlling function to generate all daily weather files."""
    logging.info("--- Starting V3.3 Daily Weather Generation (Clean Logging) ---")
    OUTPUT_WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading district geometries...")
    try:
        gdf_districts = gpd.read_file(DISTRICTS_GEOJSON_PATH).to_crs("EPSG:4326")
        gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
    except Exception as e:
        logging.error(f"FATAL: Could not load districts GeoJSON. Error: {e}")
        return

    tasks = [{'id': row['district_no'], 'geometry': row['geometry']} for _, row in gdf_districts.iterrows()]

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    logging.info(f"Distributing {len(tasks)} districts across {num_workers} worker processes...")

    total_files_processed = 0
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(process_district, tasks)
        pbar = tqdm(results_iterator, total=len(tasks), desc="Processing Districts")
        for num_processed in pbar:
            total_files_processed += num_processed

    logging.info("--- V3.3 Weather Generation Complete! ---")
    logging.info(f"Successfully generated/verified {total_files_processed} district-year weather files.")
    logging.info(f"Output files are located in: {OUTPUT_WEATHER_DIR}")


if __name__ == "__main__":
    main()