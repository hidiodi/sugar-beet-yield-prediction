# File: src/02_models/Wofost7.1/calculate_initial_conditions.py
# Description: V2.3 - Calculates WAV from ERA5-Land data.
#              FIX: Implements a robust fallback using nearest-neighbor selection for
#              small districts where the primary clip-and-mean method fails.

import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from pathlib import Path
import sys
import logging
from tqdm import tqdm
import re
import zipfile
import tempfile

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ERA5_SOIL_DATA_DIR = Path("data/01_raw/era5_land_monthly_soil")


def calculate_initial_wav_from_era5():
    """
    Main function to calculate WAV for all districts and years using ERA5-Land data.
    """
    logging.info("--- Starting V2.3 Initial Available Water (WAV) Calculation (with Fallback) ---")

    output_path = config.WOFOST_CONFIG['FILE_PATHS']['INITIAL_CONDITIONS']
    if output_path.exists():
        logging.info(f"Initial conditions file already exists at '{output_path}'. Skipping.")
        return

    logging.info("Loading static soil features and district geometries...")
    try:
        static_df = pd.read_csv(
            config.WOFOST_CONFIG['FILE_PATHS']['STATIC_SOIL_FEATURES'],
            dtype={'district_no': str}
        )
        if static_df['RDMSOL'].mean() < 10:
            static_df['RDMSOL'] = static_df['RDMSOL'] * 100

        gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH)
        gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        gdf_districts = gdf_districts.to_crs("EPSG:4326")
    except FileNotFoundError as e:
        logging.error(f"FATAL: Required data file not found. Error: {e}");
        sys.exit(1)

    era5_files = sorted(list(ERA5_SOIL_DATA_DIR.glob("*_FEBRUARY.nc")))
    if not era5_files:
        logging.error(f"FATAL: No ERA5-Land NetCDF files found in '{ERA5_SOIL_DATA_DIR}'.");
        sys.exit(1)

    all_years_results = []
    pbar = tqdm(era5_files, desc="Processing ERA5-Land Files")
    for nc_file in pbar:
        year_match = re.search(r'_(\d{4})_FEBRUARY', nc_file.name)
        if not year_match:
            logging.warning(f"Could not parse year from filename: {nc_file.name}. Skipping.");
            continue
        year = int(year_match.group(1))
        pbar.set_postfix_str(f"Year: {year}")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                actual_nc_path = nc_file
                if zipfile.is_zipfile(nc_file):
                    with zipfile.ZipFile(nc_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    extracted_files = list(Path(temp_dir).glob('*.nc'))
                    if not extracted_files:
                        logging.warning(f"No .nc file found inside zip {nc_file.name}. Skipping.");
                        continue
                    actual_nc_path = extracted_files[0]

                with xr.open_dataset(actual_nc_path, engine='netcdf4') as rds_raw:
                    data_array = rds_raw['swvl1'].squeeze(drop=True)
                    data_array = data_array.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
                    data_array = data_array.rio.write_crs("EPSG:4326")

                    zonal_stats_results = []
                    for _, district in gdf_districts.iterrows():
                        mean_swvl1 = None
                        try:
                            # --- METHOD 1: Accurate Clip-and-Mean ---
                            clipped = data_array.rio.clip([district.geometry], gdf_districts.crs, drop=True)
                            mean_swvl1 = clipped.mean().item()
                        except rioxarray.exceptions.NoDataInBounds:
                            # --- METHOD 2: Robust Nearest-Neighbor Fallback ---
                            logging.warning(
                                f"NoDataInBounds for district {district['district_no']} in {year}. Using nearest neighbor fallback.")
                            centroid = district.geometry.centroid
                            mean_swvl1 = data_array.sel(longitude=centroid.x, latitude=centroid.y,
                                                        method='nearest').item()

                        if pd.notna(mean_swvl1):
                            zonal_stats_results.append({
                                'district_no': district['district_no'],
                                'mean_feb_swvl1': mean_swvl1
                            })

                    if zonal_stats_results:
                        year_swvl1_df = pd.DataFrame(zonal_stats_results)
                        year_swvl1_df['year'] = year
                        all_years_results.append(year_swvl1_df)

        except Exception as e:
            logging.error(f"CRITICAL FAILURE processing {nc_file.name}. Error: {e}", exc_info=True);
            continue

    if not all_years_results:
        logging.error("FATAL: No soil moisture data could be processed. Aborting.");
        return

    logging.info("Aggregating yearly results and calculating final WAV...")
    full_swvl1_df = pd.concat(all_years_results, ignore_index=True)
    final_df = pd.merge(full_swvl1_df, static_df[['district_no', 'SMW', 'RDMSOL']], on='district_no')
    final_df['WAV'] = (final_df['mean_feb_swvl1'] - final_df['SMW']) * final_df['RDMSOL']
    final_df['WAV'] = final_df['WAV'].clip(lower=0)

    output_df = final_df[['year', 'district_no', 'WAV']]
    logging.info(f"Saving {len(output_df)} calculated WAV records to '{output_path}'")
    output_df.to_csv(output_path, index=False, float_format='%.4f')

    logging.info("--- SUCCESS: V2.3 Initial conditions (WAV) file created from ERA5-Land data! ---")
    print("\n--- Final WAV Data Preview ---");
    print(output_df.head())
    print("\n--- Summary Statistics for Calculated WAV (cm) ---");
    print(output_df['WAV'].describe())


if __name__ == "__main__":
    calculate_initial_wav_from_era5()