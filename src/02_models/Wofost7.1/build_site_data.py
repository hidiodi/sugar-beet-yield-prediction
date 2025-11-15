# File: src/02_models/Wofost7.1/build_site_data.py
# Description: This is a "glue" script. It takes the output from the GEE-based
# physics script (build_static_features.py) and merges it with geographic data
# (lat/lon) and historical yield data to produce the final StaticSiteData.csv
# that the main pipeline actually consumes.
#
# ENHANCEMENT: Changed merge strategy from 'left' to 'inner' to automatically
# filter out districts from yield data that are not present in the static
# soil/geo files, resolving data mismatches.

import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import logging

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
INTERMEDIATE_DATA_DIR = config.INTERMEDIATE_DATA_DIR
RAW_DATA_DIR = config.RAW_DATA_DIR

# --- Input File Paths ---
STATIC_SOIL_PHYSICS_PATH = PROCESSED_DATA_DIR / 'static_features_districts.csv'
DISTRICTS_GEOJSON_PATH = RAW_DATA_DIR / 'districts_official.geojson'
HISTORICAL_YIELD_PATH = INTERMEDIATE_DATA_DIR / 'sugarbeet_yield.csv'

# --- Output File Path ---
FINAL_OUTPUT_PATH = PROCESSED_DATA_DIR / 'StaticSiteData.csv'


def main():
    """
    Merges static soil physics, geo-coordinates, and historical yields
    into the final StaticSiteData.csv file.
    """
    logging.info("--- Starting Build Process for StaticSiteData.csv ---")

    # --- 1. Load all required input files ---
    logging.info("Loading input files...")
    try:
        df_soil_physics = pd.read_csv(STATIC_SOIL_PHYSICS_PATH, dtype={'district_no': str})
        gdf_districts = gpd.read_file(DISTRICTS_GEOJSON_PATH)
        df_yield = pd.read_csv(HISTORICAL_YIELD_PATH, dtype={'district_no': str})
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required input file was not found. Have you run all previous scripts? Error: {e}")
        sys.exit(1)

    # --- 2. Process Geodata to get Latitude and Longitude ---
    logging.info("Calculating district centroids for latitude and longitude...")
    gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    gdf_districts = gdf_districts.to_crs("EPSG:4236")
    gdf_districts['geometry'] = gdf_districts.geometry.centroid
    gdf_districts['longitude'] = gdf_districts.geometry.x
    gdf_districts['latitude'] = gdf_districts.geometry.y
    df_coords = gdf_districts[['district_no', 'latitude', 'longitude']]

    # --- 3. Prepare Yield and Soil Data ---
    df_yield.rename(columns={'yield': 'kreisYield'}, inplace=True)
    df_yield['district_no'] = df_yield['district_no'].astype(str).str.zfill(5)
    df_soil_physics['district_no'] = df_soil_physics['district_no'].astype(str).str.zfill(5)

    # --- 4. Merge all data sources using a robust INNER join ---
    # This change automatically filters for districts present in ALL input files.
    logging.info("Merging yield, soil physics, and coordinate data...")

    # Log initial counts for transparency
    initial_yield_districts = df_yield['district_no'].nunique()
    logging.info(f"Found {initial_yield_districts} unique districts in the initial yield data.")

    # Merge 1: Merge yield with soil data. 'inner' drops districts not in both.
    df_merged = pd.merge(df_yield, df_soil_physics, on='district_no', how='inner')

    # Merge 2: Merge the result with coordinate data. 'inner' continues the filtering.
    df_final = pd.merge(df_merged, df_coords, on='district_no', how='inner')

    # Log final counts to show the filtering result
    final_districts = df_final['district_no'].nunique()
    districts_dropped = initial_yield_districts - final_districts
    logging.info(f"After merging, {final_districts} districts remain.")
    if districts_dropped > 0:
        logging.warning(
            f"Dropped {districts_dropped} district(s) from yield data that did not have matching soil/geo data.")

    # --- 5. Final Validation and Save ---
    # Check if the merges introduced any missing data, which indicates a problem in the source files.
    if df_final.isnull().any().any():
        missing_rows = df_final[df_final.isnull().any(axis=1)]
        logging.warning(f"WARNING: {len(missing_rows)} rows have missing data after merging. "
                        f"This should not happen with an inner join unless the source files contain nulls.")
        logging.warning("Example of rows with missing data:\n" + missing_rows.head().to_string())
    else:
        logging.info("Validation complete: No missing values in the final dataset.")

    # Reorder columns to be logical
    key_cols = ['district_no', 'year', 'kreisYield', 'latitude', 'longitude']
    physics_cols = [col for col in df_soil_physics.columns if col != 'district_no']
    final_cols = key_cols + physics_cols
    df_final = df_final[final_cols]

    logging.info(f"Saving final merged data to '{FINAL_OUTPUT_PATH}'")
    df_final.to_csv(FINAL_OUTPUT_PATH, index=False, float_format='%.4f')

    logging.info("--- SUCCESS: StaticSiteData.csv created! ---")
    print("\n--- StaticSiteData.csv Preview ---")
    print(df_final.head().to_string())


if __name__ == "__main__":
    main()