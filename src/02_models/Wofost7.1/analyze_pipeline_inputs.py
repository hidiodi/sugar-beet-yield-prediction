# File: scripts/diagnose_sitedata_mismatch.py
# Description: A diagnostic tool to identify district ID mismatches between the
# input files used to build StaticSiteData.csv.
# VERSION 3: Simplified to use direct relative paths from the project root.

import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# --- Input File Paths ---
# Using the direct relative paths from the project root, as they should be.
# No more complex path finding.
PATHS = {
    'yield': 'data/02_intermediate/sugarbeet_yield.csv',
    'soil':  'data/03_processed/static_features_districts.csv',
    'geo':   'data/01_raw/districts_official.geojson'
}

def diagnose_mismatch():
    """
    Analyzes the input files and reports on district ID mismatches.
    """
    logging.info("--- Starting Diagnosis of Site Data Inputs ---")
    all_files_found = True
    data_sources = {}

    # --- 1. Load and Standardize District IDs from all sources ---
    for name, path_str in PATHS.items():
        # Convert the relative string path to a Path object for processing
        path = Path(path_str)
        logging.info(f"Loading '{name}' data from: {path}")

        if not path.exists():
            logging.error(f"FATAL: File not found: {path}")
            logging.error("Please ensure you are running this script from the project's root directory.")
            all_files_found = False
            continue

        try:
            if name == 'yield':
                df = pd.read_csv(path, dtype={'district_no': str})
                df['district_no'] = df['district_no'].str.zfill(5)
                data_sources[name] = {'df': df, 'ids': set(df['district_no'].unique())}

            elif name == 'soil':
                df = pd.read_csv(path, dtype={'district_no': str})
                df['district_no'] = df['district_no'].str.zfill(5)
                data_sources[name] = {'df': df, 'ids': set(df['district_no'].unique())}

            elif name == 'geo':
                gdf = gpd.read_file(path)
                if 'id' in gdf.columns:
                    gdf.rename(columns={'id': 'district_no'}, inplace=True)
                gdf['district_no'] = gdf['district_no'].astype(str).str.zfill(5)
                data_sources[name] = {'df': gdf, 'ids': set(gdf['district_no'].unique())}

        except Exception as e:
            logging.error(f"Failed to load or process file {path}. Error: {e}", exc_info=True)
            all_files_found = False

    if not all_files_found:
        logging.error("One or more essential files could not be loaded. Aborting diagnosis.")
        sys.exit(1)

    # --- 2. Perform the Analysis and Generate a Report ---
    logging.info("\n--- Mismatch Analysis Report ---")

    yield_ids = data_sources['yield']['ids']
    soil_ids = data_sources['soil']['ids']
    geo_ids = data_sources['geo']['ids']

    print(f"\n[SUMMARY] Unique District Counts:")
    print(f"  - Yield Data ('sugarbeet_yield.csv'):      {len(yield_ids)}")
    print(f"  - Soil Physics ('static_features_districts.csv'): {len(soil_ids)}")
    print(f"  - Geographic Data ('districts_official.geojson'): {len(geo_ids)}")

    missing_from_soil = sorted(list(yield_ids.difference(soil_ids)))
    print("\n[CHECK 1] Districts in Yield Data but MISSING from Soil Physics Data:")
    if missing_from_soil:
        print(f"  [PROBLEM] Found {len(missing_from_soil)} missing districts.")
        print(f"  List: {missing_from_soil}")
    else:
        print("  [OK] All districts from yield data are present in the soil physics file.")

    missing_from_geo = sorted(list(yield_ids.difference(geo_ids)))
    print("\n[CHECK 2] Districts in Yield Data but MISSING from Geographic (GeoJSON) Data:")
    if missing_from_geo:
        print(f"  [PROBLEM] Found {len(missing_from_geo)} missing districts.")
        print(f"  List: {missing_from_geo}")
    else:
        print("  [OK] All districts from yield data are present in the geojson file.")

    print("\n--- Diagnosis Conclusion ---")
    if not missing_from_soil and not missing_from_geo:
        print("[SUCCESS] All district IDs are consistent across the input files. The merge should succeed.")
    else:
        print("[ACTION REQUIRED] Mismatch detected. The `build_site_data.py` script is failing because of the district(s) listed above.")
        print("To fix this, you have two options:")
        print("  1. (Recommended) REMOVE the problematic district rows from 'data/02_intermediate/sugarbeet_yield.csv'.")
        print("  2. (Advanced) ADD the missing district data to 'static_features_districts.csv' and/or 'districts_official.geojson'.")
        print("After fixing the input files, re-run 'build_site_data.py'.")

if __name__ == "__main__":
    diagnose_mismatch()