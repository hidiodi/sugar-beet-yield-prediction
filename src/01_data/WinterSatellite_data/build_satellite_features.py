# src/features/build_satellite_features.py

import ee
import geemap
import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MODIS data is available from 2000-02-24 onwards.
# To ensure a full winter period, we start with the winter of 2000-2001 (Nov 2000 - Feb 2001),
# which corresponds to the 2001 harvest year.
START_YEAR = 2001
END_YEAR = 2024  # Or your last year of yield data
GEE_PROJECT_ID = 'augmented-audio-471809-h3'  # Your Project ID

PATH_DISTRICTS_GEO = Path("data/01_raw/districts_official.geojson")
OUTPUT_DIR = Path("data/03_processed")
OUTPUT_FILEPATH = OUTPUT_DIR / f"satellite_features_districts_{START_YEAR}-{END_YEAR}.csv"


def build_satellite_features():
    """
    Generates time-series satellite features for each German district from START_YEAR to END_YEAR.

    Features generated for the pre-season period (Nov 1 to Feb 28):
    1. Mean NDVI on cropland.
    2. Mean Land Surface Temperature (LST) on cropland.
    3. Mean number of snow-covered days on cropland.

    The script then calculates anomalies for NDVI and LST in a post-processing step.
    """
    logging.info("--- Starting Time-Series Satellite Feature Generation using GEE ---")

    # --- 1. Initialization and File Check ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILEPATH.exists():
        logging.info(f"Satellite features file at '{OUTPUT_FILEPATH}' already exists. Skipping.")
        return

    try:
        geemap.ee.Initialize(project=GEE_PROJECT_ID)
        logging.info(f"GEE initialized for project '{GEE_PROJECT_ID}'.")
    except Exception as e:
        logging.error(f"FATAL: Could not initialize GEE. Error: {e}")
        return

    # --- 2. Load Geometries and Cropland Mask ---
    logging.info("Loading district geometries...")
    gdf_districts = gpd.read_file(PATH_DISTRICTS_GEO)
    gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    districts_ee = geemap.geopandas_to_ee(gdf_districts)

    logging.info("Loading CORINE Land Cover to create a cropland mask...")
    corine = ee.Image('COPERNICUS/CORINE/V20/100m/2018').select('landcover')
    # Class 211 is "Non-irrigated arable land" - this is our mask.
    cropland_mask = corine.eq(211)

    # --- 3. Iterate Through Years to Generate Features ---
    all_years_results = []

    for year in range(START_YEAR, END_YEAR + 1):
        start_time_year = time.time()
        logging.info(f"--- Processing data for harvest year: {year} ---")

        # Define the winter period: November of the previous year to February of the current year.
        start_date = f'{year - 1}-11-01'
        end_date = f'{year}-02-28'

        # --- A. NDVI ---
        modis_sr = ee.ImageCollection('MODIS/061/MOD09A1').filterDate(start_date, end_date)
        ndvi_collection = modis_sr.map(
            lambda img: img.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']).rename('NDVI'))
        winter_ndvi_median = ndvi_collection.median().updateMask(cropland_mask)

        # --- B. Land Surface Temperature (LST) ---
        modis_lst = ee.ImageCollection('MODIS/061/MOD11A2').filterDate(start_date, end_date).select('LST_Day_1km')
        # Convert from scaled Kelvin to Celsius
        lst_celsius = modis_lst.map(lambda img: img.multiply(0.02).subtract(273.15).rename('LST'))
        winter_lst_mean = lst_celsius.mean().updateMask(cropland_mask)

        # --- C. Snow Cover ---
        modis_snow = ee.ImageCollection('MODIS/061/MOD10A1').filterDate(start_date, end_date).select('NDSI_Snow_Cover')
        # Create binary images (1 for snow, 0 for no-snow) and sum them up to get days of snow
        snow_days_collection = modis_snow.map(lambda img: img.gte(30))  # Common threshold for snow
        winter_snow_days_sum = snow_days_collection.sum().updateMask(cropland_mask)

        # --- D. Combine and Run Zonal Statistics ---
        combined_features = winter_ndvi_median.addBands(winter_lst_mean).addBands(winter_snow_days_sum) \
            .rename(['winter_cropland_ndvi_mean', 'winter_cropland_LST_mean', 'winter_cropland_snow_cover_days'])

        logging.info(f"Running zonal statistics for {year}...")
        stats_ee = combined_features.reduceRegions(
            collection=districts_ee,
            reducer=ee.Reducer.mean(),
            scale=500  # MODIS resolution is appropriate
        )

        # Download data for the current year and add a 'year' column
        df_year = geemap.ee_to_df(stats_ee)
        df_year['year'] = year
        all_years_results.append(df_year)

        logging.info(f"Finished processing {year} in {time.time() - start_time_year:.2f} seconds.")

    # --- 4. Post-Processing and Anomaly Calculation ---
    logging.info("All years processed. Combining results and calculating anomalies...")
    final_df = pd.concat(all_years_results, ignore_index=True)
    final_df.dropna(subset=['district_no'], inplace=True)  # Drop rows where district_no might be missing

    # Calculate long-term mean for each district to compute anomalies
    final_df['long_term_mean_ndvi'] = final_df.groupby('district_no')['winter_cropland_ndvi_mean'].transform('mean')
    final_df['long_term_mean_LST'] = final_df.groupby('district_no')['winter_cropland_LST_mean'].transform('mean')

    # Calculate the anomaly
    final_df['winter_cropland_ndvi_anomaly'] = final_df['winter_cropland_ndvi_mean'] - final_df['long_term_mean_ndvi']
    final_df['winter_cropland_LST_anomaly'] = final_df['winter_cropland_LST_mean'] - final_df['long_term_mean_LST']

    # --- 5. Finalize and Save ---
    # Select and reorder final columns for clarity
    output_columns = [
        'district_no',
        'year',
        'winter_cropland_ndvi_mean',
        'winter_cropland_ndvi_anomaly',
        'winter_cropland_LST_mean',
        'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days'
    ]
    final_df = final_df[output_columns]
    final_df.dropna(inplace=True)

    logging.info(f"Saving final satellite features to '{OUTPUT_FILEPATH}'")
    final_df.to_csv(OUTPUT_FILEPATH, index=False)

    logging.info("--- SUCCESS: Satellite features file created! ---")
    print("\n--- Satellite Features Preview ---")
    print(final_df.head())


if __name__ == "__main__":
    build_satellite_features()