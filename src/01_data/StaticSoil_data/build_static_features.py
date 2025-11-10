# File: src/features/build_static_features.py
# Description: V3 - CORRECTED VERSION. Builds a comprehensive set of static soil and topography features.
# - Fixes KeyError by robustly handling band names from GEE's toBands() function.
# - Refactored to use central configuration.

import ee
import geemap
import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
import time
import sys

# Ensure the project root is in the Python path to allow for `src` imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime=)s - %(levelname)s - %(message)s')


def build_static_features_v3():
    """
    Generates a CSV of V3 static features for each German district using GEE.
    Features include masked elevation, slope, and multi-layered, depth-weighted
    soil properties from SoilGrids. All parameters are now sourced from src.config.
    """
    logging.info("--- Starting V3 Static Feature Generation using GEE (Config-driven) ---")

    # --- 1. Initialize GEE ---
    try:
        geemap.ee.Initialize(project=config.GEE_PROJECT_ID, opt_url=config.GEE_HIGH_VOLUME_ENDPOINT)
        logging.info(f"GEE initialized for project '{config.GEE_PROJECT_ID}'. Using high-volume endpoint.")
    except Exception as e:
        logging.error(f"FATAL: Could not initialize GEE. Error: {e}")
        return

    # --- 2. Define Paths and Configuration from src.config ---
    output_dir = config.PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = config.STATIC_FEATURES_OUTPUT_PATH

    if output_filepath.exists():
        logging.info(f"V3 static features file at '{output_filepath}' already exists. Skipping.")
        return

    # --- 3. Load Geometries and Mask ---
    logging.info("Loading district geometries...")
    gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH)
    gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    districts_ee = geemap.geopandas_to_ee(gdf_districts)

    try:
        logging.info(f"Loading farmland mask from GEE asset: {config.FARMLAND_MASK_ASSET_ID}")
        farmland_mask_image = ee.Image(config.FARMLAND_MASK_ASSET_ID)
    except ee.ee_exception.EEException:
        logging.error(f"FATAL: GEE Asset not found at '{config.FARMLAND_MASK_ASSET_ID}'.")
        return

    # --- 4. Load GEE Datasets ---
    logging.info("Loading base topography and SoilGrids datasets from GEE...")
    dem_image = ee.Image(config.DEM_IMAGE).select('elevation')
    slope_image = ee.Terrain.slope(dem_image)

    # --- 5. Extract Multi-Layer Soil Data with Masking ---
    logging.info("Extracting multi-layer soil properties with farmland mask...")

    image_list = [dem_image.rename('avg_elevation'), slope_image.rename('avg_slope')]

    for prop in config.SOIL_PROPERTIES:
        for depth in config.SOIL_DEPTHS:
            image_id = f"projects/soilgrids-isric/{prop}_mean"
            band_name = f"{prop}_{depth}_mean"
            image = ee.Image(image_id).select(band_name)
            clean_name = f"{prop}_{depth.replace('-', '_')}"
            image_list.append(image.rename(clean_name))

    combined_images = ee.Image.cat(image_list)
    masked_data = combined_images.updateMask(farmland_mask_image)

    # --- 6. Run Zonal Statistics ---
    logging.info("Running zonal statistics for all features...")
    stats_ee = masked_data.reduceRegions(
        collection=districts_ee,
        reducer=ee.Reducer.mean(),
        scale=250
    )

    # --- 7. Download and Process Results ---
    logging.info("Downloading results from GEE... (This may take several minutes)")
    start_time = time.time()
    stats_df = geemap.ee_to_df(stats_ee)
    end_time = time.time()
    logging.info(f"Download complete in {end_time - start_time:.2f} seconds.")

    stats_df = stats_df.drop(columns=['geometry', 'system:index'], errors='ignore')
    stats_df.dropna(subset=['district_no'], inplace=True)

    # --- 8. Engineer Depth-Weighted Features ---
    logging.info("Engineering depth-weighted average soil features...")

    # Unit Conversions
    for depth in config.SOIL_DEPTHS:
        d_str = depth.replace('-', '_')
        if f'bdod_{d_str}' in stats_df.columns:
            stats_df[f'bdod_{d_str}'] /= 100
        if f'clay_{d_str}' in stats_df.columns:
            stats_df[f'clay_{d_str}'] /= 10
        if f'sand_{d_str}' in stats_df.columns:
            stats_df[f'sand_{d_str}'] /= 10
        if f'phh2o_{d_str}' in stats_df.columns:
            stats_df[f'phh2o_{d_str}'] /= 10
        if f'soc_{d_str}' in stats_df.columns:
            stats_df[f'soc_{d_str}'] = (stats_df[f'soc_{d_str}'] / 100) * 1.724
            stats_df.rename(columns={f'soc_{d_str}': f'som_{d_str}'}, inplace=True)

    # Calculate weighted average for topsoil (0-30cm)
    logging.info(" -> Calculating topsoil (0-30cm) properties...")
    total_thickness_30 = sum(config.LAYER_THICKNESS[d] for d in config.TOPSOIL_LAYERS)
    for prop in ['bdod', 'clay', 'sand', 'som', 'phh2o']:
        weighted_sum = sum(stats_df[f'{prop}_{d}'] * config.LAYER_THICKNESS[d]
                           for d in config.TOPSOIL_LAYERS if f'{prop}_{d}' in stats_df.columns)
        stats_df[f'avg_{prop}_0_30cm'] = weighted_sum / total_thickness_30

    # Calculate weighted average for the full root zone (0-100cm)
    logging.info(" -> Calculating full root zone (0-100cm) properties...")
    total_thickness_100 = sum(config.LAYER_THICKNESS[d] for d in config.ROOTZONE_LAYERS)
    for prop in ['bdod', 'clay', 'sand', 'som', 'phh2o']:
        weighted_sum = sum(stats_df[f'{prop}_{d}'] * config.LAYER_THICKNESS[d]
                           for d in config.ROOTZONE_LAYERS if f'{prop}_{d}' in stats_df.columns)
        stats_df[f'avg_{prop}_0_100cm'] = weighted_sum / total_thickness_100

    # --- 9. Finalize and Save ---
    final_feature_columns_exist = [col for col in config.FINAL_FEATURE_COLUMNS if col in stats_df.columns]
    final_df = stats_df[final_feature_columns_exist]

    logging.info(f"Saving final V3 static features to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False, float_format='%.4f')

    logging.info("--- SUCCESS: V3 static features file created! ---")
    print("\n--- V3 Static Features Preview ---")
    print(final_df.head())

if __name__ == "__main__":
    build_static_features_v3()
