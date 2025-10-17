# File: src/features/build_static_features.py
# Description: V3 - CORRECTED VERSION. Builds a comprehensive set of static soil and topography features.
# - Fixes KeyError by robustly handling band names from GEE's toBands() function.

import ee
import geemap
import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_static_features_v3():
    """
    Generates a CSV of V3 static features for each German district using GEE.
    Features include masked elevation, slope, and multi-layered, depth-weighted
    soil properties from SoilGrids.
    """
    logging.info("--- Starting V3 Static Feature Generation using GEE ---")

    # --- 1. Initialize GEE ---
    try:
        project_id = 'augmented-audio-471809-h3'
        geemap.ee.Initialize(project=project_id, opt_url='https://earthengine-highvolume.googleapis.com')
        logging.info(f"GEE initialized for project '{project_id}'. Using high-volume endpoint.")
    except Exception as e:
        logging.error(f"FATAL: Could not initialize GEE. Error: {e}")
        return

    # --- 2. Define Paths and Configuration ---
    path_districts_geo = Path("data/01_raw/districts_official.geojson")
    output_dir = Path("data/03_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / "static_features_districts.csv"

    if output_filepath.exists():
        logging.info(f"V3 static features file at '{output_filepath}' already exists. Skipping.")
        return

    # --- 3. Load Geometries and Mask ---
    logging.info("Loading district geometries...")
    gdf_districts = gpd.read_file(path_districts_geo)
    gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    districts_ee = geemap.geopandas_to_ee(gdf_districts)

    mask_asset_id = 'projects/augmented-audio-471809-h3/assets/farmland_mask_germany'
    try:
        logging.info(f"Loading farmland mask from GEE asset: {mask_asset_id}")
        farmland_mask_image = ee.Image(mask_asset_id)
    except ee.ee_exception.EEException:
        logging.error(f"FATAL: GEE Asset not found at '{mask_asset_id}'.")
        return

    # --- 4. Load GEE Datasets ---
    logging.info("Loading base topography and SoilGrids datasets from GEE...")
    dem_image = ee.Image('USGS/SRTMGL1_003').select('elevation')
    slope_image = ee.Terrain.slope(dem_image)

    soil_properties = ['bdod', 'clay', 'sand', 'soc', 'phh2o']
    depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']

    # --- 5. Extract Multi-Layer Soil Data with Masking ---
    logging.info("Extracting multi-layer soil properties with farmland mask...")

    image_list = [dem_image.rename('avg_elevation'), slope_image.rename('avg_slope')]

    for prop in soil_properties:
        for depth in depths:
            image_id = f"projects/soilgrids-isric/{prop}_mean"
            band_name = f"{prop}_{depth}_mean"
            image = ee.Image(image_id).select(band_name)
            # Create a clean, predictable name
            clean_name = f"{prop}_{depth.replace('-', '_')}"
            image_list.append(image.rename(clean_name))

    # ============================ THE FIX - Part 1 ============================
    # Instead of converting to bands immediately, we create a list of images
    # We will combine them into a single image but control the band names explicitly.
    # The `ee.Image.cat()` function is more reliable for this.
    combined_images = ee.Image.cat(image_list)
    # ==========================================================================

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
    for depth in depths:
        # ============================ THE FIX - Part 2 ============================
        # Use the correct, full column name format we created.
        d_str = depth.replace('-', '_')  # e.g., '0_5cm'
        # ==========================================================================

        # Check if the column exists before trying to convert it (robustness)
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

    layer_thickness = {'0_5cm': 5, '5_15cm': 10, '15_30cm': 15, '30_60cm': 30, '60_100cm': 40}

    # Calculate weighted average for topsoil (0-30cm)
    logging.info(" -> Calculating topsoil (0-30cm) properties...")
    topsoil_layers = ['0_5cm', '5_15cm', '15_30cm']
    total_thickness_30 = sum(layer_thickness[d] for d in topsoil_layers)
    for prop in ['bdod', 'clay', 'sand', 'som', 'phh2o']:
        # Use a list comprehension with a check to handle potentially missing columns
        weighted_sum = sum(stats_df[f'{prop}_{d}'] * layer_thickness[d]
                           for d in topsoil_layers if f'{prop}_{d}' in stats_df.columns)
        stats_df[f'avg_{prop}_0_30cm'] = weighted_sum / total_thickness_30

    # Calculate weighted average for the full root zone (0-100cm)
    logging.info(" -> Calculating full root zone (0-100cm) properties...")
    rootzone_layers = ['0_5cm', '5_15cm', '15_30cm', '30_60cm', '60_100cm']
    total_thickness_100 = sum(layer_thickness[d] for d in rootzone_layers)
    for prop in ['bdod', 'clay', 'sand', 'som', 'phh2o']:
        weighted_sum = sum(stats_df[f'{prop}_{d}'] * layer_thickness[d]
                           for d in rootzone_layers if f'{prop}_{d}' in stats_df.columns)
        stats_df[f'avg_{prop}_0_100cm'] = weighted_sum / total_thickness_100

    # --- 9. Finalize and Save ---
    final_feature_columns = [
        'district_no', 'avg_elevation', 'avg_slope',
        'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm',
        'avg_bdod_0_100cm', 'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm', 'avg_phh2o_0_100cm'
    ]
    final_feature_columns_exist = [col for col in final_feature_columns if col in stats_df.columns]
    final_df = stats_df[final_feature_columns_exist]

    logging.info(f"Saving final V3 static features to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False, float_format='%.4f')

    logging.info("--- SUCCESS: V3 static features file created! ---")
    print("\n--- V3 Static Features Preview ---")
    print(final_df.head())


if __name__ == "__main__":
    build_static_features_v3()