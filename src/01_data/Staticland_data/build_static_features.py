# src/features/build_static_features.py (PAWC VERSION)

import ee
import geemap
import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_static_features_pawc():
    """
    Generates a CSV of static features (elevation, PAWC) for each German district.
    PAWC (Plant Available Water Capacity) is calculated in-cloud using a
    pedotransfer function on the ISRIC SoilGrids dataset via Google Earth Engine.
    """
    logging.info("--- Starting Advanced Static Feature Generation (PAWC) using GEE ---")

    # --- 1. Initialize ---
    try:
        project_id = 'augmented-audio-471809-h3'  # Your Project ID
        geemap.ee.Initialize(project=project_id)
        logging.info(f"GEE initialized for project '{project_id}'.")
    except Exception as e:
        logging.error(f"FATAL: Could not initialize GEE. Error: {e}")
        return

    # --- 2. Paths ---
    path_districts_geo = Path("data/01_raw/districts_official.geojson")
    output_dir = Path("data/03_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / "static_features_districts_advanced.csv"

    if output_filepath.exists():
        logging.info(f"Advanced static features file at '{output_filepath}' already exists. Skipping.")
        return

    # --- 3. Load Geometries ---
    logging.info("Loading district geometries...")
    gdf_districts = gpd.read_file(path_districts_geo)
    gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    districts_ee = geemap.geopandas_to_ee(gdf_districts)

    # --- 4. Load GEE Datasets for PTF Calculation ---
    logging.info("Loading base soil and elevation datasets from GEE...")
    dem_image = ee.Image('USGS/SRTMGL1_003').select('elevation')

    # Load the necessary bands from the stable SoilGrids dataset
    clay_image = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean')
    silt_image = ee.Image("projects/soilgrids-isric/silt_mean").select('silt_0-5cm_mean')
    soc_image = ee.Image("projects/soilgrids-isric/soc_mean").select('soc_0-5cm_mean')

    # --- 5. Engineer PAWC Feature in the Cloud ---
    logging.info("Calculating Plant Available Water Capacity (PAWC) on the server...")

    # Unit Conversion: SoilGrids values are in g/kg or dg/kg. Convert to percent.
    # To get percent, we divide by 10.
    clay_percent = clay_image.divide(10)
    silt_percent = silt_image.divide(10)
    soc_percent = soc_image.divide(10)  # Soil Organic Carbon in %

    # Convert Soil Organic Carbon (SOC) to Soil Organic Matter (SOM)
    # The standard conversion factor is 1.724
    som_percent = soc_percent.multiply(1.724)

    # Apply the Pedotransfer Function (PTF) to estimate PAWC.
    # This is a simplified linear model for demonstration, based on common PTFs.
    # PAWC (in %) ≈ 0.45 * Silt(%) + 0.15 * Clay(%) + 0.25 * SOM(%)
    pawc_image = silt_percent.multiply(0.45) \
        .add(clay_percent.multiply(0.15)) \
        .add(som_percent.multiply(0.25))

    # --- 6. Combine Features and Run Zonal Statistics ---
    # Combine the final elevation and the *calculated* PAWC images
    static_images = dem_image.addBands(pawc_image).rename(['avg_elevation', 'avg_soil_pawc'])

    logging.info("Running zonal statistics for all static features...")
    stats_ee = static_images.reduceRegions(
        collection=districts_ee,
        reducer=ee.Reducer.mean(),
        scale=90
    )

    # --- 7. Download and Save ---
    logging.info("Downloading results...")
    stats_df = geemap.ee_to_df(stats_ee)

    final_df = stats_df[['district_no', 'avg_elevation', 'avg_soil_pawc']]
    final_df.dropna(inplace=True)

    logging.info(f"Saving final static features to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False)

    logging.info("--- SUCCESS: Advanced static features file (with PAWC) created! ---")
    print("\n--- Advanced Static Features Preview ---")
    print(final_df.head())


if __name__ == "__main__":
    build_static_features_pawc()