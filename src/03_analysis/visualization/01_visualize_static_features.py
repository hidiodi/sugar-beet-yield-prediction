# src/visualization/01_visualize_static_features.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def visualize_static_maps():
    """
    Creates a set of choropleth maps to visualize the geographic distribution of
    the new static features (Elevation and Plant Available Water Capacity).
    """
    logging.info("--- Starting Geographic Visualization of Static Features ---")

    # --- 1. Define File Paths ---
    # Use the new advanced static features file
    path_data = Path("data/03_processed/static_features_districts_advanced.csv")
    path_districts_geo = Path("data/01_raw/districts_official.geojson")

    # Create a dedicated folder for figures in a new 'reports' directory
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / "germany_static_features_maps.png"

    # --- 2. Load and Prepare Data ---
    logging.info("Loading and preparing data for mapping...")
    try:
        df_static = pd.read_csv(path_data)
    except FileNotFoundError:
        logging.error(f"FATAL: The static features file was not found at '{path_data}'.")
        logging.error("Please run the 'build_static_features.py' script first.")
        return

    gdf_districts = gpd.read_file(path_districts_geo)
    gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)

    # Ensure the key for merging is the same data type (string with leading zeros)
    df_static['district_no'] = df_static['district_no'].astype(str).str.zfill(5)
    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)

    # --- 3. Merge Geospatial and Static Data ---
    merged_gdf = gdf_districts.merge(df_static, on='district_no', how='left')

    # --- 4. Create the Maps ---
    logging.info("Generating choropleth maps...")

    # We'll create a 1x2 plot for our two main static features
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Baseline Geographic Conditions for Agriculture in Germany', fontsize=24, y=0.95)

    # Define properties for missing data (though we expect none)
    missing_kwds = {'color': 'lightgrey', 'label': 'No Data'}

    # -- Map 1: Average Elevation --
    ax1 = axes[0]
    merged_gdf.plot(column='avg_elevation', cmap='terrain', linewidth=0.5, ax=ax1,
                    edgecolor='0.7', legend=True, missing_kwds=missing_kwds,
                    legend_kwds={'label': "Average Elevation (meters)", 'orientation': "horizontal"})
    ax1.set_title('Average Elevation by District', fontsize=18)
    ax1.set_axis_off()

    # -- Map 2: Plant Available Water Capacity (PAWC) --
    ax2 = axes[1]
    merged_gdf.plot(column='avg_soil_pawc', cmap='YlGnBu', linewidth=0.5, ax=ax2,
                    edgecolor='0.7', legend=True, missing_kwds=missing_kwds,
                    legend_kwds={'label': "Estimated Plant Available Water Capacity (PAWC)",
                                 'orientation': "horizontal"})
    ax2.set_title('Soil Quality (Estimated PAWC) by District', fontsize=18)
    ax2.set_axis_off()

    # --- 5. Save the Figure ---
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_filepath, dpi=150)

    logging.info(f"--- SUCCESS: Maps saved to '{output_filepath}' ---")
    print(f"\nVisualization saved. Please open the file: {output_filepath}")


if __name__ == "__main__":
    visualize_static_maps()