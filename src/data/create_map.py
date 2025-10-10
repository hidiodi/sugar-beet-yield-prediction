# src/visualization/visualize_factory_map.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def visualize_factory_locations():
    """
    Creates and saves a static map of Germany showing district boundaries
    with the locations of factories plotted on top.
    """
    logging.info("--- Starting Factory Location Visualization ---")

    # --- 1. Define File Paths ---
    path_factories = Path("data/01_raw/HarvestData/factory_mapping.csv")
    path_districts_geo = Path("data/01_raw/districts_official.geojson")

    # Define the output directory and ensure it exists
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / "german_factories_map.png"

    # --- 2. Load and Prepare Data ---
    logging.info("Loading and preparing geospatial and factory data...")

    # Load the district boundaries
    try:
        gdf_districts = gpd.read_file(path_districts_geo)
    except Exception as e:
        logging.error(f"FATAL: Could not read the geojson file at '{path_districts_geo}'. Error: {e}")
        return

    # Load the factory data
    try:
        df_factories = pd.read_csv(path_factories)
    except FileNotFoundError:
        logging.error(f"FATAL: The factory mapping file was not found at '{path_factories}'.")
        return

    # --- 3. Convert Factory DataFrame to a GeoDataFrame ---
    # Create Point geometries from the latitude and longitude columns.
    # We set the CRS to EPSG:4326, which is the standard for lat/lon coordinates.
    gdf_factories = gpd.GeoDataFrame(
        df_factories,
        geometry=gpd.points_from_xy(df_factories.longitude, df_factories.latitude),
        crs="EPSG:4326"
    )
    logging.info(f"Successfully converted {len(gdf_factories)} factory locations to geospatial points.")

    # --- 4. Create the Map ---
    logging.info("Generating the map plot...")

    # Initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))

    # Plot the base map of German districts
    gdf_districts.plot(ax=ax,
                       color='#EEEEEE',       # A light grey for the fill
                       edgecolor='black',    # Black boundaries for clarity
                       linewidth=0.4)

    # Overlay the factory locations on the same axes (ax)
    gdf_factories.plot(ax=ax,
                       marker='o',           # Circle marker
                       color='red',
                       markersize=50,
                       alpha=0.8,
                       label='Factories')

    # --- 5. Style and Save the Figure ---
    ax.set_title('Factory Locations in Germany', fontsize=20, pad=20)
    ax.set_axis_off()

    # Adjust layout to prevent the title from being cut off
    plt.tight_layout()

    # Save the final plot to the specified file
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight')

    logging.info(f"--- SUCCESS: Map saved to '{output_filepath}' ---")
    print(f"\nVisualization saved. Please open the file: {output_filepath}")


if __name__ == "__main__":
    visualize_factory_locations()