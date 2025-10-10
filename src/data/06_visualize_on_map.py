# src/data/06_visualize_on_map.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_choropleth_maps():
    """
    Creates a set of choropleth maps to visualize the geographic distribution of
    crop yields and key weather features across Germany's districts.
    """
    logging.info("--- Starting Geographic Visualization ---")

    # --- 1. Define File Paths ---
    path_data = Path("data/03_processed/final_dataset_with_advanced_features.csv")
    path_districts_geo = Path("data/01_raw/districts_official.geojson")
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / "germany_yield_and_weather_maps.png"

    # --- 2. Load and Prepare Data ---
    logging.info("Loading and preparing data for mapping...")
    df = pd.read_csv(path_data)
    gdf_districts = gpd.read_file(path_districts_geo)
    gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)

    # Ensure the key for merging is the same data type
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)

    # To map the data, we need one value per district. Let's calculate the
    # average of our key variables across all years (2017-2024).
    df_aggregated = df.groupby('district_no').mean(numeric_only=True).reset_index()

    # --- 3. Merge Geospatial and Aggregated Data ---
    # We perform a 'left' merge to keep ALL districts from the map,
    # even if some don't have yield data.
    merged_gdf = gdf_districts.merge(df_aggregated, on='district_no', how='left')

    # --- 4. Create the Maps ---
    logging.info("Generating choropleth maps...")

    # We'll create a 1x3 plot: Avg Yield, Avg Heat Stress, Avg Peak Precipitation
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))
    fig.suptitle('Geographic Analysis of Sugar Beet Yield and Weather (2017-2024 Averages)', fontsize=24, y=0.95)

    # Define properties for missing data (districts with no yield info)
    missing_kwds = {'color': 'lightgrey', 'label': 'No Data'}

    # -- Map 1: Average Yield --
    ax1 = axes[0]
    merged_gdf.plot(column='yield', cmap='YlGn', linewidth=0.5, ax=ax1,
                    edgecolor='0.7', legend=True, missing_kwds=missing_kwds,
                    legend_kwds={'label': "Average Yield (tons/hectare)", 'orientation': "horizontal"})
    ax1.set_title('Average Sugar Beet Yield', fontsize=18)
    ax1.set_axis_off()

    # -- Map 2: Heat Stress Days --
    ax2 = axes[1]
    merged_gdf.plot(column='heat_stress_days_peak_growth', cmap='Reds', linewidth=0.5, ax=ax2,
                    edgecolor='0.7', legend=True, missing_kwds=missing_kwds,
                    legend_kwds={'label': "Average Number of Days > 30°C (Jul-Sep)", 'orientation': "horizontal"})
    ax2.set_title('Heat Stress During Peak Growth', fontsize=18)
    ax2.set_axis_off()

    # -- Map 3: Peak Growth Precipitation --
    ax3 = axes[2]
    merged_gdf.plot(column='precip_total_peak_growth', cmap='Blues', linewidth=0.5, ax=ax3,
                    edgecolor='0.7', legend=True, missing_kwds=missing_kwds,
                    legend_kwds={'label': "Total Precipitation (kg m⁻²) during Jul-Sep", 'orientation': "horizontal"})
    ax3.set_title('Precipitation During Peak Growth', fontsize=18)
    ax3.set_axis_off()

    # --- 5. Save the Figure ---
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_filepath, dpi=150)

    logging.info(f"--- SUCCESS: Maps saved to '{output_filepath}' ---")


if __name__ == "__main__":
    create_choropleth_maps()