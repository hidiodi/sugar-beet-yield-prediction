# inspect_dlr_data.py (Smarter Version)

import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# --- Configuration ---
# Define the paths to the files we will use for a guaranteed-to-work inspection.
DLR_FILE_PATH = Path("data/01_raw/dlr_croptypes/croptypes_2017.tif")
DISTRICTS_GEOJSON_PATH = Path("data/01_raw/districts_official.geojson")
SUGAR_BEET_CODE = 60
DISTRICT_TO_INSPECT = "Biberach"

def inspect_dlr_data_smart():
    """
    Loads a DLR crop type GeoTIFF, reports its characteristics, and automatically
    creates a valid sample plot by using a district polygon as a guide.
    """
    print(f"--- Smart Inspection of DLR Crop Type Data: {DLR_FILE_PATH.name} ---")

    # --- 1. Load the Raster Data ---
    try:
        rds = rioxarray.open_rasterio(DLR_FILE_PATH)
    except Exception as e:
        print(f"ERROR: Could not open the DLR GeoTIFF. Ensure the path is correct: {DLR_FILE_PATH}")
        print(f"Error details: {e}")
        return

    # --- 2. Report Key Raster Characteristics ---
    print("\n--- 1. Key Data Characteristics ---")
    crs = rds.rio.crs
    resolution = rds.rio.resolution()
    pixel_area_m2 = abs(resolution[0] * resolution[1])
    print(f"Coordinate Reference System (CRS): {crs}")
    print(f"Resolution (x, y): {resolution} meters")
    print(f"  -> Area of one pixel: {pixel_area_m2:.1f} m²")
    print(f"Data Type: {rds.dtype}")
    print(f"No-Data Value: {rds.rio.nodata}")

    # --- 3. Automatic Sampling and Plotting (No Guessing) ---
    print("\n--- 2. Automatic Sampling and Visual Inspection ---")
    try:
        # Load the district shapes
        gdf_districts = gpd.read_file(DISTRICTS_GEOJSON_PATH)

        # Select a sample district (let's use the 10th one for variety)
        #sample_district = gdf_districts.iloc[[10]]
        sample_district = gdf_districts[gdf_districts['name'].str.contains(DISTRICT_TO_INSPECT, case=False, na=False)]
        district_name = sample_district.name.iloc[0]
        print(f"Automatically selecting a sample district: '{district_name}'")

        # --- THIS IS THE CRITICAL STEP ---
        # Reproject the district's geometry to match the CRS of the raster file.
        print(f"Reprojecting district from '{sample_district.crs}' to raster CRS '{crs}'...")
        reprojected_district = sample_district.to_crs(crs)

        # Get the bounding box of the reprojected district
        minx, miny, maxx, maxy = reprojected_district.total_bounds

        # Clip the raster to this exact bounding box
        clipped_rds = rds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, auto_expand=True)

        # --- 4. Final Analysis of the Sample ---
        # Count how many sugar beet pixels are in our sample area
        sugar_beet_pixels = np.count_nonzero(clipped_rds.values == SUGAR_BEET_CODE)
        print(
            f"\nFound {sugar_beet_pixels} sugar beet pixels (code {SUGAR_BEET_CODE}) in the '{district_name}' sample area.")

        # Create the plot
        plt.figure(figsize=(12, 10))
        clipped_rds.plot(cmap='tab20', add_colorbar=False)
        plt.title(f"DLR Crop Map for sample district: '{district_name}'\n(CRS: {crs})", fontsize=16)
        plt.xlabel("Easting (meters)")
        plt.ylabel("Northing (meters)")

        plot_filename = "dlr_data_sample_auto.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\nSUCCESS: Automatically generated sample map saved to '{plot_filename}'.")
        print("This plot should clearly show field patterns for the selected district.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the automatic plotting process.")
        print(f"Error details: {e}")


if __name__ == "__main__":
    inspect_dlr_data_smart()