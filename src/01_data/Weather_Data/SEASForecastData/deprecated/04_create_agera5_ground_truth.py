# File: create_agera5_ground_truth.py
# Description: Creates the high-resolution "ground truth" weather dataset from AgERA5
#              with progress bars for all long-running operations.

import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import logging
import rioxarray
from rasterstats import zonal_stats
from tqdm import tqdm

# --- Setup detailed logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. CONFIGURATION ---
BASE_DIR = Path.cwd()
AGERA5_INPUT_PATH = BASE_DIR / "data/02_intermediate/agera5_germany_merged.nc"
GEOJSON_PATH = BASE_DIR / "data/01_raw/districts_official.geojson"
GROUND_TRUTH_CSV_PATH = BASE_DIR / "data/02_intermediate/agera5_ground_truth_FINAL.csv"
TEMP_RASTER_PATH = BASE_DIR / "data/02_intermediate/temp_raster_agera5_truth.tif"

# --- Time and Season Configuration ---
ALL_YEARS = list(range(1979, 2022))
CLIMATOLOGY_YEARS = list(range(1993, 2017))

# --- CRITICAL: Seasons defined to perfectly match the forecast data ---
SEASON_MAP = {
    'spring': [4, 5, 6],  # April, May, June
    'summer': [7, 8, 9]  # July, August, September
}

VARIABLE_MAP = {
    'temp': 'Temperature_Air_2m_Mean_24h',
    'precip': 'Precipitation_Flux'
}


# --- 2. CLIMATOLOGY FUNCTION WITH PROGRESS BAR ---

# ============================ THE FIX: RESTORED PROGRESS BAR FOR CLIMATOLOGY ============================
# This is the function you originally used, which correctly shows progress for the slow climatology step.
def calculate_agera5_climatology_with_progress(ds, years):
    logging.info(f"Calculating AGERA5 climatology from years {years[0]}-{years[-1]}...")
    yearly_monthly_means = []
    # This loop will now be visible via the progress bar
    for year in tqdm(years, desc="Calculating Climatology"):
        year_ds = ds.sel(time=ds.time.dt.year == year).load()
        monthly_mean = year_ds.groupby('time.month').mean(dim='time')
        yearly_monthly_means.append(monthly_mean)

    # Combine the results
    climatology_combined = xr.concat(yearly_monthly_means, dim='year')
    final_climatology = climatology_combined.mean(dim='year')
    logging.info("AGERA5 climatology calculation successful.")
    return final_climatology


# =======================================================================================================


# --- 3. MAIN WORKFLOW ---
def create_full_ground_truth():
    logging.info(f"Generating full AGERA5 ground truth for years {ALL_YEARS[0]}-{ALL_YEARS[-1]}...")
    agera5_ds = xr.open_dataset(AGERA5_INPUT_PATH)
    districts_gdf = gpd.read_file(GEOJSON_PATH)

    # Call the function that has the progress bar
    monthly_climatology = calculate_agera5_climatology_with_progress(agera5_ds, CLIMATOLOGY_YEARS)

    all_features = []

    try:
        # This main loop also has its own progress bar
        for year in tqdm(ALL_YEARS, desc="Processing Ground Truth Years"):
            yearly_ds = agera5_ds.sel(time=agera5_ds.time.dt.year == year).load()
            yearly_monthly_mean = yearly_ds.groupby('time.month').mean(dim='time')
            anomaly_monthly = yearly_monthly_mean - monthly_climatology

            # Helper function for zonal stats
            def get_stats(data_array):
                data_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326", inplace=True)
                data_array.rio.to_raster(TEMP_RASTER_PATH, compress='LZW')
                stats = zonal_stats(str(GEOJSON_PATH), str(TEMP_RASTER_PATH), stats="mean", geojson_out=True)
                return [s['properties']['mean'] for s in stats]

            spring_anomaly = anomaly_monthly.sel(month=SEASON_MAP['spring']).mean(dim='month')
            summer_anomaly = anomaly_monthly.sel(month=SEASON_MAP['summer']).mean(dim='month')

            spring_temp_means = get_stats(spring_anomaly[VARIABLE_MAP['temp']])
            spring_precip_means = get_stats(spring_anomaly[VARIABLE_MAP['precip']])
            summer_temp_means = get_stats(summer_anomaly[VARIABLE_MAP['temp']])
            summer_precip_means = get_stats(summer_anomaly[VARIABLE_MAP['precip']])

            for i, row in districts_gdf.iterrows():
                all_features.append({
                    'year': year, 'district_no': row['id'],
                    'spring_temp_anomaly_actual': spring_temp_means[i] if spring_temp_means[i] is not None else 0.0,
                    'spring_precip_anomaly_actual': spring_precip_means[i] if spring_precip_means[
                                                                                                 i] is not None else 0.0,
                    'summer_temp_anomaly_actual': summer_temp_means[i] if summer_temp_means[i] is not None else 0.0,
                    'summer_precip_anomaly_actual': summer_precip_means[i] if summer_precip_means[
                                                                                                 i] is not None else 0.0,
                })
    finally:
        if TEMP_RASTER_PATH.exists():
            TEMP_RASTER_PATH.unlink()

    final_df = pd.DataFrame(all_features)
    final_df.sort_values(by=['year', 'district_no'], inplace=True)
    final_df.to_csv(GROUND_TRUTH_CSV_PATH, index=False, float_format='%.6f')
    logging.info(f"SUCCESS: Complete ground truth dataset saved to {GROUND_TRUTH_CSV_PATH}")


if __name__ == "__main__":
    create_full_ground_truth()