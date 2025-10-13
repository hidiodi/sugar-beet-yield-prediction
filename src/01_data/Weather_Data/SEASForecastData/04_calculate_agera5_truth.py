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
# --- NEW, DEDICATED OUTPUT FILE ---
GROUND_TRUTH_CSV_PATH = BASE_DIR / "data/02_intermediate/agera5_ground_truth_1979_2021.csv"
TEMP_RASTER_PATH = BASE_DIR / "data/02_intermediate/temp_raster_agera5_truth.tif"

# --- Years to process for the complete ground truth ---
ALL_YEARS = list(range(1979, 2022))
CLIMATOLOGY_YEARS = list(range(1993, 2017))


# --- 2. YOUR WORKING CLIMATOLOGY FUNCTION (UNCHANGED) ---
def calculate_agera5_climatology_with_real_progress(ds, years):
    logging.info(f"Calculating AGERA5 climatology from years {years[0]}-{years[-1]}...")
    yearly_monthly_means = []
    for year in tqdm(years, desc="Calculating Climatology"):
        year_ds = ds.sel(time=ds.time.dt.year == year).load()
        monthly_mean = year_ds.groupby('time.month').mean(dim='time')
        yearly_monthly_means.append(monthly_mean)
    climatology_combined = xr.concat(yearly_monthly_means, dim='year')
    final_climatology = climatology_combined.mean(dim='year')
    logging.info("AGERA5 climatology calculation successful.")
    return final_climatology


# --- 3. MAIN WORKFLOW TO GENERATE COMPLETE GROUND TRUTH ---
def create_full_ground_truth():
    logging.info(f"Generating full AGERA5 ground truth for years {ALL_YEARS[0]}-{ALL_YEARS[-1]}...")
    agera5_ds = xr.open_dataset(AGERA5_INPUT_PATH)
    districts_gdf = gpd.read_file(GEOJSON_PATH)
    monthly_climatology = calculate_agera5_climatology_with_real_progress(agera5_ds, CLIMATOLOGY_YEARS)

    all_features = []

    try:
        # Loop over ALL available years, not just the missing ones
        for year in tqdm(ALL_YEARS, desc="Processing All Years"):
            yearly_ds = agera5_ds.sel(time=agera5_ds.time.dt.year == year).load()
            yearly_monthly_mean = yearly_ds.groupby('time.month').mean(dim='time')
            anomaly_monthly = yearly_monthly_mean - monthly_climatology

            spring_anomaly = anomaly_monthly.sel(month=[3, 4, 5]).mean(dim='month')
            summer_anomaly = anomaly_monthly.sel(month=[6, 7, 8]).mean(dim='month')

            def get_stats(data_array):
                data_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326", inplace=True)
                data_array.rio.to_raster(TEMP_RASTER_PATH, compress='LZW')
                stats = zonal_stats(str(GEOJSON_PATH), str(TEMP_RASTER_PATH), stats="mean", geojson_out=True)
                return [s['properties']['mean'] for s in stats]

            spring_temp_means = get_stats(spring_anomaly['Temperature_Air_2m_Mean_24h'])
            spring_precip_means = get_stats(spring_anomaly['Precipitation_Flux'])
            summer_temp_means = get_stats(summer_anomaly['Temperature_Air_2m_Mean_24h'])
            summer_precip_means = get_stats(summer_anomaly['Precipitation_Flux'])

            for i, row in districts_gdf.iterrows():
                all_features.append({
                    'year': year, 'district_no': row['id'],
                    # Use clear column names for "actual" ground truth values
                    'spring_temp_anomaly_actual': spring_temp_means[i] if spring_temp_means[i] is not None else 0.0,
                    'spring_precip_anomaly_actual': spring_precip_means[i] * 86400 * 1000 if spring_precip_means[i] is not None else 0.0,
                    'summer_temp_anomaly_actual': summer_temp_means[i] if summer_temp_means[i] is not None else 0.0,
                    'summer_precip_anomaly_actual': summer_precip_means[i] * 86400 * 1000 if summer_precip_means[i] is not None else 0.0,
                })
    finally:
        if TEMP_RASTER_PATH.exists(): TEMP_RASTER_PATH.unlink()

    final_df = pd.DataFrame(all_features)
    final_df.sort_values(by=['year', 'district_no'], inplace=True)
    final_df.to_csv(GROUND_TRUTH_CSV_PATH, index=False)
    logging.info(f"SUCCESS: Complete ground truth dataset saved to {GROUND_TRUTH_CSV_PATH}")

if __name__ == "__main__":
    create_full_ground_truth()