import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import logging
from rasterstats import zonal_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
BASE_DIR = Path.cwd()
SEAS5_INPUT_DIR = BASE_DIR / "data/01_raw/SEAS5_monthly_germany"
GEOJSON_PATH = BASE_DIR / "data/01_raw/districts_official.geojson"
OUTPUT_CSV_PATH = BASE_DIR / "data/02_intermediate/seas5_forecast_features_1979_2021.csv"
TEMP_RASTER_PATH = BASE_DIR / "data/02_intermediate/temp_raster_seas5_forecast.tif" # New temp path

CLIMATOLOGY_YEARS = list(range(1993, 2017))
AVAILABLE_SEAS5_YEARS = list(range(1981, 2023))
FINAL_DATASET_YEARS = list(range(1979, 2022))

def calculate_climatology(seas5_dir, climatology_years):
    logging.info(f"Calculating climatology from years {climatology_years[0]}-{climatology_years[-1]}...")
    files_to_load = [f for f in seas5_dir.glob("*.nc") if int(f.stem.split('_')[3]) in climatology_years]
    if not files_to_load: logging.error("FATAL: No climatology files found."); return None

    with xr.open_mfdataset(files_to_load, combine='nested', concat_dim='time', join='override') as ds:
        if 'tprate' in ds.data_vars: ds = ds.rename({'tprate': 'tp'})
        climatology = ds.mean(dim='number').mean(dim='time')
        if 'forecast_reference_time' in climatology.coords:
            climatology = climatology.drop_vars('forecast_reference_time')
        logging.info("Climatology calculation successful.")
        return climatology.load()


def get_stats_seas5(data_array):
    """Calculates zonal mean statistics for SEAS5 data."""
    # Set spatial dimensions and CRS before writing to raster
    data_array.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude").rio.write_crs("EPSG:4326", inplace=True)

    # Save a temporary raster
    data_array.rio.to_raster(TEMP_RASTER_PATH, compress='LZW')

    # Calculate zonal stats using the official districts GeoJSON
    stats = zonal_stats(str(GEOJSON_PATH), str(TEMP_RASTER_PATH), stats="mean", geojson_out=True)
    return [s['properties']['mean'] for s in stats]


def process_seas5_forecasts():
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    climatology = calculate_climatology(SEAS5_INPUT_DIR, CLIMATOLOGY_YEARS)
    if climatology is None: return

    # Removed the unnecessary .to_crs("EPSG:4326") and centroid calculation here
    districts_gdf = gpd.read_file(GEOJSON_PATH)
    processed_features = []

    try:  # Start the try block for file cleanup
        for year in AVAILABLE_SEAS5_YEARS:
            logging.info(f"--- Processing SEAS5 data for year: {year} ---")
            file_path = SEAS5_INPUT_DIR / f"seas5_monthly_germany_{year}_march_start.nc"
            if not file_path.exists(): continue

            with xr.open_dataset(file_path) as ds:
                if 'tprate' in ds.data_vars: ds = ds.rename({'tprate': 'tp'})
                ensemble_mean = ds.mean(dim='number')
                anomaly = ensemble_mean - climatology

                # Corrected Seasonal Definition (MAM and JJA)
                spring_anomaly = anomaly.sel(forecastMonth=slice(1, 3)).mean(dim='forecastMonth')
                summer_anomaly = anomaly.sel(forecastMonth=slice(4, 6)).mean(dim='forecastMonth')

                # Calculate Zonal Mean for all variables across all districts
                spring_temp_means = get_stats_seas5(spring_anomaly['t2m'])
                spring_precip_means = get_stats_seas5(spring_anomaly['tp'])
                summer_temp_means = get_stats_seas5(summer_anomaly['t2m'])
                summer_precip_means = get_stats_seas5(summer_anomaly['tp'])

                # --- CORRECTED LOOP: Iterate ONCE over the Zonal Mean results ---
                # The zonal_stats function returns results in the order of districts_gdf rows.
                for i, row in districts_gdf.iterrows():
                    # Safely handle None values from zonal_stats and assign them
                    s_t = spring_temp_means[i] if spring_temp_means[i] is not None else 0.0
                    s_p = spring_precip_means[i] if spring_precip_means[i] is not None else 0.0
                    su_t = summer_temp_means[i] if summer_temp_means[i] is not None else 0.0
                    su_p = summer_precip_means[i] if summer_precip_means[i] is not None else 0.0

                    processed_features.append({
                        'year': year, 'district_no': row['id'],
                        'spring_temp_anomaly_forecast': s_t,
                        # Corrected precip units (rate to mm/day)
                        'spring_precip_anomaly_forecast': s_p * 86400 * 1000,
                        'summer_temp_anomaly_forecast': su_t,
                        'summer_precip_anomaly_forecast': su_p * 86400 * 1000,
                    })

    finally:
        # Ensure temporary raster file is deleted
        if TEMP_RASTER_PATH.exists():
            logging.info(f"Cleaning up temporary file: {TEMP_RASTER_PATH}")
            TEMP_RASTER_PATH.unlink()

    # ... (placeholder_features and final_df saving logic is fine) ...
    placeholder_features = []
    placeholder_years = [y for y in FINAL_DATASET_YEARS if y not in AVAILABLE_SEAS5_YEARS]
    for district_id in districts_gdf['id']:
        for year in placeholder_years:
            placeholder_features.append({'year': year, 'district_no': district_id, 'spring_temp_anomaly_forecast': 0.0,
                                         'spring_precip_anomaly_forecast': 0.0, 'summer_temp_anomaly_forecast': 0.0,
                                         'summer_precip_anomaly_forecast': 0.0})

    final_df = pd.concat([pd.DataFrame(processed_features), pd.DataFrame(placeholder_features)], ignore_index=True)
    final_df = final_df[final_df['year'].isin(FINAL_DATASET_YEARS)].copy()
    final_df.sort_values(by=['year', 'district_no'], inplace=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    logging.info(f"SUCCESS: Complete dataset for 1979-2021 saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    process_seas5_forecasts()