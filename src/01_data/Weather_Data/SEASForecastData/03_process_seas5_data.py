import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
BASE_DIR = Path.cwd()
SEAS5_INPUT_DIR = BASE_DIR / "data/01_raw/SEAS5_monthly_germany"
GEOJSON_PATH = BASE_DIR / "data/01_raw/districts_official.geojson"
OUTPUT_CSV_PATH = BASE_DIR / "data/02_intermediate/seas5_forecast_features_1979_2021.csv"

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

def process_seas5_forecasts():
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    climatology = calculate_climatology(SEAS5_INPUT_DIR, CLIMATOLOGY_YEARS)
    if climatology is None: return

    districts_gdf = gpd.read_file(GEOJSON_PATH).to_crs("EPSG:4326")
    districts_gdf['centroid'] = districts_gdf.geometry.centroid
    processed_features = []

    for year in AVAILABLE_SEAS5_YEARS:
        logging.info(f"--- Processing SEAS5 data for year: {year} ---")
        file_path = SEAS5_INPUT_DIR / f"seas5_monthly_germany_{year}_march_start.nc"
        if not file_path.exists(): continue

        with xr.open_dataset(file_path) as ds:
            if 'tprate' in ds.data_vars: ds = ds.rename({'tprate': 'tp'})
            ensemble_mean = ds.mean(dim='number')
            anomaly = ensemble_mean - climatology

            spring_anomaly = anomaly.sel(forecastMonth=slice(2, 4)).mean(dim='forecastMonth')
            summer_anomaly = anomaly.sel(forecastMonth=slice(5, 7)).mean(dim='forecastMonth')

            for i, row in districts_gdf.iterrows():
                centroid = row['centroid']
                selection_spring = spring_anomaly.sel(longitude=centroid.x, latitude=centroid.y, method='nearest')
                selection_summer = summer_anomaly.sel(longitude=centroid.x, latitude=centroid.y, method='nearest')

                processed_features.append({
                    'year': year, 'district_no': row['id'],
                    'spring_temp_anomaly_forecast': selection_spring['t2m'].item(),
                    # --- FINAL FIX: Convert precip rate (kg/m²/s or mm/s) to mm/day. REMOVED extra * 1000 ---
                    'spring_precip_anomaly_forecast': selection_spring['tp'].item() * 86400,
                    'summer_temp_anomaly_forecast': selection_summer['t2m'].item(),
                    'summer_precip_anomaly_forecast': selection_summer['tp'].item() * 86400,
                })

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