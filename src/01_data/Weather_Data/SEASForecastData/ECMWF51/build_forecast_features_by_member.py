# File: src/features/build_forecast_features_by_member.py
# Description: Processes SEAS5 data, preserving the ensemble member dimension
#              and creating monthly anomaly features for each variable.
# VERSION: Monthly Granularity

import cdsapi
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import logging
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
BASE_DIR = Path.cwd()
RAW_DATA_DIR = BASE_DIR / "data/01_raw/ECMWF51_monthly_germany"
INTERMEDIATE_DATA_DIR = BASE_DIR / "data/02_intermediate"
GEOJSON_PATH = BASE_DIR / "data/01_raw/districts_official.geojson"
OUTPUT_CSV_PATH = INTERMEDIATE_DATA_DIR / "ecmwf51_forecast_features_BY_MEMBER.csv"

# Timeframe and Variable Configuration
HINDCAST_YEARS = list(range(1981, 2017))
FORECAST_YEARS = list(range(2017, 2025))
CLIMATOLOGY_YEARS = HINDCAST_YEARS
AVAILABLE_YEARS = HINDCAST_YEARS + FORECAST_YEARS
TARGET_DOWNLOAD_VARIABLES = ["2m_temperature", "evaporation", "runoff", "snowfall", "soil_temperature_level_1",
                             "surface_solar_radiation_downwards", "total_precipitation"]
MM_PER_DAY_FACTOR = 86400 * 1000
DOWNLOAD_TO_SHORT_NAME_MAP = {'2m_temperature': 't2m', 'total_precipitation': 'tprate',
                              'surface_solar_radiation_downwards': 'msdsrf', 'evaporation': 'erate', 'runoff': 'mrort',
                              'soil_temperature_level_1': 'stl1', 'snowfall': 'mtsfr'}
VAR_MAP = {'t2m': {'name': 'temp', 'unit_factor': 1.0}, 'tprate': {'name': 'precip', 'unit_factor': MM_PER_DAY_FACTOR},
           'msdsrf': {'name': 'solar_rad', 'unit_factor': 1.0},
           'erate': {'name': 'evaporation', 'unit_factor': MM_PER_DAY_FACTOR},
           'mrort': {'name': 'runoff', 'unit_factor': MM_PER_DAY_FACTOR},
           'stl1': {'name': 'soil_temp_l1', 'unit_factor': 1.0},
           'mtsfr': {'name': 'snowfall', 'unit_factor': MM_PER_DAY_FACTOR}}


def preprocess_ecmwf_dataset(ds):
    if 'forecastMonth' in ds.dims:
        ds = ds.rename({'forecastMonth': 'leadtime_month'})
    return ds.rename({k: v for k, v in DOWNLOAD_TO_SHORT_NAME_MAP.items() if k in ds})


def calculate_climatology(ecmwf_dir, climatology_years):
    logging.info(f"Calculating Climatology ({climatology_years[0]}-{climatology_years[-1]})")
    files_to_load = [f for f in ecmwf_dir.glob("*.nc") if int(f.stem.split('_')[-3]) in climatology_years]
    if not files_to_load:
        logging.error("CRITICAL: No NetCDF files found for climatology period.")
        return None
    with xr.open_mfdataset(files_to_load, combine='nested', concat_dim='forecast_reference_time',
                           preprocess=preprocess_ecmwf_dataset, join='override', engine='netcdf4') as ds:
        climatology = ds.mean(dim=['forecast_reference_time', 'number'])
        return climatology.load()


def process_forecasts_by_member():
    """
    Processes raw NetCDF files into a feature CSV with monthly granularity
    for each SEAS5 ensemble member.
    """
    logging.info("\n--- STAGE: Processing Raw Data into Monthly Member-Specific Features ---")
    INTERMEDIATE_DATA_DIR.mkdir(exist_ok=True)
    climatology = calculate_climatology(RAW_DATA_DIR, CLIMATOLOGY_YEARS)
    if climatology is None: return

    logging.info("Preparing District Centroids for lookup...")
    districts_gdf = gpd.read_file(GEOJSON_PATH).to_crs("EPSG:4326")
    district_centroids = districts_gdf.centroid
    target_lons = xr.DataArray(district_centroids.x, dims=['district'], coords={'district': districts_gdf['id']})
    target_lats = xr.DataArray(district_centroids.y, dims=['district'], coords={'district': districts_gdf['id']})

    all_member_features = []
    for year in tqdm(AVAILABLE_YEARS, desc="Processing Forecasts by Member"):
        file_path = RAW_DATA_DIR / f"ECMWF51_monthly_germany_{year}_march_start.nc"
        if not file_path.exists(): continue

        with xr.open_dataset(file_path) as ds_raw:
            ds = preprocess_ecmwf_dataset(ds_raw)
            anomaly = ds - climatology

            anomaly_nearest = anomaly.sel(longitude=target_lons, latitude=target_lats, method='nearest')
            year_df_long = anomaly_nearest.to_dataframe()

            variable_codes = list(VAR_MAP.keys())
            year_df_wide = year_df_long.pivot_table(
                index=['district', 'number'],
                columns='leadtime_month',
                values=variable_codes
            )

            year_df_wide.columns = [f"{var}_{int(month)}" for var, month in year_df_wide.columns]
            year_df_wide.reset_index(inplace=True)
            year_df_wide['year'] = year

            all_member_features.append(year_df_wide)

    logging.info("Concatenating all years and members...")
    final_df_raw = pd.concat(all_member_features, ignore_index=True)

    # --- CORRECTED BLOCK: Renaming and Formatting ---
    logging.info("Renaming features for final monthly output...")
    output_df = final_df_raw[['year', 'district', 'number']].copy()
    output_df.rename(columns={'district': 'district_no', 'number': 'member'}, inplace=True)

    for var_code, var_info in VAR_MAP.items():
        final_name = var_info['name']
        unit_factor = var_info['unit_factor']

        for lead_month in range(1, 8):  # leadtime_month runs from 1 to 7
            pivoted_col_name = f"{var_code}_{lead_month}"

            if pivoted_col_name in final_df_raw.columns:
                # The forecast starts in March. lead_month=1 is March (month 3).
                actual_month = lead_month + 2
                new_col_name = f"{final_name}_anomaly_forecast_{actual_month}"

                output_df[new_col_name] = final_df_raw[pivoted_col_name] * unit_factor

    final_df = output_df
    # --- END OF CORRECTED BLOCK ---

    final_df.sort_values(by=['year', 'district_no', 'member'], inplace=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')

    logging.info(f"--- Processing complete. Final member-specific dataset saved to {OUTPUT_CSV_PATH} ---")
    print("\n--- Validation of Final Output (Sample from 2021, different members) ---")
    print(final_df[final_df.year == 2021].head())
    print(f"\nTotal Rows Created: {len(final_df)}")
    print(f"Total Columns Created: {len(final_df.columns)}")


if __name__ == "__main__":
    # Note: This script assumes the download script has already been run.
    process_forecasts_by_member()