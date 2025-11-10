# File: src/features/build_forecast_features_by_member.py
# Description: Processes SEAS5 data, preserving the ensemble member dimension
#              and creating both monthly and seasonal anomaly features.
# VERSION: 2.0 (Monthly + Seasonal Granularity)

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
MM_PER_DAY_FACTOR = 86400 * 1000 # Converts m/s of precip/evap to mm/day
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
    """Renames dimensions and variables for consistency."""
    if 'forecastMonth' in ds.dims:
        ds = ds.rename({'forecastMonth': 'leadtime_month'})
    return ds.rename({k: v for k, v in DOWNLOAD_TO_SHORT_NAME_MAP.items() if k in ds})


def calculate_climatology(ecmwf_dir, climatology_years):
    """Calculates the long-term mean from hindcast files."""
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
    Processes raw NetCDF files into a feature CSV with both monthly and seasonal
    granularity for each SEAS5 ensemble member.
    """
    logging.info("\n--- STAGE: Processing Raw Data into Monthly & Seasonal Member-Specific Features ---")
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

    if not all_member_features:
        logging.error("No data was processed. Aborting.")
        return

    logging.info("Concatenating all years and members...")
    final_df_raw = pd.concat(all_member_features, ignore_index=True)

    # --- Feature Engineering Block: Create Monthly and Seasonal Features ---
    logging.info("Creating monthly anomaly features...")
    final_df = final_df_raw[['year', 'district', 'number']].copy()
    final_df.rename(columns={'district': 'district_no', 'number': 'member'}, inplace=True)

    # Create cleaned-up monthly features first
    for var_code, var_info in VAR_MAP.items():
        final_name = var_info['name']
        unit_factor = var_info['unit_factor']

        # leadtime_month runs from 1 to 7 (for a 7-month forecast)
        # March start: lead_month=1 -> March, lead_month=2 -> April, etc.
        for lead_month in range(1, 8):
            pivoted_col_name = f"{var_code}_{lead_month}"
            if pivoted_col_name in final_df_raw.columns:
                actual_month = lead_month + 2
                new_col_name = f"{final_name}_anomaly_forecast_{actual_month}"
                final_df[new_col_name] = final_df_raw[pivoted_col_name] * unit_factor

    logging.info("Calculating seasonal aggregations from monthly features...")
    for var_code, var_info in VAR_MAP.items():
        final_name = var_info['name']

        # Define Spring columns (Mar, Apr, May)
        spring_cols = [f"{final_name}_anomaly_forecast_{m}" for m in [3, 4, 5]]
        spring_cols_exist = [col for col in spring_cols if col in final_df.columns]
        if spring_cols_exist:
            final_df[f"spring_{final_name}_anomaly_forecast"] = final_df[spring_cols_exist].mean(axis=1)

        # Define Summer columns (Jun, Jul, Aug)
        summer_cols = [f"{final_name}_anomaly_forecast_{m}" for m in [6, 7, 8]]
        summer_cols_exist = [col for col in summer_cols if col in final_df.columns]
        if summer_cols_exist:
            final_df[f"summer_{final_name}_anomaly_forecast"] = final_df[summer_cols_exist].mean(axis=1)
    # --- End of Feature Engineering Block ---

    final_df.sort_values(by=['year', 'district_no', 'member'], inplace=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')

    logging.info(f"--- Processing complete. Final member-specific dataset saved to {OUTPUT_CSV_PATH} ---")
    print("\n--- Validation of Final Output (Sample from 2021, one member) ---")
    validation_sample = final_df[(final_df.year == 2021) & (final_df.district_no == '01053')].head(1).T
    print(validation_sample)
    print(f"\nTotal Rows Created: {len(final_df)}")
    print(f"Total Columns Created: {len(final_df.columns)}")
    print(f"Example columns: {final_df.columns.tolist()[:5]}...{final_df.columns.tolist()[-5:]}")


if __name__ == "__main__":
    # This script assumes the raw NetCDF data has already been downloaded.
    process_forecasts_by_member()