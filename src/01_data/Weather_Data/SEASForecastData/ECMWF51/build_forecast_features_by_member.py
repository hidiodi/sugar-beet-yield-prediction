# File: src/features/build_forecast_features_by_member.py
# Description: Processes SEAS5 data, preserving the ensemble member dimension
#              to create a feature set representing multiple possible futures.

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
OUTPUT_CSV_PATH = INTERMEDIATE_DATA_DIR / "ecmwf51_forecast_features_BY_MEMBER.csv"  # New output file

# ... (All other configuration from the original script remains the same) ...
HINDCAST_YEARS = list(range(1981, 2017));
FORECAST_YEARS = list(range(2017, 2025))
CLIMATOLOGY_YEARS = HINDCAST_YEARS;
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
    # Climatology is still the long-term average across all members
    logging.info(f"Calculating Climatology ({climatology_years[0]}-{climatology_years[-1]})")
    files_to_load = [f for f in ecmwf_dir.glob("*.nc") if int(f.stem.split('_')[-3]) in climatology_years]
    if not files_to_load: return None
    with xr.open_mfdataset(files_to_load, combine='nested', concat_dim='forecast_reference_time',
                           preprocess=preprocess_ecmwf_dataset, join='override', engine='netcdf4') as ds:
        climatology = ds.mean(dim=['forecast_reference_time', 'number'])
        return climatology.load()


def process_forecasts_by_member():
    """
    Processes all raw NetCDF files into a feature CSV where each SEAS5
    ensemble member becomes a separate data point.
    """
    logging.info("\n--- STAGE: Processing Raw Data into Member-Specific Features ---")
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

            # THE KEY CHANGE: Calculate anomaly PER MEMBER, not on the mean
            anomaly = ds - climatology  # xarray broadcasts this correctly

            spring_anomaly = anomaly.sel(leadtime_month=slice(2, 4)).mean(dim='leadtime_month')
            summer_anomaly = anomaly.sel(leadtime_month=slice(5, 7)).mean(dim='leadtime_month')

            # Extract data at nearest points. This preserves the 'number' dimension.
            spring_nearest = spring_anomaly.sel(longitude=target_lons, latitude=target_lats, method='nearest')
            summer_nearest = summer_anomaly.sel(longitude=target_lons, latitude=target_lats, method='nearest')

            # Convert the multi-dimensional xarray object to a flat DataFrame
            spring_df = spring_nearest.to_dataframe().reset_index()
            summer_df = summer_nearest.to_dataframe().reset_index()

            # Merge the seasonal dataframes
            year_df = pd.merge(spring_df, summer_df, on=['district', 'number'], suffixes=('_spring', '_summer'))
            year_df['year'] = year

            all_member_features.append(year_df)

    logging.info("Concatenating all years and members...")
    final_df_raw = pd.concat(all_member_features, ignore_index=True)

    # --- Final Feature Renaming and Formatting ---
    logging.info("Pivoting and renaming features for final output...")
    final_df = final_df_raw[['year', 'district', 'number']].copy()
    final_df.rename(columns={'district': 'district_no', 'number': 'seas5_member'}, inplace=True)

    for var_code, var_info in VAR_MAP.items():
        var_name = var_info['name']
        unit_factor = var_info['unit_factor']

        # Spring
        if f'{var_code}_spring' in final_df_raw.columns:
            final_df[f'spring_{var_name}_anomaly_forecast'] = final_df_raw[f'{var_code}_spring'] * unit_factor
        # Summer
        if f'{var_code}_summer' in final_df_raw.columns:
            final_df[f'summer_{var_name}_anomaly_forecast'] = final_df_raw[f'{var_code}_summer'] * unit_factor

    # Note: Probability features are dropped for simplicity in this design,
    # as the variation in anomalies across members is a much stronger signal of uncertainty.

    final_df.sort_values(by=['year', 'district_no', 'seas5_member'], inplace=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')

    logging.info(f"--- Processing complete. Final member-specific dataset saved to {OUTPUT_CSV_PATH} ---")
    print("\n--- Validation of Final Output (Sample from 2021, different members) ---")
    print(final_df[final_df.year == 2021].head())
    print(f"\nTotal Rows Created: {len(final_df)}")
    print(f"Total Columns Created: {len(final_df.columns)}")


if __name__ == "__main__":
    # Assuming download_ecmwf_data() has been run successfully before
    process_forecasts_by_member()