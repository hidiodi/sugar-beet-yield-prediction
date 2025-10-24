# File: build_forecast_features.py
# Description: A unified pipeline to download raw ECMWF SEAS51 forecast data
#              and process it into a clean, model-ready CSV file using a
#              fast centroid-lookup method.

import cdsapi
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

# --- 1. UNIFIED CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')

# --- Core Paths and Settings ---
BASE_DIR = Path.cwd()
RAW_DATA_DIR = BASE_DIR / "data/01_raw/ECMWF51_monthly_germany"
INTERMEDIATE_DATA_DIR = BASE_DIR / "data/02_intermediate"
GEOJSON_PATH = BASE_DIR / "data/01_raw/districts_official.geojson"
OUTPUT_CSV_PATH = INTERMEDIATE_DATA_DIR / "ecmwf51_forecast_features_FINAL.csv"

# --- Timeframe Configuration ---
HINDCAST_YEARS = list(range(1981, 2017))
FORECAST_YEARS = list(range(2017, 2025))
CLIMATOLOGY_YEARS = HINDCAST_YEARS
AVAILABLE_YEARS = HINDCAST_YEARS + FORECAST_YEARS

# --- Variable and Conversion Configuration ---
TARGET_DOWNLOAD_VARIABLES = [
    "2m_temperature", "evaporation", "runoff", "snowfall",
    "soil_temperature_level_1", "surface_solar_radiation_downwards", "total_precipitation",
    "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_dewpoint_temperature"
]

# This factor converts a rate in meters/second to a total in mm/day
MM_PER_DAY_FACTOR = 86400 * 1000

DOWNLOAD_TO_SHORT_NAME_MAP = {
    '2m_temperature': 't2m',
    'total_precipitation': 'tprate',
    'surface_solar_radiation_downwards': 'msdsrf',
    'evaporation': 'erate',
    'runoff': 'mrort',
    'soil_temperature_level_1': 'stl1',
    'snowfall': 'mtsfr',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    '2m_dewpoint_temperature': 'd2m',
}

VAR_MAP = {
    # Short Name: {Final Name, Unit Factor}
    't2m':    {'name': 'temp',        'unit_factor': 1.0},
    'tprate': {'name': 'precip',      'unit_factor': MM_PER_DAY_FACTOR},
    'msdsrf': {'name': 'solar_rad',   'unit_factor': 1.0},
    'erate':  {'name': 'evaporation', 'unit_factor': MM_PER_DAY_FACTOR},
    'mrort':  {'name': 'runoff',      'unit_factor': MM_PER_DAY_FACTOR},
    'stl1':   {'name': 'soil_temp_l1','unit_factor': 1.0},
    'mtsfr':  {'name': 'snowfall',    'unit_factor': MM_PER_DAY_FACTOR},
    'u10': {'name': 'u10_wind', 'unit_factor': 1.0},
    'v10': {'name': 'v10_wind', 'unit_factor': 1.0},
    'd2m': {'name': 'dewpoint', 'unit_factor': 1.0},
}


# --- 2. DOWNLOAD MODULE ---

def download_ecmwf_data():
    """
    Downloads the SEAS51 monthly hindcast/forecasts for each year.
    Skips any files that have already been downloaded.
    """
    logging.info("--- STAGE 1: Starting ECMWF51 Data Download ---")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        c = cdsapi.Client()
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize CDS API client. Check your .cdsapirc file: {e}")
        return False # Indicate failure

    for year in tqdm(AVAILABLE_YEARS, desc="Downloading Raw Forecast Data"):
        output_filename = RAW_DATA_DIR / f"ECMWF51_monthly_germany_{year}_march_start.nc"

        if output_filename.exists():
            logging.debug(f"SKIP: File '{output_filename.name}' already exists.")
            continue

        request = {
            'originating_centre': 'ecmwf', 'system': '51',
            'variable': TARGET_DOWNLOAD_VARIABLES,
            'product_type': 'monthly_mean',
            'year': str(year), 'month': '03',
            'leadtime_month': ['1', '2', '3', '4', '5', '6', '7'],
            'area': [55.5, 5.5, 47.0, 15.5], # Bounding box for Germany
            'format': 'netcdf',
        }

        period_type = 'hindcast' if year in HINDCAST_YEARS else 'forecast'
        if period_type == 'hindcast':
            logging.info(f"REQUESTING: MONTHLY HINDCAST for {year} (25 members)...")
            request['ensemble_member'] = [str(i) for i in range(1, 26)]
        else:
            logging.info(f"REQUESTING: MONTHLY FORECAST for {year} (51 members)...")

        try:
            c.retrieve('seasonal-monthly-single-levels', request, str(output_filename))
            logging.info(f" -> SUCCESS: Saved to '{output_filename.name}'")
        except Exception as e:
            logging.error(f" -> FAILED to download data for {year}. Error: {e}")
            continue

    logging.info("--- Download stage complete. ---")
    return True # Indicate success


# --- 3. PROCESSING MODULE ---

# ============================ FIX 2: Made Preprocessing More Robust ============================
def preprocess_ecmwf_dataset(ds):
    """
    Renames variables and the lead time dimension to consistent, standardized names.
    Checks ds.dims for reliability.
    """
    # Rename the lead time dimension first if it exists as 'forecastMonth'
    if 'forecastMonth' in ds.dims:
        ds = ds.rename({'forecastMonth': 'leadtime_month'})

    # Then, rename the data variables
    return ds.rename({k: v for k, v in DOWNLOAD_TO_SHORT_NAME_MAP.items() if k in ds})
# ===============================================================================================

def calculate_climatology(ecmwf_dir, climatology_years):
    """Calculates the long-term mean for each variable and lead month."""
    logging.info(f"Calculating Climatology ({climatology_years[0]}-{climatology_years[-1]})")
    files_to_load = [f for f in ecmwf_dir.glob("*.nc") if int(f.stem.split('_')[-3]) in climatology_years]
    if not files_to_load:
        logging.error("CRITICAL: No NetCDF files found for climatology period.")
        return None
    with xr.open_mfdataset(files_to_load, combine='nested', concat_dim='forecast_reference_time',
                           preprocess=preprocess_ecmwf_dataset, join='override', engine='netcdf4') as ds:
        climatology = ds.mean(dim=['forecast_reference_time', 'number'])
        return climatology.load()

def process_forecasts_to_csv():
    """
    Processes all raw NetCDF files into a single feature CSV using a fast
    centroid-lookup method.
    """
    logging.info("\n--- STAGE 2: Processing Raw Data into Final CSV ---")
    INTERMEDIATE_DATA_DIR.mkdir(exist_ok=True)

    climatology = calculate_climatology(RAW_DATA_DIR, CLIMATOLOGY_YEARS)
    if climatology is None: return

    logging.info("Preparing District Centroids for lookup...")
    districts_gdf = gpd.read_file(GEOJSON_PATH).to_crs("EPSG:4326")
    district_centroids = districts_gdf.centroid
    target_lons = xr.DataArray(district_centroids.x, dims=['district'], coords={'district': districts_gdf['id']})
    target_lats = xr.DataArray(district_centroids.y, dims=['district'], coords={'district': districts_gdf['id']})

    processed_features = []
    for year in tqdm(AVAILABLE_YEARS, desc="Processing Forecasts to Features"):
        file_path = RAW_DATA_DIR / f"ECMWF51_monthly_germany_{year}_march_start.nc"
        if not file_path.exists():
            logging.warning(f"File not found for year {year}. Skipping processing.")
            continue

        with xr.open_dataset(file_path) as ds_raw:
            ds = preprocess_ecmwf_dataset(ds_raw)
            aligned_ds, aligned_climatology = xr.align(ds, climatology, join='left')
            anomaly = aligned_ds.mean(dim='number') - aligned_climatology
            probability = (aligned_ds > aligned_climatology).mean(dim='number')

            # Aggregate into seasons (April-June, July-September)
            spring_anomaly = anomaly.sel(leadtime_month=slice(2, 4)).mean(dim='leadtime_month')
            summer_anomaly = anomaly.sel(leadtime_month=slice(5, 7)).mean(dim='leadtime_month')
            spring_prob = probability.sel(leadtime_month=slice(2, 4)).mean(dim='leadtime_month')
            summer_prob = probability.sel(leadtime_month=slice(5, 7)).mean(dim='leadtime_month')

            # Extract data for all districts using the fast 'nearest' method
            spring_anomaly_nearest = spring_anomaly.sel(longitude=target_lons, latitude=target_lats, method='nearest')
            summer_anomaly_nearest = summer_anomaly.sel(longitude=target_lons, latitude=target_lats, method='nearest')
            spring_prob_nearest = spring_prob.sel(longitude=target_lons, latitude=target_lats, method='nearest')
            summer_prob_nearest = summer_prob.sel(longitude=target_lons, latitude=target_lats, method='nearest')

            # Dynamically build the DataFrame for the current year
            year_results = {'year': year}
            for ds_agg, season_prefix in [
                (spring_anomaly_nearest, 'spring'), (summer_anomaly_nearest, 'summer'),
                (spring_prob_nearest, 'spring_prob'), (summer_prob_nearest, 'summer_prob')
            ]:
                df_season = ds_agg.to_dataframe().reset_index()
                if 'district_no' not in year_results:
                    year_results['district_no'] = df_season['district']

                for var_code, var_info in VAR_MAP.items():
                    if var_code in df_season:
                        final_var_name = var_info['name']
                        if 'prob' in season_prefix:
                            suffix = '_prob_warm_forecast' if 'temp' in final_var_name else '_prob_wet_forecast'
                            col_name = f"{season_prefix.replace('_prob', '')}_{final_var_name}{suffix}"
                            year_results[col_name] = df_season[var_code].values
                        else:  # Anomaly
                            col_name = f"{season_prefix}_{final_var_name}_anomaly_forecast"
                            year_results[col_name] = df_season[var_code].values * var_info['unit_factor']

            processed_features.append(pd.DataFrame(year_results))

    logging.info("Finalizing and saving dataset...")
    final_df = pd.concat(processed_features, ignore_index=True)

    logging.info("Calculating wind speed from u/v components...")
    for season in ['spring', 'summer']:
        u_col = f'{season}_u10_wind_anomaly_forecast'
        v_col = f'{season}_v10_wind_anomaly_forecast'

        # Calculate the magnitude of the wind vector anomaly
        # Note: This is an approximation of the true wind speed anomaly
        final_df[f'{season}_wind_anomaly_forecast'] = np.sqrt(final_df[u_col] ** 2 + final_df[v_col] ** 2)

        # Drop the original u and v component columns
        final_df.drop(columns=[u_col, v_col], inplace=True)

    # Rename dewpoint columns for clarity
    final_df.rename(columns={
        'spring_dewpoint_anomaly_forecast': 'spring_dewpoint_anomaly_forecast',
        'summer_dewpoint_anomaly_forecast': 'summer_dewpoint_anomaly_forecast'
    }, inplace=True)

    final_df.sort_values(by=['year', 'district_no'], inplace=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')

    logging.info(f"--- Processing stage complete. Final dataset saved to {OUTPUT_CSV_PATH} ---")
    print("\n--- Validation of Final Output (Sample from 2021) ---")
    print(final_df[final_df.year == 2021].head())
    print(f"\nTotal Columns Created: {len(final_df.columns)}")


# --- 4. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    logging.info("===== STARTING: Integrated Data Download and Processing Pipeline =====")
    if download_ecmwf_data():
        process_forecasts_to_csv()
    else:
        logging.error("Download failed. Aborting processing.")
    logging.info("===== Pipeline Finished. =====")