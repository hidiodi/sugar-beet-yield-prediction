# File: src/features/calculate_initial_conditions.py
# Description: V1 - Calculates the dynamic, year-specific Initial Available Water (WAV)
# for each district by running a simple winter fallow-season water balance model.

import pandas as pd
from pathlib import Path
import sys
import logging
from tqdm import tqdm
from pcse.util import penman_monteith
import datetime
import geopandas as gpd

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]  # Adjusted for src/features/
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

WINTER_START_MONTH = 11
WINTER_START_DAY = 1
SIMULATION_START_MONTH = 3
SIMULATION_START_DAY = 20


# --- Main Calculation Function ---
def run_winter_balance_for_district(district_weather, site_and_soil_props):
    """Runs a daily water balance for a single district over one winter period."""
    # These are read from the V4 static features file
    rdmsol = site_and_soil_props['RDMSOL']
    smw_frac = site_and_soil_props['SMW']
    smfcf_frac = site_and_soil_props['SMFCF']

    wilting_point_cm = rdmsol * smw_frac
    field_capacity_cm = rdmsol * smfcf_frac

    # Assumption: Soil profile is at Field Capacity at the start of the winter period.
    current_water_cm = field_capacity_cm

    for _, day in district_weather.iterrows():
        precip_cm = day['precip'] / 10.0

        # Penman-Monteith inputs
        date = day['date'].to_pydatetime().date()
        lat = site_and_soil_props['latitude']
        elev = site_and_soil_props['avg_elevation']
        tmin, tmax = day['tmin'], day['tmax']
        srad_kj = day['srad'] * 1000.0
        vap_hpa = day.get('vap', config.WOFOST_CONFIG['WEATHER_DEFAULTS']['VAPOR_PRESSURE']) * 10.0
        wind = day.get('wind', config.WOFOST_CONFIG['WEATHER_DEFAULTS']['WIND_SPEED'])

        et0_mm = penman_monteith(date, lat, elev, tmin, tmax, srad_kj, vap_hpa, wind)
        es0_cm = et0_mm / 10.0

        current_water_cm = current_water_cm + precip_cm - es0_cm
        current_water_cm = max(wilting_point_cm, min(current_water_cm, field_capacity_cm))

    wav_final = current_water_cm - wilting_point_cm
    return wav_final


def calculate_initial_wav():
    """Main function to orchestrate the calculation of WAV for all districts and years."""
    logging.info("--- Starting V1 Initial Available Water (WAV) Calculation ---")

    # Uses the path you just added to config.py
    output_path = config.WOFOST_CONFIG['FILE_PATHS']['INITIAL_CONDITIONS']
    if output_path.exists():
        logging.info(f"Initial conditions file already exists at '{output_path}'. Skipping.")
        return

    logging.info("Loading static soil & site data...")
    try:
        # Reads the output from your V4 build_static_features.py script
        static_df = pd.read_csv(config.WOFOST_CONFIG['FILE_PATHS']['STATIC_SOIL_FEATURES'], dtype={'district_no': str})

        # Adds latitude needed for Penman-Monteith
        gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH)
        gdf_districts['latitude'] = gdf_districts.geometry.centroid.y
        gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        static_df = pd.merge(static_df, gdf_districts[['district_no', 'latitude']], on='district_no')
    except FileNotFoundError as e:
        logging.error(f"FATAL: Required data file not found. Error: {e}");
        sys.exit(1)

    logging.info("Loading all historical weather data...")
    all_weather_dfs = []
    # Reads all weather files from the directory defined in your config
    weather_dir = config.WOFOST_CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR']
    for fpath in weather_dir.glob("*.csv"):
        df = pd.read_csv(fpath, parse_dates=['date'], dtype={'district_no': str})
        all_weather_dfs.append(df)
    if not all_weather_dfs:
        logging.error("FATAL: No historical weather files found.");
        sys.exit(1)
    full_weather_df = pd.concat(all_weather_dfs, ignore_index=True)

    results = []
    start_year = config.WOFOST_CONFIG['START_YEAR']
    end_year = config.WOFOST_CONFIG['END_YEAR']

    pbar_outer = tqdm(static_df.iterrows(), total=len(static_df), desc="Processing Districts")
    for _, district_row in pbar_outer:
        district_no = district_row['district_no']
        pbar_outer.set_postfix_str(f"District: {district_no}")

        district_weather_all_years = full_weather_df[full_weather_df['district_no'] == district_no].copy()
        if district_weather_all_years.empty:
            continue

        for year in range(start_year, end_year + 1):
            winter_start_date = f"{year - 1}-{WINTER_START_MONTH}-{WINTER_START_DAY}"
            sim_start_date = f"{year}-{SIMULATION_START_MONTH}-{SIMULATION_START_DAY}"

            mask = (
                    (district_weather_all_years['date'] >= winter_start_date) &
                    (district_weather_all_years['date'] < sim_start_date)
            )
            winter_weather = district_weather_all_years[mask].sort_values('date')

            if winter_weather.empty: continue

            wav = run_winter_balance_for_district(winter_weather, district_row)
            results.append({'year': year, 'district_no': district_no, 'WAV': wav})

    if not results:
        logging.error("No WAV results generated. Check weather data.");
        return

    results_df = pd.DataFrame(results)
    logging.info(f"Saving {len(results_df)} calculated WAV records to '{output_path}'")
    results_df.to_csv(output_path, index=False, float_format='%.4f')

    logging.info("--- SUCCESS: Initial conditions (WAV) file created! ---")


if __name__ == "__main__":
    calculate_initial_wav()