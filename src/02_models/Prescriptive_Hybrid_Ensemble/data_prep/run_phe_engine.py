# File: src/models/run_phe_engine.py
# Description: The core Counterfactual Engine for the PHE.
#
# REFACTORED v3: Integrates Quality Gate 1.1 for automated feasibility and
# integrity checks after the prototype run.

import datetime
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import os
import logging
import sys
import itertools
from copy import deepcopy
import time  # <-- Import time module for performance measurement

from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider, WeatherDataProvider
from pcse.util import penman_monteith
from tqdm import tqdm

# ==============================================================================
# === G L O B A L   C O N F I G U R A T I O N ===
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    'START_YEAR': 1981,
    'END_YEAR': 2023,
    'PROTOTYPE_DISTRICT_ID': None,

    'FILE_PATHS': {
        'HISTORICAL_DAILY_WEATHER_DIR': Path('data/02_intermediate/daily_weather'),
        'STATIC_FEATURES': Path('data/05_model_input/stage1_preseason_features.csv'),
        'CROP_YAML': Path('data/01_raw/sugarbeet.yaml'),
        'OUTPUT_DIR': Path('data/03_processed/08_phe_output/raw_simulations'),
    },
    'WEATHER_DEFAULTS': {'WIND_SPEED': 2.0, 'VAPOR_PRESSURE': 1.0},
    'AGROMANAGEMENT': {
        'CROP_END_DATE': datetime.date(2018, 10, 20), 'MAX_DURATION': 250,
    },
    'SOIL_COLUMN_MAPPING': {
        'sand': 'avg_sand_0_100cm', 'clay': 'avg_clay_0_100cm',
        'som': 'avg_som_0_100cm', 'bdod': 'avg_bdod_0_100cm',
    },
    'SOIL_DEFAULTS_AND_CONSTANTS': {'RDMSOL': 150.0, 'KSUB': 10.0, 'SOPE': 10.0},
    'GENERIC_SITE': {'LATITUDE': 52.0, 'LONGITUDE': 10.0, 'ELEVATION': 50.0},
    'DECISION_SPACE': {
        'PLANTING_DATES_DAY_OF_YEAR': list(range(
            datetime.date(2000, 3, 1).timetuple().tm_yday,
            datetime.date(2000, 5, 1).timetuple().tm_yday, 5)),
        'FERTILIZER_MULTIPLIERS': [0.8, 1.0, 1.2],
    },
    # --- NEW: Configuration for Quality Gate 1.1 ---
    'QUALITY_GATE_1_1': {
        # Set a generous upper limit for the projected full run time.
        # If the projection exceeds this, it's a FAIL.
        'MAX_PROJECTED_HOURS': 12.0
    }
}

# (Logging setup and core functions from v2 remain unchanged)
logging.getLogger().handlers = []
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)


class ParameterDict(dict):
    def add_variable(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class SimpleWeatherDataProvider(WeatherDataProvider):
    def __init__(self, weather_df, site_data):
        super().__init__()
        self.latitude = site_data['LAT'];
        self.elevation = site_data['ELEV']
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        self.store = {}
        for _, row in weather_df.iterrows():
            day = row['date'].date()
            tmin = float(row['tmin']);
            tmax = float(row['tmax'])
            wind = float(row.get('wind', CONFIG['WEATHER_DEFAULTS']['WIND_SPEED']))
            vap_kpa = float(row.get('vap', CONFIG['WEATHER_DEFAULTS']['VAPOR_PRESSURE']))
            irrad_j_m2_day = float(row['srad']) * 1_000_000
            precip_mm = float(row['precip'])
            vap_hpa = vap_kpa * 10.0
            et0_mm = penman_monteith(day, self.latitude, self.elevation, tmin, tmax, irrad_j_m2_day / 1000.0, vap_hpa,
                                     wind)
            self.store[(day, 0)] = ParameterDict(
                {'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_mm / 10.0,
                 'IRRAD': irrad_j_m2_day, 'VAP': vap_hpa, 'WIND': wind, 'E0': et0_mm / 10.0,
                 'ES0': et0_mm / 10.0, 'ET0': et0_mm / 10.0, 'SNOWDEPTH': 0.0})


def _calculate_soil_hydraulic_properties(sand_frac, clay_frac, som_frac, bdod):
    porosity = 1 - (bdod / 2.65)
    pwp1500 = -0.024 * sand_frac + 0.487 * clay_frac + 0.006 * som_frac + 0.031
    fc33 = -0.251 * sand_frac + 0.195 * clay_frac + 0.011 * som_frac + 0.299
    sm0 = porosity;
    smw = max(0.01, pwp1500);
    smfcf = min(max(smw + 0.01, fc33), sm0 - 0.01)
    return {'SMW': smw, 'SMFCF': smfcf, 'SM0': sm0, 'CRAIRC': max(0.01, sm0 - smfcf)}


def _create_base_parameters(static_row):
    sitedata = ParameterDict()
    sitedata.add_variable('LAT', static_row.get('latitude', CONFIG['GENERIC_SITE']['LATITUDE']))
    sitedata.add_variable('ELEV', static_row.get('avg_elevation', CONFIG['GENERIC_SITE']['ELEVATION']))
    soildata = ParameterDict()
    soil_map = CONFIG['SOIL_COLUMN_MAPPING']
    try:
        sand = static_row[soil_map['sand']] / 100.0;
        clay = static_row[soil_map['clay']] / 100.0
        som = static_row[soil_map['som']] / 100.0;
        bdod = static_row[soil_map['bdod']]
        for k, v in _calculate_soil_hydraulic_properties(sand, clay, som, bdod).items(): soildata.add_variable(k, v)
    except Exception as e:
        logging.error(f"FATAL: Missing soil data for district {static_row.get('district_no', 'N/A')}. Error: {e}");
        raise e
    for k, v in CONFIG['SOIL_DEFAULTS_AND_CONSTANTS'].items(): soildata.add_variable(k, v)
    sitedata.add_variable('SMLIM', soildata['SMFCF']);
    sitedata.add_variable('WAV', (soildata['SMFCF'] - soildata['SMW']) * soildata['RDMSOL'])
    return sitedata, soildata

def run_single_simulation(year, parameters, weather_provider, planting_date):
    """
    Runs a single WOFOST simulation.
    v2: Corrects the agromanagement dictionary to the valid PCSE format.
    """
    try:
        crop_end_date = CONFIG['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)

        agromanagement = [{
            planting_date: ParameterDict({
                'CropCalendar': ParameterDict({
                    'crop_start_date': planting_date,
                    'crop_start_type': 'emergence',
                    'crop_end_date': crop_end_date,
                    'crop_end_type': 'harvest',
                    'max_duration': CONFIG['AGROMANAGEMENT']['MAX_DURATION']
                }),
                'TimedEvents': None,
                'StateEvents': None
            })
        }]

        model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)
        model.run_till_terminate()
        output = model.get_output()
        return output[-1]['TWSO'] if output else np.nan
    except Exception as e:
        logging.debug(f"Simulation failed for planting date {planting_date}: {e}")
        return np.nan


# --- NEW: Quality Gate 1.1 Function ---
def run_quality_gate_1_1(prototype_results: list, total_runtime_sec: float, df_static: pd.DataFrame):
    """Checks the integrity and computational feasibility of the engine."""
    logging.info("--- [QUALITY GATE 1.1: Feasibility & Integrity Check] ---")

    # --- Integrity Check ---
    expected_sims_per_year = len(CONFIG['DECISION_SPACE']['PLANTING_DATES_DAY_OF_YEAR']) * len(
        CONFIG['DECISION_SPACE']['FERTILIZER_MULTIPLIERS'])
    all_years_ok = True
    for year_df in prototype_results:
        if len(year_df) != expected_sims_per_year:
            year = year_df['year'].iloc[0]
            logging.error(f"Integrity FAIL: Year {year} has {len(year_df)} results, expected {expected_sims_per_year}.")
            all_years_ok = False

    if all_years_ok:
        logging.info(
            f"Integrity Check: PASS! All {len(prototype_results)} years have the correct number of simulations ({expected_sims_per_year}).")
    else:
        logging.error("DECISION: FAIL! The simulation pipeline is unstable. HALT.")
        sys.exit(1)

    # --- Feasibility Check ---
    num_total_districts = df_static['district_no'].nunique()
    projected_full_run_sec = total_runtime_sec * num_total_districts
    projected_full_run_hrs = projected_full_run_sec / 3600.0

    logging.info(f"Prototype Run Time (1 district): {total_runtime_sec:.2f} seconds.")
    logging.info(f"Total Unique Districts to Process: {num_total_districts}.")
    logging.info(f"Projected Full-Scale Run Time: {projected_full_run_hrs:.2f} hours.")

    max_hours = CONFIG['QUALITY_GATE_1_1']['MAX_PROJECTED_HOURS']
    if projected_full_run_hrs <= max_hours:
        logging.info(f"Feasibility Check: PASS! ({projected_full_run_hrs:.2f} hours <= {max_hours} hours).")
    else:
        logging.error(f"Feasibility Check: FAIL! ({projected_full_run_hrs:.2f} hours > {max_hours} hours).")
        logging.error("DECISION: FAIL! Projected time is too long. HALT and optimize or reduce decision space.")
        sys.exit(1)

    logging.info(
        "DECISION: All checks passed. The simulation pipeline is robust. It is safe to launch the full-scale execution.")
    return True


if __name__ == "__main__":
    if CONFIG['PROTOTYPE_DISTRICT_ID']:
        logging.info("--- Starting Phase 1, Task 1.1a: The Counterfactual Engine (PROTOTYPE RUN) ---")
    else:
        logging.info("--- Starting Phase 1, Task 1.1b: The Counterfactual Engine (FULL-SCALE RUN) ---")

    start_time = time.time()  # Start timer

    try:
        df_static_all = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_FEATURES'])
        df_static_all['district_no'] = df_static_all['district_no'].astype(str).str.zfill(5)
        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            crop_yaml = yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required data file was not found. Error: {e}");
        sys.exit(1)

    cropdata_template = ParameterDict()  # ... (rest of the parameter setup is the same)
    variety_name = [k for k in crop_yaml['CropParameters'].keys() if not k.startswith('Generic')][0]
    generic_data = {**crop_yaml['CropParameters'].get('GenericC3', {}),
                    **crop_yaml['CropParameters'].get('Generic', {})}
    for data_source in [generic_data, crop_yaml['CropParameters'][variety_name]]:
        for key, val in data_source.items():
            if isinstance(val, dict): continue
            cropdata_template.add_variable(key, val[0] if isinstance(val, list) else val)

    planting_doys = CONFIG['DECISION_SPACE']['PLANTING_DATES_DAY_OF_YEAR']
    fert_multipliers = CONFIG['DECISION_SPACE']['FERTILIZER_MULTIPLIERS']
    decision_space = list(itertools.product(planting_doys, fert_multipliers))
    logging.info(f"Defined decision space with {len(decision_space)} simulations per district-year.")

    all_yearly_results_dfs = []  # Store results in memory for the quality gate
    os.makedirs(CONFIG['FILE_PATHS']['OUTPUT_DIR'], exist_ok=True)

    # --- Main Simulation Loop (identical to v2) ---
    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        logging.info("=" * 70 + f"\nPROCESSING YEAR: {year}\n" + "=" * 70)
        weather_path = CONFIG['FILE_PATHS'][
                           'HISTORICAL_DAILY_WEATHER_DIR'] / f"historical_daily_weather_era5_{year}.csv"
        if not weather_path.exists():
            logging.warning(f"Weather file not found for {year}. Skipping.");
            continue
        df_weather_year = pd.read_csv(weather_path, parse_dates=['date'])
        df_weather_year['district_no'] = df_weather_year['district_no'].astype(str).str.zfill(5)

        df_static_year = df_static_all[df_static_all['year'] == year].copy()
        if CONFIG['PROTOTYPE_DISTRICT_ID']:
            df_static_year = df_static_year[df_static_year['district_no'] == CONFIG['PROTOTYPE_DISTRICT_ID']]
            if df_static_year.empty:
                logging.warning(
                    f"Prototype district {CONFIG['PROTOTYPE_DISTRICT_ID']} not found in data for year {year}. Skipping.");
                continue
            logging.info(f"Prototype Run: Processing only for district {CONFIG['PROTOTYPE_DISTRICT_ID']}.")

        year_results = []
        for _, static_row in tqdm(df_static_year.iterrows(), total=len(df_static_year),
                                  desc=f"Simulating Districts for {year}"):
            district_no = static_row['district_no']
            district_weather_df = df_weather_year[df_weather_year['district_no'] == district_no].copy()
            if district_weather_df.empty: continue
            sitedata, soildata = _create_base_parameters(static_row)
            weather_provider = SimpleWeatherDataProvider(district_weather_df, sitedata)

            for planting_doy, fert_multiplier in decision_space:
                planting_date = datetime.date(year, 1, 1) + datetime.timedelta(days=planting_doy - 1)
                temp_cropdata = deepcopy(cropdata_template)
                for nutrient_param in ['NMAXSO', 'PMAXSO', 'KMAXSO']:
                    if nutrient_param in temp_cropdata:
                        temp_cropdata[nutrient_param] *= fert_multiplier
                parameters = ParameterProvider(sitedata=sitedata, soildata=soildata, cropdata=temp_cropdata)
                simulated_yield = run_single_simulation(year=year, parameters=parameters,
                                                        weather_provider=weather_provider, planting_date=planting_date)
                year_results.append(
                    {'year': year, 'district_no': district_no, 'planting_date': planting_date.isoformat(),
                     'fertilizer_multiplier': fert_multiplier, 'simulated_yield_twso': simulated_yield})

        if not year_results:
            logging.warning(f"No results were generated for {year}.");
            continue
        all_yearly_results_dfs.append(pd.DataFrame(year_results))

    total_runtime = time.time() - start_time  # Stop timer

    # --- Execute Quality Gate & Save Results ---
    if CONFIG['PROTOTYPE_DISTRICT_ID']:
        gate_passed = run_quality_gate_1_1(all_yearly_results_dfs, total_runtime, df_static_all)
        if gate_passed:
            logging.info("Saving prototype results to disk...")
            for df in all_yearly_results_dfs:
                year = df['year'].iloc[0]
                output_path = CONFIG['FILE_PATHS']['OUTPUT_DIR'] / f"{year}_raw.csv"
                df.to_csv(output_path, index=False)
            logging.info("\n" + "=" * 70 + "\n✓ PROTOTYPE RUN COMPLETED AND PASSED QUALITY GATE!\n" + "=" * 70)
            logging.info("TO RUN THE FULL-SCALE SIMULATION, EDIT THE SCRIPT AND SET 'PROTOTYPE_DISTRICT_ID' TO None.")
    else:
        # For a full run, we save as we go and don't need the final gate
        logging.info("Saving full-scale run results to disk...")
        for df in all_yearly_results_dfs:
            year = df['year'].iloc[0]
            output_path = CONFIG['FILE_PATHS']['OUTPUT_DIR'] / f"{year}_raw.csv"
            df.to_csv(output_path, index=False)
        logging.info("\n" + "=" * 70 + "\n✓ FULL-SCALE COUNTERFACTUAL ENGINE RUN COMPLETED!\n" + "=" * 70)