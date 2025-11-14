# File: src/02_models/Wofost7.1/calibrate_regional_parameters.py
# Description: Calibrates regional crop parameters for the WOFOST model.
# ENHANCEMENT 4: This script provides the functionality for regional parameter calibration.

import pandas as pd
import yaml
from pathlib import Path
import sys
import logging
import json
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config
from src.02_models.Wofost7_1.run_wofost_pipeline import SimpleWeatherDataProvider, ParameterProvider, ParameterDict, Wofost72_WLP_FD, _create_district_specific_parameters

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG

# Define German regions by the first two digits of the district_no (Landkreis-Schlüssel)
REGIONS = {
    "North": ["01", "02", "03", "04", "05"],  # SH, HH, NI, HB, NRW
    "East": ["10", "11", "12", "13", "14", "15", "16"], # SL, BE, BB, MV, SN, ST, TH
    "South": ["08", "09"],                 # BW, BY
    "West": ["06", "07"],                   # HE, RP
}

def get_region(district_no):
    """Maps a district_no to its region."""
    prefix = district_no[:2]
    for region, prefixes in REGIONS.items():
        if prefix in prefixes:
            return region
    return "Unknown"

def run_single_simulation(params_to_calibrate, base_crop_params, static_site_row, weather_df, initial_condition_row):
    """
    Runs a single WOFOST simulation with a given set of parameters.
    """
    try:
        cropdata = ParameterDict(base_crop_params.copy())
        cropdata['AMAX'] = params_to_calibrate[0]
        cropdata['TSUM1'] = params_to_calibrate[1]
        cropdata['RUE'] = params_to_calibrate[2]

        parameters, site_data = _create_district_specific_parameters(static_site_row, cropdata, initial_condition_row)
        weather_provider = SimpleWeatherDataProvider(weather_df, site_data)

        crop_start = pd.to_datetime(initial_condition_row['sowing_date']).date()
        crop_end = pd.to_datetime(initial_condition_row['CROP_END_DATE']).date().replace(year=crop_start.year)

        agromanagement = [{
            crop_start: ParameterDict({
                'CropCalendar': ParameterDict({
                    'crop_start_date': crop_start, 'crop_start_type': 'emergence',
                    'crop_end_date': crop_end, 'crop_end_type': 'harvest',
                    'max_duration': CONFIG['AGROMANAGEMENT']['MAX_DURATION']
                }), 'TimedEvents': None, 'StateEvents': None
            })
        }]

        model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)
        model.run_till_terminate()
        output = model.get_output()

        simulated_yield_dry_kgha = output[-1]['TWSO'] if output else 0
        dmc = CONFIG['CONSTANTS']['DMC_SUGARBEET']
        simulated_yield_fresh_dtha = (simulated_yield_dry_kgha / dmc) / 100.0
        return simulated_yield_fresh_dtha

    except Exception:
        return None

def objective_function(params, region_data, base_crop_params):
    """
    Objective function for the optimizer. Calculates RMSE between simulated and actual yields.
    """
    simulated_yields, actual_yields = [], []

    for _, row in region_data.iterrows():
        sim_yield = run_single_simulation(params, base_crop_params, row, row['weather'], row['initial_conditions'])
        if sim_yield is not None:
            simulated_yields.append(sim_yield)
            actual_yields.append(row['kreisYield'])

    if not simulated_yields: return 1e6 # High error if no simulations succeed

    rmse = mean_squared_error(actual_yields, simulated_yields, squared=False)
    return rmse

def main():
    """
    Main function to run the regional calibration.
    """
    logging.info("--- Starting Regional Parameter Calibration ---")

    # Load all necessary data
    try:
        df_yield = pd.read_csv(CONFIG['FILE_PATHS']['YIELD_DATA'], dtype={'district_no': str})
        df_static = pd.read_csv(PROCESSED_DATA_DIR / 'StaticSiteData.csv', dtype={'district_no': str})
        df_ic = pd.read_csv(PROCESSED_DATA_DIR / 'InitialConditions.csv', dtype={'district_no': str}, parse_dates=['sowing_date', 'CROP_END_DATE'])
        df_weather = pd.read_csv(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'], dtype={'district_no': str}, parse_dates=['date'])
        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            cp = yaml.safe_load(f)['CropParameters']
        base_crop_params = {**cp.get('GenericC3', {}), **cp['EcoTypes']['sugarbeet'], **cp['Varieties']['Sugarbeet_601']}
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required data file was not found. Error: {e}"); sys.exit(1)

    # Prepare a master dataframe
    df_master = pd.merge(df_yield, df_static, on='district_no')
    df_master = pd.merge(df_master, df_ic, on=['year', 'district_no'])
    df_master['region'] = df_master['district_no'].apply(get_region)

    # Attach weather and IC data to each row for easier processing
    df_master['weather'] = df_master.apply(lambda row: df_weather[(df_weather['district_no'] == row['district_no']) & (df_weather['date'].dt.year == row['year'])], axis=1)
    df_master['initial_conditions'] = df_master.apply(lambda row: row.to_dict(), axis=1)

    calibrated_params = {}

    for region, df_region in df_master.groupby('region'):
        if region == "Unknown": continue
        logging.info(f"--- Calibrating for region: {region} ({len(df_region)} records) ---")

        # Initial guesses and bounds for [AMAX, TSUM1, RUE]
        initial_guess = [base_crop_params['AMAX'][0], base_crop_params['TSUM1'][0], base_crop_params['RUE'][0]]
        bounds = [(20, 60), (500, 800), (1.5, 2.5)] # Sensible bounds for the parameters

        result = minimize(
            objective_function,
            initial_guess,
            args=(df_region, base_crop_params),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True, 'maxiter': 50} # Limit iterations for speed
        )

        optimal_params = {
            "AMAX": result.x[0],
            "TSUM1": result.x[1],
            "RUE": result.x[2]
        }
        calibrated_params[region] = optimal_params
        logging.info(f"Optimal parameters for {region}: {optimal_params}")

    # Save the calibrated regional parameters
    output_path = PROCESSED_DATA_DIR / 'SugarbeetGenes_Regional.json'
    with open(output_path, 'w') as f:
        json.dump(calibrated_params, f, indent=4)
    logging.info(f"--- Regional parameters saved to {output_path} ---")

if __name__ == "__main__":
    main()
