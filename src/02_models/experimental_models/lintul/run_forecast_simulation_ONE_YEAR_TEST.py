# File: src/models/experimental_models/lintul/run_forecast_simulation_ONE_YEAR_TEST.py
# Description: A FAST-RUNNING test that simulates a forecast for a single year.
#              It uses the SEAS5 monthly anomalies and the Weather Generator to create
#              daily scenarios and run the crop model.

import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import logging
import os
import time

# Import our custom components
from crop_model_wrapper import LINTUL_PCSE
from build_weather_generator import WeatherGenerator

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')

# --- TEST CONFIGURATION ---
TEST_YEAR = 2018

# --- Input File Paths ---
SEAS5_MEMBER_FEATURES_PATH = 'data/02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv'
WEATHER_GENERATOR_PATH = 'src/models/weather_generator_TEST.joblib'  # Must exist!
OUTPUT_FORECAST_PATH = f'data/03_primary/forecast_simulation_{TEST_YEAR}_TEST.csv'


def run_forecast_simulation_test():
    """
    Orchestrates the forecast simulation for the single test year.
    """
    logging.info(f"--- Running Forecast Simulation for ONE-YEAR TEST ({TEST_YEAR}) ---")

    # --- 1. Load all necessary inputs ---
    try:
        df_member = pd.read_csv(SEAS5_MEMBER_FEATURES_PATH)
        weather_generator = joblib.load(WEATHER_GENERATOR_PATH)
        lintul_model = LINTUL_PCSE()
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required input file was not found. {e}")
        logging.error(
            "Please ensure you have run 'build_forecast_features_by_member.py' and have a trained 'weather_generator.joblib'.")
        return
    except Exception as e:
        logging.error(f"An error occurred during initialization: {e}")
        return

    # --- 2. Filter data for the single test year ---
    seas5_test_year_df = df_member[df_member['year'] == TEST_YEAR]
    if seas5_test_year_df.empty:
        logging.error(f"No SEAS5 forecast data found for the year {TEST_YEAR}. Cannot run test.")
        return

    seas5_test_year_df['district_no'] = seas5_test_year_df['district_no'].astype(str).str.zfill(5)

    forecast_results = []

    # --- 3. Run the simulation loop ---
    # Group by district to process all members for a district at once
    for district_no, group in tqdm(seas5_test_year_df.groupby('district_no'),
                                   desc=f"Simulating Forecast for {TEST_YEAR}"):

        simulated_yields_for_district = []

        # For each of the 51 SEAS5 members, generate a daily weather scenario and run LINTUL
        for _, member_recipe in group.iterrows():

            # Create the monthly anomaly "recipe" for the weather generator
            monthly_anomalies = {}
            for month in range(3, 11):  # March to October
                # This logic assumes two seasons; adjust if your feature file is monthly
                temp_anomaly = member_recipe.get(
                    'spring_temp_anomaly_forecast' if month <= 6 else 'summer_temp_anomaly_forecast', 0)
                precip_anomaly = member_recipe.get(
                    'spring_precip_anomaly_forecast' if month <= 6 else 'summer_precip_anomaly_forecast', 0)

                monthly_anomalies[f'temp_anomaly_{month}'] = temp_anomaly
                # The Weather Generator expects a factor for precip (e.g., -20% anomaly -> 0.8 factor)
                monthly_anomalies[
                    f'precip_anomaly_{month}'] = precip_anomaly / 100  # Assuming anomaly is in % in the CSV

            synthetic_weather = weather_generator.generate(district_no, f'{TEST_YEAR}-03-01', f'{TEST_YEAR}-10-31',
                                                           monthly_anomalies)

            if synthetic_weather.empty:
                logging.warning(
                    f"Weather generator failed for {district_no}, member {member_recipe['seas5_member']}. Appending a failure value.")
                simulated_yields_for_district.append(300.0)  # Failure yield
                continue

            run_id = f"{district_no}_{TEST_YEAR}_{member_recipe['seas5_member']}_forecast_test"
            district_params = {'planting_date': f'{TEST_YEAR}-03-15', 'run_id': run_id}

            simulated_yield = lintul_model.run(synthetic_weather, district_params)
            simulated_yields_for_district.append(simulated_yield)

        # The final forecast baseline is the average of the 51 scenarios
        forecast_baseline = np.mean(simulated_yields_for_district)
        forecast_uncertainty = np.std(simulated_yields_for_district)

        forecast_results.append({
            'year': TEST_YEAR,
            'district_no': district_no,
            'lintul_forecast_baseline': forecast_baseline,
            'forecast_uncertainty_std': forecast_uncertainty  # The spread of the 51 scenarios
        })

    # --- 4. Save and Analyze Results ---
    if not forecast_results:
        logging.error("The simulation produced no results.")
        return

    forecast_df = pd.DataFrame(forecast_results)
    forecast_df.to_csv(OUTPUT_FORECAST_PATH, index=False)

    logging.info(f"✅ ONE-YEAR forecast simulation saved to {OUTPUT_FORECAST_PATH}")
    print("\n--- FORECAST SIMULATION RESULTS ---")
    print(forecast_df.head())
    print(f"\nAnalysis for {TEST_YEAR}:")
    print(forecast_df[['lintul_forecast_baseline', 'forecast_uncertainty_std']].describe())


if __name__ == "__main__":
    run_forecast_simulation_test()