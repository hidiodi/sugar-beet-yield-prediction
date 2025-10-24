# File: src/features/generate_historical_residuals.py
# Description: Uses the updated PCSE/WOFOST wrapper and historical daily weather
#              to generate the historical errors (residuals) of the sugar beet crop model.

# ACTION: Updated to use the new PCSE-based wrapper
from crop_model_wrapper import LINTUL_PCSE
import pandas as pd
from tqdm import tqdm
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')

# --- Configuration ---
STATIC_FEATURES_PATH = 'data/05_model_input/stage1_preseason_features.csv'
HISTORICAL_DAILY_WEATHER_PATH = 'data/02_intermediate/historical_daily_weather_era5.csv'
OUTPUT_RESIDUALS_PATH = 'data/03_primary/historical_residuals.csv'

def generate_residuals():
    """
    Runs the WOFOST sugar beet model for each historical year and district, calculates
    the error (residual) against observed yields, and saves the results.
    """
    logging.info("--- Generating Historical Residuals for ML Training ---")
    os.makedirs(os.path.dirname(OUTPUT_RESIDUALS_PATH), exist_ok=True)

    try:
        df_static = pd.read_csv(STATIC_FEATURES_PATH)
        df_daily = pd.read_csv(HISTORICAL_DAILY_WEATHER_PATH, parse_dates=['date'])
    except FileNotFoundError as e:
        logging.error(f"FATAL: Could not find required input data. Error: {e}")
        return

    # Ensure consistent district number formatting
    df_static['district_no'] = df_static['district_no'].astype(str).str.zfill(5)
    df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)

    # ACTION: Initialize the new, simplified LINTUL_PCSE model wrapper
    try:
        lintul_model = LINTUL_PCSE()
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize the LINTUL_PCSE wrapper. Error: {e}")
        return

    if lintul_model.is_proxy:
        logging.warning("!!! LINTUL wrapper is running in PROXY mode. This should not be the case for the PCSE model. !!!")

    results = []
    unique_cases = df_static[['year', 'district_no']].drop_duplicates()
    for _, row in tqdm(unique_cases.iterrows(), total=len(unique_cases), desc="Simulating Historical Yields"):
        year, district_no = row['year'], row['district_no']

        case_static_data = df_static[(df_static['year'] == year) & (df_static['district_no'] == district_no)]
        if case_static_data.empty:
            continue
        actual_yield = case_static_data['kreisYield'].iloc[0]

        # Filter weather data for the specific year and district
        weather_for_sim = df_daily[(df_daily['date'].dt.year == year) & (df_daily['district_no'] == district_no)]
        if weather_for_sim.empty:
            logging.warning(f"No historical weather data found for {district_no} in {year}. Skipping.")
            continue

        # Each simulation needs a unique ID for temporary files
        run_id = f"{district_no}_{year}_{int(time.time_ns())}"
        # Provide the necessary parameters for the simulation run
        district_params = {'planting_date': f"{year}-03-15", 'run_id': run_id}

        simulated_yield = lintul_model.run(weather_for_sim, district_params)
        residual = actual_yield - simulated_yield

        results.append({
            'year': year,
            'district_no': district_no,
            'actual_yield': actual_yield,
            'lintul_yield': simulated_yield,
            'residual': residual
        })

    if not results:
        logging.error("No simulations were successful. Residuals file will not be created.")
        return

    residuals_df = pd.DataFrame(results)
    residuals_df.to_csv(OUTPUT_RESIDUALS_PATH, index=False)
    logging.info(f"✅ Historical residuals generated and saved to {OUTPUT_RESIDUALS_PATH}")


if __name__ == "__main__":
    generate_residuals()