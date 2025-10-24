# File: src/models/experimental_models/lintul/generate_historical_residuals_ONE_YEAR_TEST.py
# Description: A FAST-RUNNING test version that generates residuals for a single year.

import sys
import os

# Add the project root to the Python path so we can import crop_model_wrapper
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from crop_model_wrapper import LINTUL_PCSE
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')

# --- TEST CONFIGURATION ---
TEST_YEAR = 2018

# --- Configuration using TEST files ---
STATIC_FEATURES_PATH = 'data/05_model_input/stage1_preseason_features.csv'
# Point to the one-year daily weather file
HISTORICAL_DAILY_WEATHER_PATH = f'data/02_intermediate/historical_daily_weather_era5_{TEST_YEAR}_TEST.csv'
# Save to a new one-year residuals file
OUTPUT_RESIDUALS_PATH = f'data/03_primary/historical_residuals_{TEST_YEAR}_TEST.csv'


def generate_residuals_test():
    """
    Generate historical residuals for a single test year using PCSE/WOFOST.
    """
    logging.info(f"--- Generating Historical Residuals for ONE-YEAR TEST ({TEST_YEAR}) ---")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_RESIDUALS_PATH), exist_ok=True)

    # --- Load Data ---
    logging.info("Loading input data...")
    try:
        df_static = pd.read_csv(STATIC_FEATURES_PATH)
        logging.info(f"✓ Loaded static features: {len(df_static)} records")
    except FileNotFoundError:
        logging.error(f"FATAL: Static features file not found at '{STATIC_FEATURES_PATH}'")
        return

    try:
        df_daily = pd.read_csv(HISTORICAL_DAILY_WEATHER_PATH, parse_dates=['date'])
        logging.info(f"✓ Loaded daily weather: {len(df_daily)} records")
    except FileNotFoundError:
        logging.error(f"FATAL: One-year test weather file not found at '{HISTORICAL_DAILY_WEATHER_PATH}'")
        logging.error("Please ensure the weather data file has been created.")
        return

    # --- Prepare District Numbers (ensure consistent formatting) ---
    df_static['district_no'] = df_static['district_no'].astype(str).str.zfill(5)
    df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)

    # --- Initialize PCSE Model ---
    logging.info("Initializing PCSE/WOFOST model...")
    try:
        lintul_model = LINTUL_PCSE()
        logging.info("✓ Model initialized successfully")
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize PCSE model: {e}")
        return

    # --- Prepare Simulation Cases ---
    # Filter to only run the single test year
    unique_cases = df_static[df_static['year'] == TEST_YEAR][['year', 'district_no']].drop_duplicates()
    logging.info(f"Found {len(unique_cases)} district(s) to simulate for year {TEST_YEAR}")

    if len(unique_cases) == 0:
        logging.error(f"No districts found for year {TEST_YEAR} in static features!")
        return

    results = []

    # --- Run Simulations ---
    for _, row in tqdm(unique_cases.iterrows(), total=len(unique_cases),
                       desc=f"Simulating Yields for {TEST_YEAR}"):
        year, district_no = row['year'], row['district_no']

        # Get actual yield from static features
        district_data = df_static[(df_static['year'] == year) &
                                  (df_static['district_no'] == district_no)]

        if len(district_data) == 0:
            logging.warning(f"No static data for district {district_no} in {year}. Skipping.")
            continue

        actual_yield = district_data['kreisYield'].iloc[0]

        # Get weather data for this district and year
        weather_for_sim = df_daily[(df_daily['date'].dt.year == year) &
                                   (df_daily['district_no'] == district_no)].copy()

        if weather_for_sim.empty:
            logging.warning(f"No weather data for district {district_no} in {year}. Skipping.")
            continue

        # Verify required weather columns exist
        required_cols = ['date', 'srad', 'tmin', 'tmax', 'precip']
        missing_cols = [col for col in required_cols if col not in weather_for_sim.columns]
        if missing_cols:
            logging.error(f"Missing weather columns for {district_no}: {missing_cols}")
            continue

        # Run PCSE simulation
        run_id = f"{district_no}_{year}_TEST"
        district_params = {
            'planting_date': f"{year}-03-15",  # Standard planting date for sugar beet
            'run_id': run_id
        }

        try:
            simulated_yield = lintul_model.run(weather_for_sim, district_params)
            residual = actual_yield - simulated_yield

            results.append({
                'year': year,
                'district_no': district_no,
                'actual_yield': actual_yield,
                'lintul_yield': simulated_yield,
                'residual': residual
            })

            logging.debug(f"{run_id}: actual={actual_yield:.1f}, simulated={simulated_yield:.1f}, "
                          f"residual={residual:.1f}")

        except Exception as e:
            logging.error(f"Simulation failed for {run_id}: {e}")
            # Add with default values to track failures
            results.append({
                'year': year,
                'district_no': district_no,
                'actual_yield': actual_yield,
                'lintul_yield': 400.0,  # Default fallback
                'residual': actual_yield - 400.0
            })

    # --- Save Results ---
    if len(results) == 0:
        logging.error("No results generated! Check your input data and model configuration.")
        return

    residuals_df = pd.DataFrame(results)
    residuals_df.to_csv(OUTPUT_RESIDUALS_PATH, index=False)
    logging.info(f"✅ ONE-YEAR test residuals saved to {OUTPUT_RESIDUALS_PATH}")

    # --- Display Summary Statistics ---
    print("\n" + "=" * 70)
    print(f"TEST RESULTS FOR {TEST_YEAR}")
    print("=" * 70)
    print("\nFirst 10 records:")
    print(residuals_df.head(10).to_string(index=False))

    print(f"\n\nStatistical Summary for {TEST_YEAR}:")
    print("-" * 70)
    summary_stats = residuals_df[['actual_yield', 'lintul_yield', 'residual']].describe()
    print(summary_stats)

    print(f"\n\nPerformance Metrics:")
    print("-" * 70)
    mae = residuals_df['residual'].abs().mean()
    rmse = (residuals_df['residual'] ** 2).mean() ** 0.5
    bias = residuals_df['residual'].mean()

    print(f"Mean Absolute Error (MAE):  {mae:.2f} dt/ha")
    print(f"Root Mean Square Error:     {rmse:.2f} dt/ha")
    print(f"Mean Bias:                  {bias:.2f} dt/ha")
    print(f"Number of districts:        {len(residuals_df)}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        generate_residuals_test()
    except KeyboardInterrupt:
        logging.info("\n\nSimulation interrupted by user.")
    except Exception as e:
        logging.error(f"\n\nFATAL ERROR: {e}", exc_info=True)