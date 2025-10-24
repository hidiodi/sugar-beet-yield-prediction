# File: src/models/experimental_models/lintul/build_weather_generator_ONE_YEAR_TEST.py
# Description: A FAST-RUNNING test version that trains the Weather Generator
#              ONLY on the single year of test data.

import pandas as pd
import joblib
import os
import logging
from build_weather_generator import WeatherGenerator # Imports the class from the original script

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')

# --- TEST CONFIGURATION ---
TEST_YEAR = 2018
# KEY CHANGE: Point to the test weather file you already created
HISTORICAL_DAILY_WEATHER_PATH = f'data/02_intermediate/historical_daily_weather_era5_{TEST_YEAR}_TEST.csv'
# KEY CHANGE: Save to a temporary test generator file
OUTPUT_WG_PATH = 'src/models/weather_generator_TEST.joblib'

if __name__ == "__main__":
    logging.info(f"===== Training TEST Weather Generator on {TEST_YEAR} data ONLY =====")
    try:
        df_daily = pd.read_csv(HISTORICAL_DAILY_WEATHER_PATH, parse_dates=['date'])
        df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)

        wg = WeatherGenerator()
        wg.fit(df_daily)

        os.makedirs(os.path.dirname(OUTPUT_WG_PATH), exist_ok=True)
        joblib.dump(wg, OUTPUT_WG_PATH)

        logging.info(f"✅ TEST Weather Generator trained and saved successfully to {OUTPUT_WG_PATH}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)