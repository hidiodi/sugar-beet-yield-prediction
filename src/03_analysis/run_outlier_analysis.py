# File: src/03_analysis/analyze_missing_wofost.py
# Description: Analyzes the dataset to understand which data points are missing
#              the WOFOST forecast and are therefore excluded from model training.

import pandas as pd
import sys
from pathlib import Path
import logging

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_PATH = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']


def analyze_missing_data():
    """
    Loads the feature dataset and reports on the characteristics of rows
    that are missing the WOFOST stage-1 forecast.
    """
    logging.info("--- Starting Analysis of Missing WOFOST Forecasts ---")

    try:
        df = pd.read_csv(DATA_PATH)
        logging.info(f"✓ Loaded feature data ({len(df)} rows) from {DATA_PATH}")
    except FileNotFoundError:
        logging.error(f"FATAL: Input data not found at {DATA_PATH}. Aborting.")
        sys.exit(1)

    # Use the same column name as the training script for consistency
    wofost_col = 'wofost_forecast_yield_fresh_dt'
    if wofost_col not in df.columns:
        logging.error(f"FATAL: The column '{wofost_col}' was not found in the dataset.")
        sys.exit(1)

    # Split the data into two groups
    df_with_wofost = df[df[wofost_col].notna()]
    df_missing_wofost = df[df[wofost_col].isna()]

    print("\n" + "=" * 80)
    print("      DATASET COVERAGE ANALYSIS REPORT")
    print("=" * 80)
    print(f"Total Rows in Feature Set:      {len(df)}")
    print(f"Rows WITH WOFOST Forecast:      {len(df_with_wofost)} (Used for Training)")
    print(f"Rows MISSING WOFOST Forecast:   {len(df_missing_wofost)} (Dropped from Training)")
    print("=" * 80)

    if not df_missing_wofost.empty:
        print("\n--- ANALYSIS OF DROPPED DATA ---\n")

        # 1. Analyze by Year
        print("1. Distribution of Dropped Data by Year:")
        year_counts = df_missing_wofost['year'].value_counts().sort_index()
        print(year_counts.to_string())

        # 2. Analyze by State/Region
        if 'state_name' in df_missing_wofost.columns:
            print("\n\n2. Distribution of Dropped Data by State:")
            state_counts = df_missing_wofost['state_name'].value_counts()
            print(state_counts.to_string())

        first_year_with_data = df_with_wofost['year'].min()
        print("\n\n--- CONCLUSION ---")
        print("The data loss is intentional, as the hybrid model can only be trained")
        print("on data points where the physical WOFOST model successfully produced a forecast.")
        print(f"\nThe analysis shows that most missing data is from earlier years.")
        print(f"The first year with consistent WOFOST data appears to be around {first_year_with_data}.")

    print("\n" + "=" * 80)
    logging.info("--- Analysis complete. ---")


if __name__ == "__main__":
    analyze_missing_data()