# src/data/01_preprocess_agronomic_data.py

"""
Processes the raw harmonized district-level data from the Thünen Institute.

This script performs the following streamlined steps:
1.  Loads the 'Final_data.csv' file from the 'data/01_raw/' directory.
2.  Filters the data to keep only records where 'var' is 'sugarbeet' AND
    'measure' is 'yield'.
3.  Drops any rows where the yield value is missing.
4.  Selects and renames the essential columns: 'district_no', 'year', and 'yield'.
5.  Saves the cleaned, focused sugar beet yield data to the
    'data/02_intermediate/' directory.
"""

import os
import pandas as pd
import logging

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    Runs the simplified data preprocessing steps for the agronomic data.
    """
    logging.info("--- 01: Preprocessing Agronomic Data (Simplified) ---")

    # --- 1. Define File Paths ---
    raw_data_path = 'data/01_raw/Final_data.csv'
    intermediate_dir = 'data/02_intermediate'
    # The output file is now more accurately named.
    output_path = os.path.join(intermediate_dir, 'sugarbeet_yield.csv')

    # --- 2. Check for Input File ---
    logging.info(f"Checking for raw data file at '{raw_data_path}'...")
    if not os.path.exists(raw_data_path):
        logging.error(f"File not found at '{raw_data_path}'.")
        logging.error("Please download 'Final_data.csv' from https://doi.org/10.3220/DATA20231117103252-0")
        logging.error("and place it in the 'data/01_raw/' directory.")
        return

    logging.info("SUCCESS: Raw data file found.")

    # --- 3. Load and Filter Data Efficiently ---
    logging.info("Loading and filtering for 'sugarbeet' yield data...")
    try:
        # Load the data, recognizing 'NA' as a null value
        df = pd.read_csv(raw_data_path, na_values='NA')

        # Chain the filtering conditions for a single, efficient operation
        df_filtered = df[
            (df['var'] == 'sugarbeet') &
            (df['measure'] == 'yield')
            ].copy()
        logging.info(f"Found {len(df_filtered)} raw 'sugarbeet' yield records.")

    except Exception as e:
        logging.error(f"An error occurred while reading or filtering the data: {e}")
        return

    # --- 4. Clean and Finalize the Dataset ---
    logging.info("Cleaning data and selecting final columns...")

    # Drop rows where the 'value' (our yield column) is missing
    initial_rows = len(df_filtered)
    df_filtered.dropna(subset=['value'], inplace=True)
    rows_dropped = initial_rows - len(df_filtered)
    if rows_dropped > 0:
        logging.info(f"Dropped {rows_dropped} rows with missing yield values.")

    # Select only the columns we need for the final output
    df_processed = df_filtered[['district_no', 'year', 'value']].copy()

    # Rename 'value' to 'yield' for clarity in downstream scripts
    df_processed.rename(columns={'value': 'yield'}, inplace=True)

    # --- 5. Save Processed Data ---
    try:
        os.makedirs(intermediate_dir, exist_ok=True)
        df_processed.to_csv(output_path, index=False)

        logging.info("--- Preprocessing Complete ---")
        logging.info(f"SUCCESS: Processed data saved to '{output_path}'.")
        logging.info(f"Final dataset has {len(df_processed)} rows and columns: {df_processed.columns.tolist()}")

    except Exception as e:
        logging.error(f"An error occurred while saving the data: {e}")


if __name__ == '__main__':
    main()