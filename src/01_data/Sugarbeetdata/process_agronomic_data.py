"""
Processes and combines agronomic data from multiple sources.

This script performs the following streamlined steps:
1.  Loads the 'Final_data.csv' (Thünen Institute data).
2.  Filters it for sugar beet yield, cleans it, and converts units to dt/ha.
3.  Loads a supplementary dataset, 'modernYield_clean.csv'.
4.  Standardizes the column names of the supplementary data.
5.  Combines the two datasets into a single master file.
6.  Removes any duplicate district-year entries.
7.  Saves the final, combined sugar beet yield data to the
    'data/02_intermediate/' directory.
"""

import os
import pandas as pd
import logging

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    Runs the data preprocessing and merging steps for the agronomic data.
    """
    logging.info("--- 01: Preprocessing and Merging Agronomic Data ---")

    # --- 1. Define File Paths ---
    thuenen_data_path = 'data/01_raw/Final_data.csv'
    modern_yield_path = 'data/01_raw/modernYield_clean.csv'
    intermediate_dir = 'data/02_intermediate'
    output_path = os.path.join(intermediate_dir, 'sugarbeet_yield.csv')

    # --- 2. Process Thünen Institute Data ---
    logging.info(f"Checking for Thünen data file at '{thuenen_data_path}'...")
    if not os.path.exists(thuenen_data_path):
        logging.error(f"File not found at '{thuenen_data_path}'.")
        logging.error("Please download 'Final_data.csv' from https://doi.org/10.3220/DATA20231117103252-0")
        logging.error("and place it in the 'data/01_raw/' directory.")
        return
    logging.info("SUCCESS: Thünen data file found.")

    try:
        logging.info("Loading and filtering Thünen data for 'sugarbeet' yield...")
        df_thuenen = pd.read_csv(thuenen_data_path, na_values='NA')
        df_thuenen_filtered = df_thuenen[
            (df_thuenen['var'] == 'sugarbeet') & (df_thuenen['measure'] == 'yield')
        ].copy()

        logging.info("Cleaning, converting, and finalizing Thünen data...")
        df_thuenen_filtered.dropna(subset=['value'], inplace=True)
        df_thuenen_processed = df_thuenen_filtered[['district_no', 'year', 'value']].copy()
        df_thuenen_processed.rename(columns={'value': 'yield'}, inplace=True)

        # Convert yield from gt/ha to dt/ha (1 gt = 10 dt) and round
        df_thuenen_processed['yield'] = (df_thuenen_processed['yield'] * 10).round(1)
        logging.info(f"Processed Thünen data has {len(df_thuenen_processed)} records.")

    except Exception as e:
        logging.error(f"An error occurred while processing the Thünen data: {e}")
        return

    # --- 3. Load and Process Modern Yield Data ---
    logging.info(f"Checking for modern yield data file at '{modern_yield_path}'...")
    if not os.path.exists(modern_yield_path):
        logging.warning(f"Optional file not found at '{modern_yield_path}'. Skipping merge.")
        df_final = df_thuenen_processed
    else:
        logging.info("SUCCESS: Modern yield data file found. Processing and merging...")
        try:
            # Load the data using semicolon as the delimiter
            df_modern = pd.read_csv(modern_yield_path, sep=';')
            logging.info(f"Loaded {len(df_modern)} records from modern yield data.")

            # Rename columns to match the standard format
            df_modern.rename(columns={
                'Jahr': 'year',
                'ID': 'district_no',
                'Zuckerrben': 'yield'
            }, inplace=True)

            # Ensure the column order is consistent
            df_modern = df_modern[['district_no', 'year', 'yield']]

            # --- 4. Combine Datasets ---
            logging.info("Combining Thünen data with modern yield data...")
            df_final = pd.concat([df_thuenen_processed, df_modern], ignore_index=True)

            # Sort to make duplicates more apparent before dropping
            df_final.sort_values(by=['district_no', 'year'], inplace=True)

            initial_rows = len(df_final)
            df_final.drop_duplicates(subset=['district_no', 'year'], keep='last', inplace=True)
            rows_dropped = initial_rows - len(df_final)
            if rows_dropped > 0:
                logging.info(f"Dropped {rows_dropped} duplicate district-year entries, keeping the last one.")

        except Exception as e:
            logging.error(f"An error occurred while processing or merging the modern yield data: {e}")
            return

    # --- 5. Save Final Combined Data ---
    try:
        os.makedirs(intermediate_dir, exist_ok=True)
        df_final.to_csv(output_path, index=False)

        logging.info("--- Preprocessing Complete ---")
        logging.info(f"SUCCESS: Combined data saved to '{output_path}'.")
        logging.info(f"Final dataset has {len(df_final)} rows and columns: {df_final.columns.tolist()}")

    except Exception as e:
        logging.error(f"An error occurred while saving the final data: {e}")


if __name__ == '__main__':
    main()