# src/data/01_preprocess_agronomic_data.py

"""
Processes the raw harmonized district-level data from the Thünen Institute.

This script performs the following steps:
1.  Loads the 'Final_data.csv' file from the 'data/01_raw/' directory.
2.  Filters the data to keep only records related to 'sugarbeet'.
3.  Pivots the data from a long format (with 'measure' and 'value' columns)
    to a wide format, creating separate 'area' and 'yield' columns.
4.  Drops rows where both 'area' and 'yield' are missing.
5.  Saves the cleaned, wide-format sugar beet data to the
    'data/02_intermediate/' directory.
"""

import os
import pandas as pd


def main():
    """
    Runs the data preprocessing steps for the agronomic data.
    """
    print("--- 01: Preprocessing Agronomic Data ---")

    # Define file paths
    raw_data_path = 'data/01_raw/Final_data.csv'
    intermediate_dir = 'data/02_intermediate'
    output_path = os.path.join(intermediate_dir, 'sugarbeet_yield_area.csv')

    # --- 1. Check for Input File ---
    print(f"Checking for raw data file at '{raw_data_path}'...")
    if not os.path.exists(raw_data_path):
        print(f"ERROR: File not found at '{raw_data_path}'.")
        print("Please download 'Final_data.csv' from https://doi.org/10.3220/DATA20231117103252-0")
        print("and place it in the 'data/01_raw/' directory.")
        return  # Exit if the source file is not found

    print("SUCCESS: Raw data file found.")

    # --- 2. Load and Filter Data ---
    print("Loading and filtering for 'sugarbeet' data...")
    try:
        # Load the data, recognizing 'NA' as a null value
        df = pd.read_csv(raw_data_path, na_values='NA')

        # Filter for rows where the variable is 'sugarbeet'
        df_sugarbeet = df[df['var'] == 'sugarbeet'].copy()

        # We no longer need the 'var' or 'outlier' columns
        df_sugarbeet = df_sugarbeet.drop(columns=['var', 'outlier'])

    except Exception as e:
        print(f"An error occurred while reading or filtering the data: {e}")
        return

    # --- 3. Reshape Data (Pivot) ---
    print("Reshaping data from long to wide format...")
    # Set the index for our unique identifiers
    id_vars = ['district_no', 'district', 'nuts_id', 'year']

    # Pivot the 'measure' column so 'area' and 'yield' become columns
    df_wide = df_sugarbeet.pivot_table(
        index=id_vars,
        columns='measure',
        values='value'
    ).reset_index()

    # Clean up the column index name left by the pivot operation
    df_wide.columns.name = None

    print(f"Initial processed rows: {len(df_wide)}")

    # --- 4. Clean Missing Data ---
    # As per the requirements, we drop a record only if BOTH area and yield are missing.
    print("Dropping records where both 'area' and 'yield' are missing...")
    df_processed = df_wide.dropna(subset=['area', 'yield'], how='all')

    rows_dropped = len(df_wide) - len(df_processed)
    print(f"Dropped {rows_dropped} rows.")

    # --- 5. Save Processed Data ---
    try:
        # Ensure the output directory exists
        os.makedirs(intermediate_dir, exist_ok=True)

        # Save the final dataframe to a new CSV file
        df_processed.to_csv(output_path, index=False)

        print("--- Preprocessing Complete ---")
        print(f"SUCCESS: Processed data saved to '{output_path}'.")

    except Exception as e:
        print(f"An error occurred while saving the data: {e}")


if __name__ == '__main__':
    main()