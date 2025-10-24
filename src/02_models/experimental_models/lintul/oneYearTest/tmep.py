# File: check_merged_file.py
# Description: A simple, non-invasive diagnostic script to list the variables
#              inside the final merged NetCDF file.

import xarray as xr
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# This path MUST point to the final output of your unchanged '03_process_agera5_data.py'
MERGED_NC_PATH = Path("data/02_intermediate/agera5_germany_merged.nc")


def check_file_contents():
    """Opens the merged file and reports its contents."""
    print("\n" + "=" * 70)
    print(f"--- Checking contents of: {MERGED_NC_PATH.name} ---")
    print("=" * 70)

    if not MERGED_NC_PATH.exists():
        logging.error(f"FAIL: File not found at '{MERGED_NC_PATH}'")
        print("Please run your '03_process_agera5_data.py' script first.")
        return

    try:
        with xr.open_dataset(MERGED_NC_PATH) as ds:
            print("\nFile opened successfully. Here is the structure:")
            print("\n--- DIMENSIONS ---")
            print(ds.dims)

            print("\n--- COORDINATES ---")
            print(list(ds.coords))

            print("\n--- DATA VARIABLES (This is the most important part) ---")
            variables_found = list(ds.data_vars)
            print(variables_found)

            print("\n--- ANALYSIS ---")
            expected_vars = [
                'Temperature_Air_2m_Min_24h', 'Temperature_Air_2m_Max_24h',
                'Precipitation_Flux', 'Solar_Radiation_Flux'
            ]

            missing_vars = [v for v in expected_vars if v not in variables_found]

            if not missing_vars:
                logging.info("✅ SUCCESS: All expected weather variables were found in the file.")
            else:
                logging.error(f"❌ FAIL: The file is MISSING the following variables: {missing_vars}")

    except Exception as e:
        logging.error(f"An error occurred while trying to read the file: {e}")
        print("The file may be corrupted.")


if __name__ == "__main__":
    check_file_contents()