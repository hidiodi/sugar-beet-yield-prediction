# src/data/03_process_agera5_data.py

import xarray as xr
import logging
from pathlib import Path
import zipfile
import tempfile
import shutil

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_and_merge_agera5_data():
    """
    Performs a fully resilient two-stage merge.
    Stage 1: Consolidates daily files sequentially to prevent file locking errors.
    Stage 2: Validates each consolidated file before performing the final parallel merge.
    """
    logging.info("--- Starting FULLY RESILIENT Two-Stage AgERA5 Data Processing ---")

    # --- Define paths ---
    raw_data_dir = Path("data/01_raw/agera5")
    intermediate_data_dir = Path("data/02_intermediate")
    consolidated_dir = intermediate_data_dir / "consolidated_agera5"

    intermediate_data_dir.mkdir(parents=True, exist_ok=True)
    consolidated_dir.mkdir(parents=True, exist_ok=True)

    output_filepath = intermediate_data_dir / "agera5_germany_2017_2024_merged.nc"

    if output_filepath.exists():
        logging.info(f"Final merged file '{output_filepath}' already exists. Skipping.")
        return

    # --- Stage 1: Consolidate daily files from each zip archive ---
    logging.info("--- STAGE 1 of 2: Consolidating daily files (Sequentially) ---")
    zip_files = sorted(list(raw_data_dir.glob('*.zip')))

    for i, zip_path in enumerate(zip_files):
        consolidated_filename = consolidated_dir / zip_path.stem
        logging.info(f"Processing archive {i + 1}/{len(zip_files)}: {zip_path.name}")

        if consolidated_filename.exists():
            logging.info(f"  -> Consolidated file '{consolidated_filename.name}' exists. Skipping.")
            continue

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                daily_nc_files = list(Path(temp_dir).glob('*.nc'))
                if not daily_nc_files:
                    logging.warning(f"  -> No .nc files in {zip_path.name}. Skipping.")
                    continue

                with xr.open_mfdataset(daily_nc_files, combine='by_coords', parallel=False) as single_ds:
                    single_ds.to_netcdf(consolidated_filename, engine='netcdf4')
                logging.info(f"  -> Successfully created: {consolidated_filename.name}")

        except Exception as e:
            logging.error(f"  -> FAILED to process {zip_path.name}. Error: {e}")
            continue

    # --- Stage 2: Validate and Merge the consolidated files ---
    logging.info("\n--- STAGE 2 of 2: Validating and Merging Consolidated Files ---")
    all_consolidated_files = sorted(list(consolidated_dir.glob('*.nc')))

    # --- VALIDATION STEP ---
    logging.info(f"Validating {len(all_consolidated_files)} consolidated files before final merge...")
    valid_files = []
    for f in all_consolidated_files:
        try:
            with xr.open_dataset(f) as ds:
                pass  # Try to open the file to check its integrity
            valid_files.append(f)
        except Exception as e:
            logging.warning(f"  -> CORRUPT FILE DETECTED. Skipping: {f.name}. Reason: {e}")

    if not valid_files:
        logging.error("No valid consolidated files found to merge. Stopping.")
        return

    try:
        logging.info(f"Merging {len(valid_files)} valid files...")
        with xr.open_mfdataset(valid_files, combine='by_coords', parallel=True) as final_ds:
            final_ds.to_netcdf(output_filepath, engine='netcdf4')
        logging.info(f"--- SUCCESS: Final merged data saved to '{output_filepath}'! ---")

    except Exception as e:
        logging.error(f"FAILED during final merge. Error: {e}")


if __name__ == "__main__":
    process_and_merge_agera5_data()