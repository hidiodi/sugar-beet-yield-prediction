# src/data/03_process_agera5_data.py
# important this will run for 5-6h+ so take your time
# 15:27:25,658 -> 20:21:20,424
import xarray as xr
import logging
from pathlib import Path
import zipfile
import tempfile
from dask.diagnostics import progress
from dask.distributed import Client, LocalCluster

# --- Setup basic logging ---
# Set logging level to DEBUG to capture detailed information if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_ds_info(ds, filename):
    """Helper function to extract and format dataset information."""
    if not ds or not isinstance(ds, xr.Dataset):
        return {"Error": "Invalid Dataset Object"}

    try:
        # Check if 'time' exists and has data before calling min/max
        time_range = "N/A"
        if 'time' in ds.coords or 'time' in ds.dims:
            try:
                t_min = ds['time'].min().item()
                t_max = ds['time'].max().item()
                time_range = f"{t_min} to {t_max}"
            except Exception:
                time_range = "Invalid Time Data"

        return {
            "File": filename,
            "Dimensions": dict(ds.dims),
            "Coordinates": list(ds.coords),
            "DataVariables": list(ds.data_vars),
            "Time_Range": time_range
        }
    except Exception as e:
        logging.error(f"Failed to extract info from {filename}: {e}", exc_info=True)
        return {"Error": f"Extraction Failed: {e}"}


def log_ds_info(log_level, header, info_dict):
    """Helper function to log structured dataset information."""
    if info_dict.get("Error"):
        logging.log(log_level, f"--- {header} FAILED ---: {info_dict['Error']}")
        return

    logging.log(log_level, f"\n--- {header} ({info_dict['File']}) STRUCTURE ---")
    # Using .sizes for consistency (as noted by the FutureWarning)
    logging.log(log_level, f"  Dimensions: {info_dict['Dimensions']}")
    logging.log(log_level, f"  Coordinates: {info_dict['Coordinates']}")
    logging.log(log_level, f"  Variables: {info_dict['DataVariables']}")
    logging.log(log_level, f"  Time Range: {info_dict['Time_Range']}")
    logging.log(log_level, "------------------------------------------------\n")


def process_and_merge_agera5_data():
    """
    Performs a fully resilient two-stage merge with extensive logging and Dask progress monitoring.
    """
    logging.info("--- Starting FULLY RESILIENT Two-Stage AgERA5 Data Processing (Heavy Logging & Dask Monitoring) ---")

    # --- Define paths ---
    raw_data_dir = Path("data/01_raw/agera5")
    intermediate_data_dir = Path("data/02_intermediate")
    consolidated_dir = intermediate_data_dir / "consolidated_agera5"

    intermediate_data_dir.mkdir(parents=True, exist_ok=True)
    consolidated_dir.mkdir(parents=True, exist_ok=True)

    output_filepath = intermediate_data_dir / "agera5_germany_merged.nc"

    if output_filepath.exists():
        logging.info(f"Final merged file '{output_filepath}' already exists. Skipping.")
        return

    # --- Stage 1: Consolidate daily files from each zip archive ---
    # (Stage 1 is unchanged as it seems to be working to create the intermediate files)
    logging.info("--- STAGE 1 of 2: Consolidating daily files (Sequentially) ---")
    zip_files = sorted(list(raw_data_dir.glob('*.zip')))

    for i, zip_path in enumerate(zip_files):
        consolidated_filename = consolidated_dir / f"{zip_path.stem}.nc"
        # Skip Stage 1 logging/processing if file already exists, for brevity
        if consolidated_filename.exists():
            continue

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                daily_nc_files = list(Path(temp_dir).glob('*.nc'))
                if not daily_nc_files:
                    continue

                with xr.open_mfdataset(daily_nc_files, combine='by_coords', parallel=False) as single_ds:

                    # CRITICAL FIX 1: Standardize coordinate/dimension order
                    standardized_ds = single_ds.transpose('time', 'lat', 'lon')

                    # CRITICAL FIX 2: Drop 'crs' variable
                    if 'crs' in standardized_ds.variables:
                        standardized_ds = standardized_ds.drop_vars('crs', errors='ignore')

                    standardized_ds.to_netcdf(consolidated_filename, engine='netcdf4')

        except Exception as e:
            logging.error(f"  -> FAILED to process {zip_path.name}. Error: {e}", exc_info=True)
            continue
    logging.info("--- Stage 1 Complete (Existing files were assumed valid) ---")

    # --- Stage 2: Validate and Merge the consolidated files ---
    logging.info("\n--- STAGE 2 of 2: Validating and Merging Consolidated Files ---")
    all_consolidated_files = sorted(list(consolidated_dir.glob('*.nc')))

    # ... (Validation Step is omitted for brevity, assuming 'valid_files' contains all 228) ...
    valid_files = []
    for f in all_consolidated_files:
        try:
            with xr.open_dataset(f) as ds:
                pass
            valid_files.append(f)
        except Exception:
            # Re-running the validation step is often unnecessary if the previous run established validity
            pass

    if len(valid_files) != len(all_consolidated_files):
        logging.warning(f"Validation found {len(valid_files)} valid files out of {len(all_consolidated_files)}.")

    if not valid_files:
        logging.error("No valid consolidated files found to merge. Stopping.")
        return

    try:
        logging.info(f"Merging {len(valid_files)} valid files...")

        # We re-introduce parallelism and explicitly set encoding for compression.
        with xr.open_mfdataset(
                valid_files,
                combine='nested',
                concat_dim='time',
                parallel=True
        ) as final_ds:

            log_ds_info(logging.INFO, "FINAL MERGED DATASET", get_ds_info(final_ds, output_filepath.name))

            logging.info("Starting final NetCDF write with Dask progress monitoring and compression...")

            # --- NEW: Define Encoding for Compression (Crucial for large files) ---
            # NetCDF compression (level 9 is max compression, 1 is fast but weak)
            comp = dict(zlib=True, complevel=3)

            # Apply the encoding to all primary data variables
            encoding = {var: comp for var in final_ds.data_vars}

            with progress.ProgressBar():
                # Pass the encoding dictionary to to_netcdf
                final_ds.to_netcdf(output_filepath, encoding=encoding, engine='netcdf4')

            logging.info(f"--- SUCCESS: Final merged data saved to '{output_filepath}'! ---")

    except Exception as e:
        # CRITICAL LOGGING: Print the full traceback for final merge failure
        logging.error(f"FAILED during final merge. Error: {e}", exc_info=True)


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=8, threads_per_worker=1, memory_limit='3.5GB')  # Adjust params as needed
    client = Client(cluster)

    # This provides a link to a dashboard where you can SEE the work happening!
    logging.info(f"Dask Dashboard is available at: {client.dashboard_link}")

    process_and_merge_agera5_data()
