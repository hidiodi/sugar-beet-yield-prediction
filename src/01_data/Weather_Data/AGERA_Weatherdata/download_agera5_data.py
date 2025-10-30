import cdsapi
import logging
from pathlib import Path
import time

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_agera5_data_final(start_year=1979, end_year=2024):
    """
    Downloads daily AgERA5 data for Germany, making one request per variable/statistic per year.
    This version incorporates the mandatory 'version' parameter and correct list formatting.
    """
    logging.info(f"--- Starting FINAL AgERA5 Data Download for {start_year}-{end_year} ---")

    # Define the output directory to include the 'agera5' subdirectory
    output_directory = Path("data/01_raw/agera5") # MODIFIED LINE
    output_directory.mkdir(parents=True, exist_ok=True)

    # Define a list of download tasks based on the corrected API requirements
    download_tasks = [
        {'variable': '2m_temperature', 'statistic': '24_hour_maximum', 'filename_part': 'temp_maximum'},
        {'variable': '2m_temperature', 'statistic': '24_hour_mean', 'filename_part': 'temp_mean'},
        {'variable': '2m_temperature', 'statistic': '24_hour_minimum', 'filename_part': 'temp_minimum'},
        {'variable': 'precipitation_flux', 'statistic': None, 'filename_part': 'precipitation_flux'},
        {'variable': 'solar_radiation_flux', 'statistic': None, 'filename_part': 'solar_radiation_flux'},
        {'variable': '10m_wind_speed', 'statistic': '24_hour_mean', 'filename_part': 'wind_speed_mean'},
        {'variable': '2m_dewpoint_temperature', 'statistic': '24_hour_mean', 'filename_part': 'dewpoint_temp_mean'},
    ]

    # Initialize the CDS API client
    try:
        c = cdsapi.Client()
    except Exception as e:
        logging.error(f"Failed to initialize CDS API client. Check your .cdsapirc file: {e}")
        return

    # --- Main download loop ---
    for year in range(start_year, end_year + 1):
        for task in download_tasks:
            year_str = str(year)

            output_filename_base = f"agera5_germany_{year_str}_{task['filename_part']}"
            # The output filename will now be inside the 'agera5' directory
            output_filename = output_directory / f"{output_filename_base}.nc"
            download_zip_filename = str(output_filename) + ".zip"

            # This check will now look in the correct subdirectory
            if Path(download_zip_filename).exists() or output_filename.exists():
                logging.info(f"SKIP: Data for {output_filename_base} already exists.")
                continue

            # --- Build the request dictionary EXACTLY as required ---
            request = {
                'format': 'zip',
                'variable': [task['variable']],  # Variable as a list
                'year': [year_str],  # Year as a list
                'month': [f'{month:02d}' for month in range(1, 13)],
                'day': [f'{day:02d}' for day in range(1, 32)],
                'area': [
                    55.5, 5.5, 47.0, 15.5,  # Bounding box for Germany
                ],
                'time_aggregation': 'daily_statistics',
                'version': '2_0',  # CRITICAL: Added version parameter
            }

            # Add the statistic parameter as a list if it is required for this task
            if task['statistic']:
                request['statistic'] = [task['statistic']]  # CRITICAL: Statistic as a list

            logging.info(f"REQUESTING: {output_filename_base}...")

            try:
                c.retrieve(
                    'sis-agrometeorological-indicators',
                    request,
                    download_zip_filename
                )
                logging.info(f"SUCCESS: Downloaded {output_filename_base}.")
                # A small delay to be polite to the API server
                time.sleep(1)

            except Exception as e:
                logging.error(f"FAILED: Download for '{output_filename_base}'. Error: {e}")
                logging.error("Stopping process. You can restart the script later to resume downloading.")
                return

    logging.info("\n--- All download requests have been processed successfully! ---")
    logging.info("Next step: Unzip all the individual '.zip' files in the 'data/01_raw/agera5' directory.")


if __name__ == "__main__":
    download_agera5_data_final()