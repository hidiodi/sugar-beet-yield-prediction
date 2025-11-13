import cdsapi
import logging
from pathlib import Path
import time

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
# This is the OFFICIAL, CORRECT name for the monthly averaged dataset.
# See: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means
DATASET_NAME = 'reanalysis-era5-land-monthly-means'
START_YEAR = 1981
END_YEAR = 2024
OUTPUT_DIRECTORY = Path("data/01_raw/era5_land_monthly_soil")

def download_era5_land_february_soil_moisture(start_year=START_YEAR, end_year=END_YEAR):
    """
    Downloads MONTHLY AVERAGED ERA5-Land soil moisture (Layer 1) for FEBRUARY ONLY
    for all specified years. This serves as the initial condition for the March 1st forecast.
    """
    logging.info(f"--- Starting ERA5-Land FEBRUARY Soil Moisture Download for {start_year}-{end_year} ---")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    try:
        c = cdsapi.Client()
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize CDS API client. Check your .cdsapirc file: {e}")
        return

    for year in range(start_year, end_year + 1):
        year_str = str(year)
        # We will name the file clearly to indicate it's February data.
        output_filename = OUTPUT_DIRECTORY / f"era5_land_soil_moisture_germany_{year_str}_FEBRUARY.nc"

        if output_filename.exists():
            logging.info(f"SKIP: Data for {year_str} February already exists.")
            continue

        request = {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'variable': 'volumetric_soil_water_layer_1',
            'year': year_str,
            'month': '02',  # FEBRUARY ONLY. This corrects my previous error.
            'time': '00:00',
            'area': [55.5, 5.5, 47.0, 15.5],  # Germany
        }

        logging.info(f"REQUESTING: February soil moisture for {year_str}...")

        try:
            c.retrieve(
                DATASET_NAME,
                request,
                str(output_filename)
            )
            logging.info(f"SUCCESS: Downloaded {output_filename.name}.")
            time.sleep(2)

        except Exception as e:
            logging.error(f"FAILED: Download for year '{year_str}'. Error: {e}")
            logging.error("Stopping process. Check API key and dataset name spelling.")
            return

    logging.info("\n--- All ERA5-Land February soil moisture download requests processed. ---")


if __name__ == "__main__":
    download_era5_land_february_soil_moisture()