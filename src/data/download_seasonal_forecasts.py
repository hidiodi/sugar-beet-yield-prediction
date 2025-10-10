import cdsapi
import logging
from pathlib import Path

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
HINDCAST_YEARS = list(range(2016, 2024))
TARGET_VARIABLES = ['2m_temperature', 'total_precipitation']
OUTPUT_DIR = Path("data/01_raw/SEAS5_hindcasts")
# The official climatology period for SEAS5 hindcasts
CLIMATOLOGY_YEARS = [str(y) for y in range(1993, 2017)]


def download_seasonal_data():
    """
    Downloads SEAS5 hindcast and climatology data.
    DEFINITIVE VERSION: Uses the correct, modern API request format based on official documentation.
    """
    logging.info(f"--- Starting SEAS5 Hindcast Download for years: {HINDCAST_YEARS} ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        c = cdsapi.Client()
    except Exception as e:
        logging.error(f"Failed to initialize CDS API client. Check your .cdsapirc file: {e}")
        return

    # Base request common to both hindcast and climatology
    base_request = {
        'originating_centre': 'ecmwf',
        'system': '5',
        'leadtime_month': ['1', '2', '3', '4', '5', '6'],  # For Apr-Sep
        'area': [55.5, 5.5, 47.0, 15.5],  # Germany
        'format': 'netcdf',
        'product_type': 'monthly_mean',
    }

    for year in HINDCAST_YEARS:
        for variable in TARGET_VARIABLES:
            # --- 1. Download the Hindcast for the specific year ---
            hindcast_filename = OUTPUT_DIR / f"seas5_hindcast_germany_{year}_{variable}.nc"
            if not hindcast_filename.exists():
                logging.info(f"REQUESTING: HINDCAST for {year}, variable: {variable}...")
                hindcast_request = base_request.copy()
                hindcast_request.update({
                    'variable': variable,
                    'year': str(year),
                    'month': '03',  # Forecasts issued in March
                })
                try:
                    c.retrieve('seasonal-monthly-single-levels', hindcast_request, str(hindcast_filename))
                    logging.info(f" -> SUCCESS: Saved to '{hindcast_filename.name}'")
                except Exception as e:
                    logging.error(f" -> FAILED to download hindcast for {year}. Error: {e}")
                    return
            else:
                logging.info(f"SKIP: Hindcast file '{hindcast_filename.name}' already exists.")

    # --- 2. Download the Climatology files (only needs to be done once) ---
    logging.info("\n--- Checking/Downloading SEAS5 Climatology (1993-2016) ---")
    for variable in TARGET_VARIABLES:
        climatology_filename = OUTPUT_DIR / f"seas5_climatology_germany_1993-2016_{variable}.nc"
        if not climatology_filename.exists():
            logging.info(f"REQUESTING: CLIMATOLOGY for variable: {variable}...")

            # --- FIX APPLIED HERE: Correct API format for climatology ---
            climatology_request = base_request.copy()
            climatology_request.update({
                'variable': variable,
                'year': CLIMATOLOGY_YEARS,  # Provide the full list of years
                'month': '03',  # Climatology is also based on the issuing month
            })
            try:
                c.retrieve('seasonal-monthly-single-levels', climatology_request, str(climatology_filename))
                logging.info(f" -> SUCCESS: Saved to '{climatology_filename.name}'")
            except Exception as e:
                logging.error(f" -> FAILED to download climatology. Error: {e}")
                return
        else:
            logging.info(f"SKIP: Climatology file '{climatology_filename.name}' already exists.")

    logging.info("\n--- SEAS5 data download process complete. ---")


if __name__ == "__main__":
    download_seasonal_data()