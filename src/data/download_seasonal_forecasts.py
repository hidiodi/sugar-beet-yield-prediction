import cdsapi
import logging
from pathlib import Path

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
HINDCAST_YEARS = list(range(1981, 2017))
FORECAST_YEARS = list(range(2017, 2025))
TARGET_VARIABLES = ['2m_temperature', 'total_precipitation']
OUTPUT_DIR = Path("data/01_raw/SEAS5_monthly_germany")


def download_yearly_data():
    """
    Downloads the SEAS5 monthly hindcast/forecasts for each year.
    This is the only download step required. The climatology will be calculated
    manually from these downloaded hindcast files in the next processing step.
    """
    logging.info("--- Starting SEAS5 Monthly Hindcast/Forecast Download ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        c = cdsapi.Client()
    except Exception as e:
        logging.error(f"Failed to initialize CDS API client. Check your .cdsapirc file: {e}")
        return

    all_years = HINDCAST_YEARS + FORECAST_YEARS

    for year in all_years:
        output_filename = OUTPUT_DIR / f"seas5_monthly_germany_{year}_march_start.nc"

        if output_filename.exists():
            logging.info(f"SKIP: File '{output_filename.name}' already exists.")
            continue

        request = {
            'originating_centre': 'ecmwf',
            'system': '5',
            'variable': TARGET_VARIABLES,
            'product_type': 'monthly_mean',
            'year': str(year),
            'month': '03',
            'leadtime_month': ['1', '2', '3', '4', '5', '6', '7'],
            'area': [55.5, 5.5, 47.0, 15.5],
            'format': 'netcdf',
        }

        period_type = 'hindcast' if year in HINDCAST_YEARS else 'forecast'

        if period_type == 'hindcast':
            logging.info(f"REQUESTING: MONTHLY HINDCAST for {year} (25 members)...")
            request['ensemble_member'] = [str(i) for i in range(1, 26)]
        else: # forecast
            logging.info(f"REQUESTING: MONTHLY FORECAST for {year} (51 members)...")

        try:
            c.retrieve('seasonal-monthly-single-levels', request, str(output_filename))
            logging.info(f" -> SUCCESS: Saved to '{output_filename.name}'")
        except Exception as e:
            logging.error(f" -> FAILED to download data for {year}. Error: {e}")
            continue

    logging.info("\n--- All available yearly SEAS5 data has been checked/downloaded. ---")


if __name__ == "__main__":
    download_yearly_data()