import cdsapi
import logging
from pathlib import Path

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION for Météo-France System 9 ---
HINDCAST_YEARS = list(range(1993, 2017))
FORECAST_YEARS = list(range(2017, 2026))

TARGET_VARIABLES = [
    '2m_temperature',
    'total_precipitation',
    'surface_solar_radiation'
]

OUTPUT_DIR = Path("data/01_raw/MF_Sys9_monthly_germany")

def download_mf_system9_data_final():
    """
    Downloads the complete Météo-France System 9 dataset.
    - Hindcasts are downloaded year-by-year for robustness.
    - Forecasts are downloaded year-by-year.
    """
    logging.info("--- Starting Météo-France System 9 Data Download (Final Looping Method) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        c = cdsapi.Client()
    except Exception as e:
        logging.error(f"Failed to initialize CDS API client. Check your .cdsapirc file: {e}")
        return

    # --- PART 1: Download the Hindcast Dataset Year by Year (1993-2016) ---
    logging.info("\n--- STEP 1: Requesting hindcast dataset year by year for March start ---")

    for year in HINDCAST_YEARS:
        hindcast_filename = OUTPUT_DIR / f"mf_sys9_monthly_germany_HINDCAST_{year}_march_start.nc"

        if hindcast_filename.exists():
            logging.info(f"SKIP: Hindcast for {year} ('{hindcast_filename.name}') already exists.")
            continue

        hindcast_request = {
    "originating_centre": "meteo_france",
    "system": "9",
    "variable": [
        "2m_temperature",
        "surface_solar_radiation",
        "total_precipitation"
    ],
    "product_type": [
        "ensemble_mean",
        "hindcast_climate_mean",
        "monthly_mean",
        "monthly_minimum",
        "monthly_maximum",
        "monthly_standard_deviation"
    ],
    "year": [
        "1993", "1994", "1995",
        "1996", "1997", "1998",
        "1999", "2000", "2001",
        "2002", "2003", "2004",
        "2005", "2006", "2007",
        "2008", "2009", "2010",
        "2011", "2012", "2013",
        "2014", "2015", "2016",
        "2017", "2018", "2019",
        "2020", "2021", "2022",
        "2023", "2024", "2025"
    ],
    "month": [
        "04", "05", "06",
        "07", "08", "09",
        "10", "11"
    ],
    "leadtime_month": ["3"],
    "data_format": "grib"
}
        try:
            logging.info(f"Requesting hindcast for year {year}...")
            c.retrieve('seasonal-monthly-single-levels', hindcast_request, str(hindcast_filename))
            logging.info(f" -> SUCCESS: Hindcast for {year} saved to '{hindcast_filename.name}'")
        except Exception as e:
            logging.error(f" -> FAILED to download hindcast for {year}. Error: {e}")
            continue # Continue to the next year even if one fails

    # --- PART 2: Download the Forecast Years (2017-2025) ---
    logging.info("\n--- STEP 2: Requesting individual forecast years for March start ---")
    for year in FORECAST_YEARS:
        forecast_filename = OUTPUT_DIR / f"mf_sys9_monthly_germany_FORECAST_{year}_march_start.nc"

        if forecast_filename.exists():
            logging.info(f"SKIP: Forecast for {year} already exists.")
            continue

        forecast_request = {
            'originating_centre': 'meteo_france',
            'system': '9',
            'variable': TARGET_VARIABLES,
            'product_type': 'monthly_mean',
            'year': str(year),
            'month': '03',
            'leadtime_month': ['1', '2', '3', '4', '5', '6', '7'],
            'area': [55.5, 5.5, 47.0, 15.5],
            'format': 'netcdf',
        }
        try:
            logging.info(f"Requesting forecast for {year}...")
            c.retrieve('seasonal-monthly-single-levels', forecast_request, str(forecast_filename))
            logging.info(f" -> SUCCESS: Saved forecast for {year} to '{forecast_filename.name}'")
        except Exception as e:
            logging.error(f" -> FAILED to download forecast for {year}. Error: {e}")
            continue

    logging.info("\n--- Météo-France System 9 data download check complete. ---")


if __name__ == "__main__":
    download_mf_system9_data_final()