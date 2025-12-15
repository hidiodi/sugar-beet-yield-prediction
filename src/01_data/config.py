from pathlib import Path
from src.config import BASE_DIR, DATA_DIR, RAW_DATA_DIR, INTERMEDIATE_DATA_DIR, PROCESSED_DATA_DIR

DOWNLOAD_PIPELINE_NAME = "Main Data Downloading Pipeline"
DOWNLOAD_SCRIPTS_TO_RUN = [
    "src/01_data/StaticSoil_data/build_static_features.py",
    "src/01_data/WinterSatellite_data/build_satellite_features.py",
    "src/01_data/Weather_Data/SEASForecastData/ECMWF51/download_ECMWF51_forecast.py",
    "src/01_data/Weather_Data/AGERA_Weatherdata/download_agera5_data.py"
]

PROCESS_PIPELINE_NAME = "Main Data Processing Pipeline"
PROCESS_SCRIPTS_TO_RUN = [
    "src/01_data/Weather_Data/AGERA_Weatherdata/process_agera5_data.py",
    "src/01_data/Sugarbeetdata/process_agronomic_data.py",
    "src/01_data/Weather_Data/SEASForecastData/ECMWF51/build_forecast_features_by_member.py",
    "src/01_data/Weather_Data/04_create_crop_dataset_with_weather.py",
    "src/02_models/Wofost/04_create_daily_weather_file.py",
]

# --- Weather Data Configuration ---
AGERA5_CONSOLIDATED_DIR = DATA_DIR / "02_intermediate/consolidated_agera5"
DAILY_WEATHER_DIR = DATA_DIR / "02_intermediate/daily_weather"
WEATHER_VARIABLE_MAP = {
    'tmin': ('temp_minimum', 'Temperature_Air_2m_Min_24h'),
    'tmax': ('temp_maximum', 'Temperature_Air_2m_Max_24h'),
    'precip': ('precipitation_flux', 'Precipitation_Flux'),
    'srad': ('solar_radiation_flux', 'Solar_Radiation_Flux'),
    'wind': ('wind_speed_mean', 'Wind_Speed_10m_Mean_24h'),
    'vap': ('dewpoint_temp_mean', 'Dew_Point_Temperature_2m_Mean_24h'),
}

# Path to the raw SEAS5 NetCDF files
SEAS5_RAW_DIR = BASE_DIR / "data" / "01_raw" / "ECMWF51_monthly_germany"
# Path to the output of the V3 (zonal stats) daily weather script
DAILY_WEATHER_DIR_V3 = BASE_DIR / "data" / "04_feature" / "weather_district_daily"

WEATHER_START_YEAR = 1981
WEATHER_END_YEAR = 2024

# --- Feature Engineering Configuration ---
FEATURE_ENGINEERING_CONFIG = {
    'FILE_PATHS': {
        'MASTER_DATASET': DATA_DIR / '04_master/master_dataset.csv',
        'PRODUCER_PRICE_CSV': DATA_DIR / '01_raw/Bundesdatenbank/61211-0001_de.csv',
        'INPUT_PRICE_CSV': DATA_DIR / '01_raw/Bundesdatenbank/61221-0003_de.csv',
        'SATELLITE_FEATURES_CSV': DATA_DIR / '03_processed/satellite_features_districts_2001-2024.csv',
        'GEOJSON_DISTRICTS': DATA_DIR / '01_raw/districts_official.geojson',
        'ECMWF_FORECAST_FEATURES_CSV': DATA_DIR / '02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv',
        'DAILY_WEATHER_DIR': DATA_DIR / '02_intermediate/daily_weather', #we cant use this, this is not know in March!!!
        'WALKFORWARD_FORECAST_CSV': DATA_DIR / '05_model_input/wofost_walkforward/final_honest_forecasts.csv', #technical trend model
        'WOFOST_ENSEMBLE_CSV': DATA_DIR / '06_model_output/multi_year_final/forecast_ensemble_results_raw.csv', #actual wofost yield output
        'WOFOST_METRICS_CSV': DATA_DIR / '06_model_output/multi_year_final/forecast_extreme_weather_metrics.csv', #actual wofost weather output
        'WOFOST_INITIAL_CONDITIONS': DATA_DIR / '03_processed/InitialConditions.csv',
        'WOFOST_STATIC_SITE_DATA': DATA_DIR / '03_processed/StaticSiteData.csv',

        'OUTPUT_DIR': DATA_DIR / '05_model_input/',
        'OUTPUT_FILE': DATA_DIR / '05_model_input/stage1_preseason_features.csv'
    },
    'WEATHER_FEATURE_YEAR_START': 1981,
    'WEATHER_FEATURE_YEAR_END': 2024,
    'PHYSIOLOGY_PARAMS': {
        'TMAX_STRESS_THRESHOLD': 30.0,
        'TMIN_STRESS_THRESHOLD': 17.0,
        'TMAX_OPTIMAL_MIN': 17.0,
        'TMAX_OPTIMAL_MAX': 25.0,
        'TMIN_OPTIMAL_MAX': 15.0,
        'PRECIP_DEFICIT_WINDOW': 30,
        'PRECIP_DEFICIT_THRESHOLD': 20.0,
        'ECES_EXPONENT': 1.5,
        'DTR_SUNNY_DAY_QUANTILE': 0.75
    }
}

# --- GEE Configuration ---
GEE_PROJECT_ID = 'augmented-audio-471809-h3'
GEE_HIGH_VOLUME_ENDPOINT = 'https://earthengine-highvolume.googleapis.com'
DISTRICTS_GEOJSON_PATH = RAW_DATA_DIR / "districts_official.geojson"
STATIC_FEATURES_OUTPUT_PATH = PROCESSED_DATA_DIR / "static_features_districts.csv"
FARMLAND_MASK_ASSET_ID = 'projects/augmented-audio-471809-h3/assets/farmland_mask_germany'
DEM_IMAGE = 'USGS/SRTMGL1_003'
SOIL_PROPERTIES = ['bdod', 'clay', 'sand', 'soc', 'phh2o']
SOIL_DEPTHS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
LAYER_THICKNESS = {'0_5cm': 5, '5_15cm': 10, '15_30cm': 15, '30_60cm': 30, '60_100cm': 40}
TOPSOIL_LAYERS = ['0_5cm', '5_15cm', '15_30cm']
ROOTZONE_LAYERS = ['0_5cm', '5_15cm', '15_30cm', '30_60cm', '60_100cm']

FINAL_FEATURE_COLUMNS = [
    'district_no', 'avg_elevation', 'avg_slope',
    'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm',
    'avg_bdod_0_100cm', 'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm', 'avg_phh2o_0_100cm'
]
