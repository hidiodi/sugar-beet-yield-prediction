"""
Configuration file for the hybrid model pipeline.

This file centralizes all configurable parameters for the pipeline,
including file paths, model settings, and execution flags.
"""
from pathlib import Path

# --- Project Structure ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"

# --- Pipeline Configuration ---
PIPELINE_NAME = "Main Hybrid Model Pipeline"

SCRIPTS_TO_RUN = [
    "src/01_data/download_all_data_pipeline.py",
    "src/01_data/process_input_data_pipeline.py",
    "src/02_models/Wofost7.1/04_create_daily_weather_file.py",
    "src/02_models/Wofost7.1/run_wofost_pipeline.py",
    "src/02_models/Wofost7.1/apply_detrending_correction.py",
    "src/01_data/FeatureEngineering/build_stage1_features.py",
    "src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py",
    "src/02_models/XGBoost/regression_model/Testing/backtest_final_quantile_model.py",
    "src/02_models/FinalEnsemble/backtest_final_ensemble.py",
    "src/03_analysis/basic_analysis/compare_model_versions.py",
    "src/03_analysis/shap_analysis_xgb.py",
    "src/03_analysis/run_hybrid_analysis_pipeline.py",
]

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
import datetime

WEATHER_START_YEAR = 1981
WEATHER_END_YEAR = 2024

# --- WOFOST Model Configuration ---
WOFOST_CONFIG = {
    # max range 1982 - 2024 and None to run all districts
    'START_YEAR': 1982, 'END_YEAR': 2024, 'DISTRICT_LIMIT': None,
    'FILE_PATHS': {
        'HISTORICAL_DAILY_WEATHER_DIR': DATA_DIR / '02_intermediate/daily_weather',
        'YIELD_DATA': DATA_DIR / '02_intermediate/sugarbeet_yield.csv',
        'STATIC_SOIL_FEATURES': DATA_DIR / '03_processed/static_features_districts.csv',
        'SEAS5_MEMBER_FEATURES': DATA_DIR / '02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv',
        'CROP_YAML': DATA_DIR / '01_raw/sugarbeet.yaml',
        'OUTPUT_DIR': DATA_DIR / '06_model_output/multi_year_final',
        'EXTREME_WEATHER_METRICS_OUTPUT': DATA_DIR / '06_model_output/ensemble_extreme_weather_metrics.csv'
    },
    'WEATHER_DEFAULTS': {'WIND_SPEED': 2.0, 'VAPOR_PRESSURE': 1.0},
    'WEATHER_GENERATOR': {'PRECIP_THRESHOLD_MM': 0.3, 'MIN_SRAD': 1.0},
    'AGROMANAGEMENT': {
        'CROP_START_DATE': datetime.date(2018, 3, 22), 'CROP_END_DATE': datetime.date(2018, 11, 15),
        'MAX_DURATION': 250,
    },
    'CONSTANTS': {
        'DMC_SUGARBEET': 0.25, 'INITIAL_ROOTING_DEPTH_CM': 10.0, 'SOIL_PARTICLE_DENSITY': 2.65,
    },
    'SOIL_COLUMN_MAPPING': {
        'sand': 'avg_sand_0_100cm', 'clay': 'avg_clay_0_100cm', 'som': 'avg_som_0_100cm', 'bdod': 'avg_bdod_0_100cm',
    },
    'SOIL_DEFAULTS_AND_CONSTANTS': {
        'RDMSOL': 150.0, 'KSUB': 10.0, 'SOPE': 10.0
    },
    'GENERIC_SITE': {'LATITUDE': 52.0, 'LONGITUDE': 10.0, 'ELEVATION': 50.0},
    'ANALOG_YEAR_CONFIG': {'NUM_ANALOGS': 5, 'MIN_YEARS_FOR_FIT': 10, }
}

# --- Detrending Configuration ---
DETRENDING_CONFIG = {
    'FILE_PATHS': {
        'INPUT_YIELD_CSV': DATA_DIR / '02_intermediate/sugarbeet_yield.csv',
        'OUTPUT_DIR': DATA_DIR / '05_model_input/wofost_walkforward',
    },
    'ARIMA_ORDER': (1, 0, 0),
    'GAM_SPLINES': 10,
    'MIN_TRAIN_SIZE': 10
}

# --- Feature Engineering Configuration ---
FEATURE_ENGINEERING_CONFIG = {
    'FILE_PATHS': {
        'MASTER_DATASET': DATA_DIR / '04_master/master_dataset.csv',
        'PRODUCER_PRICE_CSV': DATA_DIR / '01_raw/Bundesdatenbank/61211-0001_de.csv',
        'INPUT_PRICE_CSV': DATA_DIR / '01_raw/Bundesdatenbank/61221-0003_de.csv',
        'SATELLITE_FEATURES_CSV': DATA_DIR / '03_processed/satellite_features_districts_2001-2024.csv',
        'GEOJSON_DISTRICTS': DATA_DIR / '01_raw/districts_official.geojson',
        'DAILY_WEATHER_DIR': DATA_DIR / '02_intermediate/daily_weather',
        'WALKFORWARD_FORECAST_CSV': DATA_DIR / '05_model_input/wofost_walkforward/final_honest_forecasts.csv',
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

# --- XGBoost Model Training Configuration ---
XGBOOST_TRAINING_CONFIG = {
    'DATA_PATH': DATA_DIR / '05_model_input/stage1_preseason_features.csv',
    'MODEL_OUTPUT_DIR': BASE_DIR / 'src/models',
    'FEATURE_COLS': [
        # --- Original SEAS5 Weather Anomaly Features (Antecedent & Seasonal) ---
        'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
        'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
        'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
        'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
        'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
        'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast',

        # --- Original SEAS5 Weather Probability Features ---
        'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast', 'summer_temp_prob_warm_forecast',
        'summer_precip_prob_wet_forecast',

        # --- Static Geographic & Soil Features ---
        'lat', 'lon', 'avg_elevation', 'avg_slope', 'avg_bdod_0_30cm', 'avg_clay_0_30cm',
        'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm',

        # --- Satellite Features (Early Season Condition) ---
        'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean',
        'winter_cropland_LST_anomaly', 'winter_cropland_snow_cover_days',

        # --- Teleconnection Indices ---
        'nao_winter_avg', 'sca_winter_avg', 'enso_mei_winter_avg',

        # --- Lagged Economic Features & Anomalies ---
        'profit_margin_proxy_lag1', 'cost_of_inputs_lag1', 'producer_price_index_lag1_anomaly',
        'seed_price_index_lag1_anomaly', 'energy_price_index_lag1_anomaly',
        #'fertilizer_price_index_lag1_anomaly',
        'plant_protection_price_index_lag1_anomaly',
        'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',

        # --- Stage 1 Model & Hybrid Features ---
        'stage1_forecast',  # Note: This is the column name from the file, used as 'stage1_forecast'
        'wofost_forecast_x_profit_margin',
        'has_wofost_data',

        # --- General Regional & Temporal Features ---
        'state_encoded',
        'year_trend',

        # --- Original Interaction & Polynomial Features ---
        'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip', 'summer_heat_x_profit_margin',
        'summer_precip_x_input_costs',
        'hot_dry_interaction',
        'lat_x_summer_temp', 'sandy_soil_x_drought',
        'antecedent_gdd_sum_anomaly_sq', 'spring_temp_prob_warm_forecast_sq',
        'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq',
        'summer_precip_prob_wet_forecast_sq', 'summer_precip_anomaly_forecast_sq',

        # --- NEW Physiologically-Grounded Features for Extremes ---
        'CASDI_Phase2_Count',  # Compounded Abiotic Stress (Heat & Drought)
        'NMSD_Phase2_Count',  # Nighttime Metabolic Stress Days
        'OSAW_Phase2_Count',  # Optimal Sugar Accumulation Window
        'ECES_Phase1_Cumulative',  # Early Canopy Establishment Stress
        'summer_days_tmax_gt_30c'  # Retained as a simple, direct measure of heat
    ],
    'BEST_PARAMS': {
        'n_estimators': 914, 'learning_rate': 0.026114, 'max_depth': 5,
        'subsample': 0.922850, 'colsample_bytree': 0.811573, 'gamma': 1.830853,
        'min_child_weight': 2, 'random_state': 42, 'n_jobs': -1
    },
'QUANTILES': {'lower': 0.025, 'median': 0.5, 'upper': 0.975},
    'LOWER_MODEL_PATH': BASE_DIR / 'src/models/final_quantile_model_lower.joblib',
    'MEDIAN_MODEL_PATH': BASE_DIR / 'src/models/final_quantile_model_median.joblib',
    'UPPER_MODEL_PATH': BASE_DIR / 'src/models/final_quantile_model_upper.joblib'
}

# --- Backtesting Configuration ---
BACKTESTING_CONFIG = {
    'GEOJSON_PATH': DATA_DIR / '01_raw/districts_official.geojson',
    'REPORT_DIR': BASE_DIR / 'reports/figures/district_level_diagnostics/final_quantile_champion',
    'BACKTEST_START_YEAR': 2000,
    'BACKTEST_END_YEAR': 2024,
    'LOW_DATA_THRESHOLD': 10,
    'MIN_DATAPOINTS_FOR_PLOT': 10,
    'CALIBRATION_SET_SIZE': 0.15,
    'NOMINAL_COVERAGE': 0.95
}

# --- Ensemble Backtesting Configuration ---
ENSEMBLE_BACKTESTING_CONFIG = {
    'HYBRID_XGB_INPUT_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv',
    'ADAPTIVE_CQR_INPUT_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/adaptive_cqr_champion/full_backtest_predictions.csv',
    'GEOJSON_PATH': DATA_DIR / '01_raw/districts_official.geojson',
    'REPORT_DIR': BASE_DIR / 'reports/figures/district_level_diagnostics/final_ensemble_champion',
    'LOW_DATA_THRESHOLD': 10,
    'MIN_DATAPOINTS_FOR_PLOT': 10,
    'NOMINAL_COVERAGE': 0.95
}

# --- Model Comparison Configuration ---
MODEL_COMPARISON_CONFIG = {
    'NOMINAL_COVERAGE_PERCENT': 95.0,
    'FINAL_ENSEMBLE_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/final_ensemble_champion/full_backtest_predictions.csv',
    'NGBOOST_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/final_ngboost_champion/full_backtest_predictions.csv',
    'ADAPTIVE_CQR_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/adaptive_cqr_champion/full_backtest_predictions.csv',
    'HYBRID_XGB_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv',
    'OUTPUT_DIR': BASE_DIR / 'reports/figures/final_model_comparison'
}

# --- SHAP Analysis Configuration ---
SHAP_ANALYSIS_CONFIG = {
    'SHAP_OUTPUT_DIR': BASE_DIR / 'reports/shap_analysis',
    'SHAP_SAMPLE_SIZE': 5000
}

# --- Analysis Pipeline Configuration ---
ANALYSIS_PIPELINE_NAME = "Analysis Hybrid Model Pipeline"
ANALYSIS_SCRIPTS_TO_RUN = [
    "src/03_analysis/hybrid_model_analysis/analyze_input_features.py",
    "src/03_analysis/hybrid_model_analysis/analyze_wofost_pipeline.py",
    "src/03_analysis/hybrid_model_analysis/analyze_hybrid_model.py",
]


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
