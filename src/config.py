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
    #"src/01_data/download_all_data_pipeline.py",
    #"src/01_data/process_input_data_pipeline.py",
    #"src/02_models/Wofost7.1/04_create_daily_weather_file.py",
    #"src/02_models/Wofost7.1/run_wofost_pipeline.py",
    #"src/02_models/Wofost7.1/apply_detrending_correction.py",
    #"src/01_data/FeatureEngineering/build_stage1_features.py",
    #"src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py", # trains on residual of wofost
    #"src/02_models/XGBoost/regression_model/Testing/backtest_final_quantile_model.py",  # trains on residual of wofost
    "src/02_models/XGBoost/regression_model/ModelScripts/train_standalone_xgb_model.py", # uses wofost as a simple input and trains with detrended yield as target
    "src/02_models/XGBoost/regression_model/Testing/backtest_standalone_xgb_model.py", # uses wofost as a simple input and trains with detrended yield as target
    #"src/02_models/NGboost/train_final_ngboost_model.py",
    #"src/02_models/NGboost/backtest_final_ngboost_model.py",
    #"src/02_models/FinalEnsemble/backtest_final_ensemble.py",
    "src/03_analysis/basic_analysis/compare_model_versions.py",
    #"src/02_models/XGBoost/regression_model/Tuning/tune_quantiles.py",
    #"src/03_analysis/shap_analysis_xgb.py",
    #"src/03_analysis/run_hybrid_analysis_pipeline.py",
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
        'ECMWF_FORECAST_FEATURES_CSV': DATA_DIR / '02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv',
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
# --- XGBoost Model Training Configuration ---
# --- XGBoost Model Training Configuration ---
XGBOOST_TRAINING_CONFIG = {
    'DATA_PATH': DATA_DIR / '05_model_input/stage1_preseason_features.csv',
    'MODEL_OUTPUT_DIR': BASE_DIR / 'src/models',
    'FEATURE_COLS': [
        # --- Trend & Technology Proxy ---
        'national_avg_yield_lag1',

        # --- Antecedent Weather Features (Observed by March) ---
        'antecedent_precip_sum',
        'antecedent_frost_days',
        'antecedent_heavy_precip_days',
        'antecedent_gdd_sum_anomaly',

        # --- Seasonal Forecast - Central Tendency (Ensemble Mean) ---
        'spring_temp_anomaly_forecast_mean',
        'spring_precip_anomaly_forecast_mean',
        'summer_temp_anomaly_forecast_mean',
        'summer_precip_anomaly_forecast_mean',

        # --- Seasonal Forecast - Uncertainty & Spread (Ensemble Std Dev) ---
        'spring_temp_anomaly_forecast_std',
        'summer_temp_anomaly_forecast_std',
        'spring_precip_anomaly_forecast_std',
        'summer_precip_anomaly_forecast_std',

        # --- Seasonal Forecast - Tail Risk (Ensemble Quantiles) ---
        'spring_temp_anomaly_forecast_p10',
        'spring_temp_anomaly_forecast_p90',
        'summer_temp_anomaly_forecast_p10',
        'summer_temp_anomaly_forecast_p90',
        'summer_precip_anomaly_forecast_p10',
        'summer_precip_anomaly_forecast_p90',

        # --- Monthly Summer Temperature Risk ---
        'june_temp_anomaly_p10',
        'july_temp_anomaly_p10',
        'august_temp_anomaly_p10',

        # --- Solar Radiation Forecast Features ---
        'summer_solar_rad_anomaly_forecast_mean',
        'summer_solar_rad_anomaly_forecast_std',
        'summer_solar_rad_anomaly_forecast_p10',
        'summer_solar_rad_anomaly_forecast_p90',

        # --- Seasonal Forecast - Probabilistic Risk Features ---
        'prob_hot_summer',
        'prob_dry_summer',

        # --- Static Geographic & Soil Features ---
        'lat', 'lon', 'avg_elevation', 'avg_slope',
        'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm',
        'avg_som_0_30cm', 'avg_phh2o_0_30cm',

        # --- Satellite Features ---
        'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
        'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days',

        # --- Teleconnection Indices ---
        'nao_winter_avg', 'sca_winter_avg', 'enso_mei_winter_avg',

        # --- Lagged Economic Features & Anomalies ---
        'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
        'producer_price_index_lag1_anomaly', 'seed_price_index_lag1_anomaly',
        'energy_price_index_lag1_anomaly', 'plant_protection_price_index_lag1_anomaly',
        'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',

        # --- WOFOST-Related Hybrid Features (but NOT the forecast itself) ---
        'wofost_forecast_x_profit_margin',
        'has_wofost_data',

        # --- General Regional & Temporal Features ---
        'state_encoded',
        'year_trend',

        # --- Interaction & Risk Features ---
        'gdd_x_fertilizer_price',
        'hot_dry_interaction',
        'forecast_hot_dry_risk_p90',
        'profit_margin_proxy_lag1_x_spring_temp_anomaly_mean',
        'antecedent_precip_sum_x_spring_temp_anomaly_mean',

        'summer_temp_forecast_range',
        'summer_precip_forecast_range',
        'enso_x_spring_temp_forecast',
        'sca_x_spring_temp_forecast',
    ],
    'BEST_PARAMS_LOWER': {
        'n_estimators': 1397,
        'learning_rate': 0.070545,
        'max_depth': 9,
        'subsample': 0.612895,
        'colsample_bytree': 0.957773,
        'gamma': 2.371925,
        'min_child_weight': 7,
        'random_state': 42,
        'n_jobs': -1
    },
    'BEST_PARAMS_MEDIAN': {
        'n_estimators': 1433,
        'learning_rate': 0.055466,
        'max_depth': 5,
        'subsample': 0.954196,
        'colsample_bytree': 0.854452,
        'gamma': 6.528400,
        'min_child_weight': 9,
        'random_state': 42,
        'n_jobs': -1
    },
    'BEST_PARAMS_UPPER': {
        'n_estimators': 1374,
        'learning_rate': 0.098116,
        'max_depth': 9,
        'subsample': 0.804305,
        'colsample_bytree': 0.875319,
        'gamma': 10.142525,
        'min_child_weight': 6,
        'random_state': 42,
        'n_jobs': -1
    },
    'QUANTILES': {'lower': 0.025, 'median': 0.5, 'upper': 0.975},
    'LOWER_MODEL_PATH': BASE_DIR / 'src/models/final_quantile_model_lower.joblib',
    'MEDIAN_MODEL_PATH': BASE_DIR / 'src/models/final_quantile_model_median.joblib',
    'UPPER_MODEL_PATH': BASE_DIR / 'src/models/final_quantile_model_upper.joblib'
}
# --- XGBoost Model Tuning Configuration ---
XGBOOST_TUNING_CONFIG = {
    'N_TRIALS_PER_MODEL': 75,  # Number of trials for EACH quantile model
    'STUDY_NAMES': {
        'lower': "xgb_yield_LOWER_tuning_v7",
        'median': "xgb_yield_MEDIAN_tuning_v7",
        'upper': "xgb_yield_UPPER_tuning_v7"
    },
    'VALIDATION_START_YEAR': 2007,
    'VALIDATION_END_YEAR': 2014,
    'STORAGE_DB_NAME': "xgb_yield_separate_tuning_v7.db" # A single DB can hold multiple studies
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

STANDALONE_XGB_CONFIG = {
    # UPDATED: Point back to the ORIGINAL source file.
    'DATA_PATH': DATA_DIR / '05_model_input/stage1_preseason_features.csv',
    'MODEL_OUTPUT_DIR': BASE_DIR / 'src/models/standalone_xgb',

    'FEATURE_COLS': XGBOOST_TRAINING_CONFIG['FEATURE_COLS'] + [
        'wofost_forecast_yield_fresh_dt'
    ],

    # The script will CREATE and use 'yield_detrended' as the target
    'TARGET_COL': 'yield_detrended',

    'BEST_PARAMS_LOWER': XGBOOST_TRAINING_CONFIG['BEST_PARAMS_LOWER'],
    'BEST_PARAMS_MEDIAN': XGBOOST_TRAINING_CONFIG['BEST_PARAMS_MEDIAN'],
    'BEST_PARAMS_UPPER': XGBOOST_TRAINING_CONFIG['BEST_PARAMS_UPPER'],

    'QUANTILES': {'lower': 0.025, 'median': 0.5, 'upper': 0.975},

    'LOWER_MODEL_PATH': BASE_DIR / 'src/models/standalone_xgb/standalone_model_lower.joblib',
    'MEDIAN_MODEL_PATH': BASE_DIR / 'src/models/standalone_xgb/standalone_model_median.joblib',
    'UPPER_MODEL_PATH': BASE_DIR / 'src/models/standalone_xgb/standalone_model_upper.joblib'
}

# --- Standalone XGBoost Backtesting Configuration (OVERRIDDEN) ---
STANDALONE_BACKTESTING_CONFIG = {
    'GEOJSON_PATH': DATA_DIR / '01_raw/districts_official.geojson',
    'REPORT_DIR': BASE_DIR / 'reports/figures/district_level_diagnostics/standalone_xgb_champion',
    'BACKTEST_START_YEAR': 2000,
    'BACKTEST_END_YEAR': 2024,
    'LOW_DATA_THRESHOLD': 10,
    'MIN_DATAPOINTS_FOR_PLOT': 10,
    'CALIBRATION_SET_SIZE': 0.15,
    'NOMINAL_COVERAGE': 0.95
}

# --- Model Comparison Configuration ---
MODEL_COMPARISON_CONFIG = {
    'NOMINAL_COVERAGE_PERCENT': 95.0,
    'FINAL_ENSEMBLE_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/final_ensemble_champion/full_backtest_predictions.csv',
    'NGBOOST_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/final_ngboost_champion/full_backtest_predictions.csv',
    'ADAPTIVE_CQR_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/adaptive_cqr_champion/full_backtest_predictions.csv',
    'HYBRID_XGB_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv',
    'STANDALONE_XGB_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/standalone_xgb_champion/full_backtest_predictions.csv',
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
