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
INTERMEDIATE_DATA_DIR = DATA_DIR / "02_intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"

# --- Pipeline Configuration ---
PIPELINE_NAME = "Main Hybrid Model Pipeline"

SCRIPTS_TO_RUN = [
    #"src/01_data/download_all_data_pipeline.py",
    #"src/01_data/process_input_data_pipeline.py",
    #"src/02_models/Wofost7.1/build_initial_conditions.py",
    #"src/02_models/Wofost7.1/build_site_data.py",
    #"src/02_models/Wofost7.1/build_genetic_parameters.py",
    #"src/02_models/Wofost7.1/build_forecast_weather.py",
    #"src/02_models/Wofost7.1/analyze_pipeline_inputs.py",
    #"src/02_models/Wofost7.1/run_wofost_pipeline.py",
    #"src/02_models/Wofost7.1/validation_dashboard.py",
    #"src/02_models/Wofost7.1/create_trendModel.py",
    #"src/01_data/FeatureEngineering/build_stage1_features.py",
    #"src/03_analysis/basic_analysis/analyze_stage1_features.py",
    "src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py", # trains on residual of wofost
    "src/02_models/XGBoost/regression_model/Testing/backtest_final_quantile_model.py",  # trains on residual of wofost
    "src/02_models/XGBoost/regression_model/ModelScripts/train_standalone_xgb_model.py", # uses wofost as a simple input and trains with yield as target
    "src/02_models/XGBoost/regression_model/Testing/backtest_standalone_xgb_model.py", # uses wofost as a simple input and trains with yield as target
    #"src/02_models/experimental_models/backtest_adaptive_cqr_model.py",
    #"src/02_models/NGboost/train_final_ngboost_model.py",
    #"src/02_models/NGboost/backtest_final_ngboost_model.py",
    #"src/02_models/FinalEnsemble/backtest_final_ensemble.py",
    "src/03_analysis/basic_analysis/compare_model_versions.py",
    "src/03_analysis/hybrid_model_analysis/analyze_hybrid_model.py",
    "src/03_analysis/run_counterfactual_analysis.py",
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

# Path to the raw SEAS5 NetCDF files
SEAS5_RAW_DIR = BASE_DIR / "data" / "01_raw" / "ECMWF51_monthly_germany"
# Path to the output of the V3 (zonal stats) daily weather script
DAILY_WEATHER_DIR_V3 = BASE_DIR / "data" / "04_feature" / "weather_district_daily"

import datetime

WEATHER_START_YEAR = 1981
WEATHER_END_YEAR = 2024

# --- WOFOST Model Configuration ---
WOFOST_CONFIG = {
    # max range 1982 - 2024 and None to run all districts
    'START_YEAR': 1982, 'END_YEAR': 2024, 'DISTRICT_LIMIT': None,
    'FILE_PATHS': {
        'HISTORICAL_DAILY_WEATHER_DIR': DATA_DIR / '02_intermediate/daily_weather',
        'CORRECT_WEATHER_DIR': DATA_DIR / '04_feature/weather_district_daily',
        'YIELD_DATA': DATA_DIR / '02_intermediate/sugarbeet_yield.csv',
        'STATIC_SOIL_FEATURES': DATA_DIR / '03_processed/static_features_districts.csv',
        'INITIAL_CONDITIONS': DATA_DIR / '03_processed/InitialConditions.csv',
        'SEAS5_MEMBER_FEATURES': DATA_DIR / '02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv',
        'CROP_YAML': DATA_DIR / '01_raw/sugarbeet.yaml',
        'OUTPUT_DIR': DATA_DIR / '06_model_output/multi_year_final',
        'EXTREME_WEATHER_METRICS_OUTPUT': DATA_DIR / '06_model_output/ensemble_extreme_weather_metrics.csv',
    },

    'WEATHER_DEFAULTS': {'WIND_SPEED': 2.0, 'VAPOR_PRESSURE': 1.0},
    'WEATHER_GENERATOR': {'PRECIP_THRESHOLD_MM': 0.3, 'MIN_SRAD': 1.0},
    'AGROMANAGEMENT': {
        'CROP_START_DATE': datetime.date(2018, 3, 22), 'CROP_END_DATE': datetime.date(2018, 11, 15),
        'MAX_DURATION': 250,
    },
    'SOIL_COLUMN_MAPPING': {
        'sand': 'avg_sand_0_100cm', 'clay': 'avg_clay_0_100cm', 'som': 'avg_som_0_100cm', 'bdod': 'avg_bdod_0_100cm',
    },
    'SOIL_DEFAULTS_AND_CONSTANTS': {
        'RDMSOL': 120.0, 'KSUB': 10.0, 'SOPE': 10.0
    },
    'GENERIC_SITE': {'LATITUDE': 52.0, 'LONGITUDE': 10.0, 'ELEVATION': 50.0},
    'ANALOG_YEAR_CONFIG': {'NUM_ANALOGS': 5, 'MIN_YEARS_FOR_FIT': 10, },
    'CONSTANTS': {
        'DMC_SUGARBEET': 0.225,
        'INITIAL_ROOTING_DEPTH_CM': 10.0,
        'SOIL_PARTICLE_DENSITY': 2.65,
        'HARVEST_LOSS_FACTOR': 0.95,
        'MAX_ROOTING_DEPTH_GLOBAL_CAP_CM': 150.0
    },


    'GENETIC_GAIN_PARAMS': {
        'REFERENCE_YEAR': 2017,
        'START_YEAR': 1981,
    },

    'OPTIMIZATION': {
            'N_TRIALS': 500  # Start with 500, increase to 2000+ for a real run
        }
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

#Residual XGboost model config
XGBOOST_TRAINING_CONFIG = {
    'DATA_PATH': DATA_DIR / '05_model_input/stage1_preseason_features.csv',
    'HEAT_DAY_DIR': DATA_DIR / '02_intermediate/predicted_heat_stress_march.csv',  # forecasted heat days
    'MODEL_OUTPUT_DIR': BASE_DIR / 'src/models',
    'FEATURE_COLS': [
        # --- 1. The Core Physics (V14 Logic) ---
        'Index_Failure',
        'Index_Bumper',

        # --- 2. The Components (Context) ---
        'z_heat',
        'z_tank',
        'z_anoxia',
        'z_rain',
        'z_sow',
        'z_bal',

        # --- 3. The Baseline (NOTE: These are in the list so the Config knows about them,
        # but the script will DROP them from X_train automatically) ---
        'stage1_forecast',
        #'year',
        'is_gdr',
        'state_encoded',

        # --- 4. Critical Interactions ---
        'trend_x_failure',
        'trend_x_bumper',

        # --- 5. Validated Support Features ---
        'effective_winter_water',
        'anoxia_events',
        'wofost_yield_water_limited',
    ],
    'BEST_PARAMS_LOWER': {
        'n_estimators': 1242,
        'learning_rate': 0.057488,
        'max_depth': 3,
        'subsample': 0.540631,
        'colsample_bytree': 0.699607,
        'gamma': 0.264377,
        'min_child_weight': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'BEST_PARAMS_MEDIAN': {
        'n_estimators': 2500,
        'learning_rate': 0.02,       # Slower learning to capture subtle physics
        'max_depth': 6,              # Deep enough for interactions (State x Weather)
        'subsample': 0.7,
        'colsample_bytree': 0.4,     # <--- CRITICAL: Force model to look at other features
        'gamma': 1.0,                # Pruning to prevent overfitting noise
        'min_child_weight': 10,
        'n_jobs': -1,
        'random_state': 42

    },
    'BEST_PARAMS_UPPER': {
        'n_estimators': 667,
        'learning_rate': 0.096660,
        'max_depth': 6,
        'subsample': 0.782940,
        'colsample_bytree': 0.763611,
        'gamma': 12.620991,
        'min_child_weight': 7,
        'random_state': 42,
        'n_jobs': -1
    },
    'MONOTONE_CONSTRAINTS': {
        #'stage1_forecast': 1,
        'wofost_yield_water_limited': 1,
        'effective_winter_water': 1,
        #'year': 1,  # Trend is positive

        # V14 Constraints
        'Index_Bumper': 1,  # Positive Driver
        'Index_Failure': -1,  # Negative Driver
        'trend_x_bumper': 1,
        'trend_x_failure': -1,

        # Components
        'z_tank': 1,
        'z_rain': 1,
        'z_heat': -1,
        'z_anoxia': -1,
        'z_sow': -1
    },
    'QUANTILES': {'lower': 0.025, 'median': 0.5, 'upper': 0.975},
    'LOWER_MODEL_PATH': BASE_DIR / 'src/models/final_quantile_model_lower.joblib',
    'MEDIAN_MODEL_PATH': BASE_DIR / 'src/models/final_quantile_model_median.joblib',
    'UPPER_MODEL_PATH': BASE_DIR / 'src/models/final_quantile_model_upper.joblib'
}

SWITCHER_CONFIG = {
    'MODEL_PATH': BASE_DIR / 'src/models/switcher/regime_classifier.joblib',
    'ENCODER_PATH': BASE_DIR / 'src/models/switcher/label_encoder.joblib',
    # Mapping for Quantile Selection
    # If Class is X, use Quantile Y
    'ACTION_MAP': {
        'Crash': 'lower',     # Pessimistic
        'Normal': 'median',   # Realistic
        'Bumper': 'upper'     # Optimistic
    }
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

#Standalone XGboost model config
STANDALONE_XGB_CONFIG = {
    # UPDATED: Point back to the ORIGINAL source file.
    'DATA_PATH': DATA_DIR / '05_model_input/stage1_preseason_features.csv',
    'MODEL_OUTPUT_DIR': BASE_DIR / 'src/models/standalone_xgb',

    'FEATURE_COLS': XGBOOST_TRAINING_CONFIG['FEATURE_COLS'],

    # The script will CREATE and use 'yield_detrended' as the target
    'TARGET_COL': 'kreisYield',

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

    # --- Paths to Model Predictions ---
    'HYBRID_XGB_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv',  #Final Quantile Model # trains on residual of wofost
    'STANDALONE_XGB_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/standalone_xgb_champion/full_backtest_predictions.csv',   #standalone_xgb_champion  # uses wofost as a simple input and trains with yield as target
    'ADAPTIVE_CQR_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/adaptive_cqr_champion/full_backtest_predictions.csv',
    'NGBOOST_PREDICTIONS_FILE': BASE_DIR / 'reports/figures/district_level_diagnostics/final_ngboost_champion/full_backtest_predictions.csv',
    'STATISTICAL_TREND_FILE': DATA_DIR / '05_model_input/wofost_walkforward/final_honest_forecasts.csv',
    'PURE_WOFOST_ENSEMBLE_FILE': DATA_DIR / '06_model_output/multi_year_final/forecast_ensemble_1982-2024.csv',
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


