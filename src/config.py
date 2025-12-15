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
    #"src/02_models/01_simulation/Wofost7.1/prepare_initial_conditions.py",
    #"src/02_models/01_simulation/Wofost7.1/prepare_site_data.py",
    #"src/02_models/01_simulation/Wofost7.1/prepare_genetic_parameters.py",
    #"src/02_models/01_simulation/Wofost7.1/prepare_forecast_weather.py",
    #"src/02_models/01_simulation/Wofost7.1/analyze_pipeline_inputs.py",
    #"src/02_models/01_simulation/Wofost7.1/execute_wofost_simulation.py",
    #"src/02_models/01_simulation/Wofost7.1/validation_dashboard.py",
    #"src/02_models/03_components/statistical/estimate_yield_trend.py",
    "src/01_data/FeatureEngineering/build_stage1_features.py",
    #"src/03_analysis/basic_analysis/analyze_stage1_features.py",
    "src/02_models/03_components/native_ensemble/train_physics_informed_model.py",
    "src/02_models/experimental_models/run_regime_switch.py",
    "src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py", # trains on residual of wofost
    "src/02_models/XGBoost/regression_model/Testing/backtest_final_quantile_model.py",  # trains on residual of wofost
    "src/02_models/03_components/hybrid_xgb/train_yield_ratio_xgb.py", # uses wofost as a simple input and trains with yield as target
    "src/02_models/03_components/hybrid_xgb/backtest_yield_ratio_xgb.py", # uses wofost as a simple input and trains with yield as target
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
