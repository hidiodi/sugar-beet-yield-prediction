from pathlib import Path
from src.config import BASE_DIR, DATA_DIR

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

# --- Standalone XGBoost Backtesting Configuration (OVERRIDDEN) ---
STANDALONE_BACKTESTING_CONFIG = {
    'GEOJSON_PATH': DATA_DIR / '01_raw/districts_official.geojson',
    'REPORT_DIR': BASE_DIR / 'reports/figures/district_level_diagnostics/standalone_xgb_champion',
    'BACKTEST_START_YEAR': 2014,
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
