import sys
import os
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils.pipeline_runner import run_pipeline

# Define the pipeline sequence comment/uncomment to enable disable single steps
SCRIPTS = [
    # --- 1. Simulation Prep & Execution (WOFOST) ---
    "src/02_models/01_simulation/Wofost7.2/prepare_genetic_parameters.py",
    "src/02_models/01_simulation/Wofost7.2/prepare_site_data.py",
    "src/02_models/01_simulation/Wofost7.2/prepare_forecast_weather.py",
    "src/02_models/01_simulation/Wofost7.2/prepare_initial_conditions.py",
    "src/02_models/01_simulation/Wofost7.2/execute_wofost_simulation.py",

    # --- 1. Simulation Prep & Execution (Heat Signal) ---
    "src/02_models/01_simulation/multivariate_heat_signal.py",

    # --- 2. Feature Engineering & Trend ---
    "src/02_models/03_components/statistical/estimate_yield_trend.py",
    "src/02_models/02_features/generate_stage1_features.py",
    "src/02_models/02_features/generate_stage2_features.py",

    # --- 3. Component Models ---
    "src/02_models/03_components/native_ensemble/train_physics_informed_model.py",
    "src/02_models/03_components/native_ensemble/train_physics_ensemble.py",
    "src/02_models/03_components/hybrid_xgb/train_yield_ratio_xgb.py",
    "src/02_models/03_components/hybrid_xgb/backtest_yield_ratio_xgb.py" ,
    "src/02_models/03_components/robust_linear/train_robust_integrator.py",

    # --- 4. Super Ensemble ---
    "src/02_models/04_super_ensemble/prepare_ensemble_data.py",
    "src/02_models/04_super_ensemble/train_meta_regressor.py",
    "src/02_models/04_super_ensemble/execute_ensemble_forecast.py",

    # --- 5. Analysis & Diagnostics ---
    "src/02_models/05_analysis/check_data_leakage.py",
    "src/02_models/05_analysis/analyze_error_distribution.py",
    "src/02_models/05_analysis/analyze_super_ensemble.py",
    "src/02_models/05_analysis/compare_model_versions.py",
]

PIPELINE_NAME = "Hybrid Model Execution Pipeline"

def main():
    """
    Executes the comprehensive hybrid model pipeline.
    """
    run_pipeline(pipeline_name=PIPELINE_NAME, script_paths=SCRIPTS)

if __name__ == "__main__":
    main()
