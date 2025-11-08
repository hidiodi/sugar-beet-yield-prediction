import sys
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from src.utils.pipeline_runner import run_pipeline

def main():
    """
    Defines and executes the main data processing pipeline.
    This script is just a definition file; all execution logic
    is handled by the pipeline_runner.
    """
    PIPELINE_NAME = "Main Hybrid Model Pipeline"

    # Define the sequence of scripts to execute.
    SCRIPTS_TO_RUN = [
        "src/02_models/Wofost7.1/04_create_daily_weather_file.py",
        "src/02_models/Wofost7.1/run_wofost_pipeline.py",
        "src/02_models/Wofost7.1/apply_detrending_correction.py",
        "src/01_data/FeatureEngineering/build_stage1_features.py",
        "src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py",
        "src/02_models/XGBoost/regression_model/Testing/backtest_final_quantile_model.py",
        "src/02_models/FinalEnsemble/backtest_final_ensemble.py"
        "src/03_analysis/basic_analysis/compare_model_versions.py",
    ]

    # Execute the pipeline
    run_pipeline(pipeline_name=PIPELINE_NAME, script_paths=SCRIPTS_TO_RUN)

if __name__ == "__main__":
    main()