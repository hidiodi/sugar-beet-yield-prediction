
from src.utils.pipeline_runner import run_pipeline

def main():
    """
    Defines and executes the main data processing pipeline.
    This script is just a definition file; all execution logic
    is handled by the pipeline_runner.
    """
    PIPELINE_NAME = "Main Hybrid Model Pipeline"

    # Define the sequence of scripts to execute.
    # Paths must be relative to this file's location (the project root).
    SCRIPTS_TO_RUN = [
        #"src/02_models/Wofost/run_wofost_pipeline.py",
        "src/01_data/FeatureEngineering/build_stage1_features.py",
        "src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py",
        "src/02_models/XGBoost/regression_model/Testing/backtest_final_quantile_model.py",
        "src/03_analysis/compare_model_versions.py",
    ]

    # Execute the pipeline
    run_pipeline(pipeline_name=PIPELINE_NAME, script_paths=SCRIPTS_TO_RUN)


if __name__ == "__main__":
    main()