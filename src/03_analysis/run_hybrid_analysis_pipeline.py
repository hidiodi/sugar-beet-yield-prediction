# C:\Users\timSc\PycharmProjects\sugar-beet-yield-prediction\src\03_analysis\run_hybrid_analysis_pipeline.py

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
    PIPELINE_NAME = "Analysis Hybrid Model Pipeline"

    # Define the sequence of scripts to execute.
    SCRIPTS_TO_RUN = [
        "src/03_analysis/hybrid_model_analysis/analyze_input_features.py",
        "src/03_analysis/hybrid_model_analysis/analyze_wofost_pipeline.py",
        "src/03_analysis/hybrid_model_analysis/analyze_hybrid_model.py",
    ]

    # Execute the pipeline
    run_pipeline(pipeline_name=PIPELINE_NAME, script_paths=SCRIPTS_TO_RUN)


if __name__ == "__main__":
    main()