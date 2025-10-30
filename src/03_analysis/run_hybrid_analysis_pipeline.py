
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
        "src/03_analysis/hybrid_model_analysis/analyze_input_features.py",
        "src/03_analysis/hybrid_model_analysis/analyze_wofost_pipeline.py",
        "src/03_analysis/hybrid_model_analysis/analyze_hybrid_model.py",
    ]

    # Execute the pipeline
    run_pipeline(pipeline_name=PIPELINE_NAME, script_paths=SCRIPTS_TO_RUN)


if __name__ == "__main__":
    main()