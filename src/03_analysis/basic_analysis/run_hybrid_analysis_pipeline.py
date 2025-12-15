# C:\Users\timSc\PycharmProjects\sugar-beet-yield-prediction\src\03_analysis\run_hybrid_analysis_pipeline.py
# Refactored to use central configuration from src.config

import sys
import os
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils.pipeline_runner import run_pipeline
from src import config

def main():
    """
    Defines and executes the main data processing pipeline.
    This script now sources its configuration from the central config file.
    """
    run_pipeline(
        pipeline_name=config.ANALYSIS_PIPELINE_NAME,
        script_paths=config.ANALYSIS_SCRIPTS_TO_RUN
    )


if __name__ == "__main__":
    main()
