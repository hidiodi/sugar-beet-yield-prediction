import sys
import os

# Ensure the project root is in the Python path
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from src.utils.pipeline_runner import run_pipeline
from src import config as global_config

def main():
    """
    Defines and executes the main data processing pipeline.
    This script now sources its configuration from the central config file.
    """
    # Execute the pipeline using settings from the config file
    run_pipeline(pipeline_name=global_config.PIPELINE_NAME, script_paths=global_config.SCRIPTS_TO_RUN)

if __name__ == "__main__":
    main()
