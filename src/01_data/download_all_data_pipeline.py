import sys
from pathlib import Path

# Ensure the project root is in the Python path to allow for `src` imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils.pipeline_runner import run_pipeline
from src import config

def main():
    """
    Runs the data downloading pipeline using configuration from src.config.
    """
    run_pipeline(
        pipeline_name=config.DOWNLOAD_PIPELINE_NAME,
        script_paths=config.DOWNLOAD_SCRIPTS_TO_RUN
    )

if __name__ == "__main__":
    main()
