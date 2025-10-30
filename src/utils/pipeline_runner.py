# src/utils/pipeline_runner.py

import subprocess
import sys
import logging
from pathlib import Path
import time
import os  # <-- Import the 'os' module

# --- Setup basic logging for the runner itself ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(module)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _execute_script(script_path: Path) -> bool:
    """
    Internal function to execute a single Python script from the project root.
    This version forces UTF-8 encoding to prevent Unicode errors.

    Args:
        script_path: A Path object pointing to the script to execute.

    Returns:
        True if the script ran successfully, False otherwise.
    """
    if not script_path.is_file():
        logging.error(f"Script not found at '{script_path}'. Skipping.")
        return False

    logging.info(f"--- Starting script: {script_path} ---")
    start_time = time.time()

    try:
        # --- ROBUST ENCODING FIX ---
        # Create a copy of the current environment
        env = os.environ.copy()
        # Set the PYTHONUTF8 environment variable to '1'. This forces the
        # child process to use UTF-8 for its stdin, stdout, and stderr.
        env['PYTHONUTF8'] = '1'
        # ---------------------------

        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',  # This tells the PARENT how to decode the stream
            bufsize=1,
            env=env  # <-- Pass the modified environment to the child process
        )

        for line in iter(process.stdout.readline, ''):
            print(line, end='')

        process.wait()
        end_time = time.time()
        duration = end_time - start_time

        if process.returncode == 0:
            logging.info(f"--- Finished Successfully: {script_path} (Duration: {duration:.2f}s) ---")
            return True
        else:
            logging.error(
                f"--- Script Failed: {script_path} (Exit Code: {process.returncode}) (Duration: {duration:.2f}s) ---")
            return False

    except Exception as e:
        logging.error(f"--- An unexpected error occurred while trying to run {script_path}: {e} ---")
        return False


def run_pipeline(pipeline_name: str, script_paths: list[str]):
    """
    Runs a defined pipeline by executing a list of scripts in sequence.

    Args:
        pipeline_name: The name of the pipeline for logging purposes.
        script_paths: A list of script paths (as strings) relative to the
                      project root where the pipeline is run from.
    """
    logging.info(f">>> Starting Pipeline: {pipeline_name} <<<")

    total_start_time = time.time()

    for script_str in script_paths:
        script_path = Path(script_str)
        success = _execute_script(script_path)
        if not success:
            logging.error(f">>> Pipeline '{pipeline_name}' halted due to an error. <<<")
            break

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    logging.info(f">>> Pipeline '{pipeline_name}' Finished. (Total Duration: {total_duration:.2f}s) <<<")