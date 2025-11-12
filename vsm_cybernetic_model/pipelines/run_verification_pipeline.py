# 03_run_verification_pipeline.py
import subprocess
import sys
from pathlib import Path

def run_verification_pipeline():
    """
    Runs all verification scripts in the VSM module.

    This orchestrator finds all Python scripts within any 'verification'
    subdirectory of the VSM module and executes them sequentially. This
    provides a single entry point to generate all validation artifacts
    for the feature engineering process.
    """
    print("--- Starting VSM-CPS Verification Pipeline ---")

    base_dir = Path(__file__).parent.parent
    verification_scripts = list(base_dir.glob("**/verification/*.py"))

    if not verification_scripts:
        print("No verification scripts found. Nothing to run.")
        return

    print(f"Found {len(verification_scripts)} verification scripts to run.")

    for script_path in verification_scripts:
        print(f"\n--- Running: {script_path.relative_to(base_dir)} ---")
        try:
            # We run each script as a separate process to ensure a clean environment
            # and to prevent any single script failure from halting the entire pipeline.
            result = subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("--- Stderr ---")
                print(result.stderr)
            print(f"--- Finished: {script_path.relative_to(base_dir)} ---")
        except subprocess.CalledProcessError as e:
            print(f"!!! ERROR running {script_path.relative_to(base_dir)} !!!")
            print(e.stdout)
            print(e.stderr)
            print("--- Halting verification pipeline due to error. ---")
            # In a CI/CD environment, you might want this to exit with an error code
            # sys.exit(1)

    print("\n--- VSM-CPS Verification Pipeline finished successfully. ---")

if __name__ == "__main__":
    run_verification_pipeline()
