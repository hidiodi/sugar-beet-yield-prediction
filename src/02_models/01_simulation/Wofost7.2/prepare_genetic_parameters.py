# File: src/02_models/Wofost7.1/build_genetic_parameters.py
import json
from pathlib import Path
import sys
import logging

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config as global_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
# STATIC GENETICS STRATEGY (v7.0 - The "Weather Sensor" Update):
# We remove the historical genetic trend simulation.
# We simulate a constant "Standard Modern Cultivar" (2020 baseline) for all years (1980-2024).
# This ensures that year-to-year yield variations in WOFOST are driven PURELY by weather.
# The XGBoost model will later map this "Weather Potential" to the actual Yield Trend.

GLOBAL_SCALAR = 0.85


def main():
    logging.info("--- Building Genetic Factors (Static Modern Cultivar) ---")

    # Range of years to generate (wider than needed to be safe)
    start_year = 1979
    end_year = 2025

    output_json = global_config.PROCESSED_DATA_DIR / 'GeneticGainFactors.json'

    final_factors = {}

    for year in range(start_year, end_year + 1):
        # NO TREND. Constant Modern Genetics.
        final_factors[str(year)] = {
            'EFF_FACTOR': 1.0,  # Light Use Efficiency (Standard)
            'CVO_FACTOR': 1.0,  # Convert to storage (Standard)
            'AMAX_FACTOR': 1.0 * GLOBAL_SCALAR,  # Max Assimilation (Standard * Scalar)
            'ROOT_FACTOR': 1.0,  # Root depth (Standard)
            'TSUM1_FACTOR': 1.0  # Phenology (Standard)
        }

    with open(output_json, 'w') as f:
        json.dump(final_factors, f, indent=4)

    logging.info(f"Generated static genetic factors for {start_year}-{end_year}")
    logging.info(f"Strategy: Pure Weather Sensor (Scalar {GLOBAL_SCALAR})")


if __name__ == "__main__":
    main()