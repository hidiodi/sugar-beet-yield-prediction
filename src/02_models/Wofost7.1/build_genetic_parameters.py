# File: src/02_models/Wofost7.1/build_genetic_parameters.py
import pandas as pd
import json
from pathlib import Path
import sys
import logging

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- COEFFICIENTS ---
# Tuning Strategy v6.7 (Rotation & Drop):
# Anchor: 2000 (FIXED).
# 1. Increase AMAX (0.25 -> 0.45): This "rotates" the line. It pushes 2013+ yields UP
#    and pushes 1980 yields DOWN. This solves the "flat tail" issue.
# 2. Decrease Scalar (0.75 -> 0.72): This drops the pivot point to fix the "overpredicting everywhere".

EFF_SENSITIVITY = 0.0
AMAX_SENSITIVITY = 0.45  # Increased to capture the post-2013 growth
TSUM1_SENSITIVITY = 0.05
CVO_SENSITIVITY = 0.05
ROOT_SENSITIVITY = 0.0

# Global Yield Gap Scalar
# 0.72 accounts for harvest loss, pests, and non-optimal management.
GLOBAL_YIELD_GAP_SCALAR = 0.72


def calculate_factors_from_data(input_csv, dataset_start_year=1980, model_reference_year=2000):
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"FATAL: Official genetic data not found at {input_csv}")
        sys.exit(1)

    # 1. Find Reference Values (Anchor: 2000)
    try:
        ref_row = df[df['year'] == model_reference_year].iloc[0]
        ref_yield_idx = ref_row['relative_sugar_yield_index']
        ref_sugar_pol = ref_row['sugar_content_pol']
    except IndexError:
        logging.error(f"Reference year {model_reference_year} not found.")
        sys.exit(1)

    final_factors = {}

    for _, row in df.iterrows():
        year = int(row['year'])

        # 2. Calculate Delta relative to 2000
        raw_delta_yield = (row['relative_sugar_yield_index'] - ref_yield_idx) / 100.0
        delta_pol = row['sugar_content_pol'] - ref_sugar_pol

        # 3. Calculate Factors
        eff_raw = 1.0

        # High sensitivity here forces the 'End' up and the 'Start' down
        amax_raw = 1.0 + (raw_delta_yield * AMAX_SENSITIVITY)

        tsum1_raw = 1.0 + (raw_delta_yield * TSUM1_SENSITIVITY)
        cvo_raw = 1.0 + (delta_pol * CVO_SENSITIVITY)
        root_raw = 1.0 + (raw_delta_yield * ROOT_SENSITIVITY)

        # 4. Apply Global Yield Gap Scalar
        amax_final = amax_raw * GLOBAL_YIELD_GAP_SCALAR

        # Safety clips
        amax_final = max(0.5, amax_final)
        tsum1_final = max(0.9, tsum1_raw)

        final_factors[str(year)] = {
            'EFF_FACTOR': 1.0,
            'CVO_FACTOR': round(cvo_raw, 4),
            'AMAX_FACTOR': round(amax_final, 4),
            'ROOT_FACTOR': round(root_raw, 4),
            'TSUM1_FACTOR': round(tsum1_final, 4)
        }

    return final_factors


def main():
    logging.info(
        f"--- Building Genetic Factors (Rotated: AMAX {AMAX_SENSITIVITY}, Scalar {GLOBAL_YIELD_GAP_SCALAR}) ---")
    input_csv = config.DATA_DIR / '01_raw/official_genetic_trends.csv'
    output_json = config.PROCESSED_DATA_DIR / 'GeneticGainFactors.json'
    factors = calculate_factors_from_data(input_csv, 1980, 2000)
    with open(output_json, 'w') as f: json.dump(factors, f, indent=4)


if __name__ == "__main__":
    main()