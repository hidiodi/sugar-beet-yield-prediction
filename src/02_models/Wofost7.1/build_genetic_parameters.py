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
# Tuning Strategy v6.14 (The Absolute Flatline):
# Problem: 1980-1990 grew despite AMAX=0. (Cause: TSUM1/CVO were leaking trend).
#          2014+ was too high (970+ vs 840).
# Fix:
# 1. Kill TSUM1/CVO Sensitivity (0.0). Drive trend PURELY via AMAX to ensure control.
# 2. Scalar 0.71. Sets 1980 baseline to ~440 dt/ha.
# 3. AMAX Sensitivities:
#    - 1980-1995: 0.00 (True Flatline).
#    - 1996-2012: 0.20 (Suppress the middle bulge).
#    - 2013+:     0.65 (Reduced from 0.95 to hit reasonable modern highs).

GLOBAL_YIELD_GAP_SCALAR = 0.71

EFF_SENSITIVITY = 0.0
TSUM1_SENSITIVITY = 0.0  # Disabled to prevent "creeping" growth in early years
CVO_SENSITIVITY = 0.0  # Disabled to prevent "creeping" growth
ROOT_SENSITIVITY = 0.0


def get_piecewise_sensitivity(year):
    """
    Returns AMAX sensitivity based on the era.
    """
    if year < 1996:
        return 0.00  # True Flatline for the first 15 years.
    elif year < 2013:
        return 0.20  # Suppress the 2000-2010 bulge.
    else:
        return 0.65  # Moderate modern growth (Was 0.95, which overshot).


def calculate_factors_from_data(input_csv, dataset_start_year=1980, model_reference_year=1980):
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"FATAL: Official genetic data not found at {input_csv}")
        sys.exit(1)

    # 1. Find Reference Values (Anchor: 1980)
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

        # 2. Calculate Delta relative to 1980
        raw_delta_yield = (row['relative_sugar_yield_index'] - ref_yield_idx) / 100.0
        delta_pol = row['sugar_content_pol'] - ref_sugar_pol

        # 3. Get Sensitivity (Piecewise)
        amax_sens = get_piecewise_sensitivity(year)

        # 4. Calculate Factors
        # TSUM1 and CVO are now LOCKED to 1.0 relative to trend to prevent leakage.
        eff_raw = 1.0
        tsum1_raw = 1.0
        cvo_raw = 1.0
        root_raw = 1.0

        # AMAX carries the entire burden of the Yield Trend
        amax_raw = 1.0 + (raw_delta_yield * amax_sens)

        # 5. Apply Global Yield Gap Scalar
        amax_final = amax_raw * GLOBAL_YIELD_GAP_SCALAR

        # Safety clips
        amax_final = max(0.5, amax_final)

        final_factors[str(year)] = {
            'EFF_FACTOR': 1.0,
            'CVO_FACTOR': round(cvo_raw, 4),
            'AMAX_FACTOR': round(amax_final, 4),
            'ROOT_FACTOR': round(root_raw, 4),
            'TSUM1_FACTOR': round(tsum1_raw, 4)
        }

    return final_factors


def main():
    logging.info(f"--- Building Genetic Factors (Pure AMAX Driver: Scalar {GLOBAL_YIELD_GAP_SCALAR}) ---")
    input_csv = config.DATA_DIR / '01_raw/official_genetic_trends.csv'
    output_json = config.PROCESSED_DATA_DIR / 'GeneticGainFactors.json'
    factors = calculate_factors_from_data(input_csv, 1980, 1980)
    with open(output_json, 'w') as f: json.dump(factors, f, indent=4)


if __name__ == "__main__":
    main()