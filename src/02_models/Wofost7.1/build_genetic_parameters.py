# File: src/02_models/Wofost7.1/build_genetic_parameters.py
# Description: Calculates genetic gain factors based on OFFICIAL BSA TRENDS.
#              Method: Loel et al. (2014) coefficients for EFF and CVO.
#              Normalization: Factors are normalized to 2017 = 1.0 (Model Reference Year).

import pandas as pd
import json
from pathlib import Path
import sys
import logging

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- SCIENTIFIC COEFFICIENTS ---
# Source: Loel et al. 2014 / Hoffmann 2018
# 1. EFF (Light Use Efficiency): 1% Yield Index gain = 0.8% EFF gain.
EFF_SENSITIVITY = 1.3

# 2. CVO (Conversion/Sink): 1.0 Pol point increase = 4% CVO gain.
CVO_SENSITIVITY = 0.05

# 3. AMAX (Max CO2 Assimilation):
# Literature suggests AMAX is relatively stable, but we allow slight scaling
# to account for modern "stay-green" traits.
AMAX_SENSITIVITY = 0.15


def calculate_factors_from_data(input_csv, dataset_start_year=1980, model_reference_year=2017):
    """
    Reads official BSA trend data and calculates WOFOST factors.
    Normalizes all factors so that model_reference_year == 1.0.
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"FATAL: Official genetic data not found at {input_csv}")
        sys.exit(1)

    # 1. Get Dataset Baseline Values (1980)
    # We calculate raw change relative to the start of the dataset first.
    base_row = df[df['year'] == dataset_start_year]
    if base_row.empty:
        logging.error(f"FATAL: Dataset start year {dataset_start_year} not in dataset.")
        sys.exit(1)

    base_yield_idx = base_row['relative_sugar_yield_index'].values[0]
    base_sugar_pol = base_row['sugar_content_pol'].values[0]

    # Dictionary to hold raw factors (relative to 1980)
    raw_factors = {}

    for _, row in df.iterrows():
        year = int(row['year'])

        # Calculate Delta relative to 1980
        delta_yield_pct = (row['relative_sugar_yield_index'] - base_yield_idx) / 100.0
        delta_pol = row['sugar_content_pol'] - base_sugar_pol

        # Calculate Raw Factors (1980 = 1.0)
        eff_raw = 1.0 + (delta_yield_pct * EFF_SENSITIVITY)
        cvo_raw = 1.0 + (delta_pol * CVO_SENSITIVITY)
        amax_raw = 1.0 + (delta_yield_pct * AMAX_SENSITIVITY)

        raw_factors[year] = {
            'EFF_FACTOR': eff_raw,
            'CVO_FACTOR': cvo_raw,
            'AMAX_FACTOR': amax_raw,
            'TSUM1_FACTOR': 1.0,
            'SPAN_FACTOR': 1.0,
            'RDMCR_FACTOR': 1.0,
            'TSUM2_FACTOR': 1.0
        }

    # 2. Normalize to Model Reference Year (2017)
    # This ensures our YAML configuration (Sugarbeet_601) acts as the 2017 standard.
    if model_reference_year not in raw_factors:
        logging.error(f"FATAL: Model reference year {model_reference_year} not in calculated factors.")
        sys.exit(1)

    ref_values = raw_factors[model_reference_year]

    final_factors = {}

    for year, facs in raw_factors.items():
        normalized_facs = {}
        for param, val in facs.items():
            # Normalize: Value_Year / Value_RefYear
            # e.g., if 1980 is 1.0 and 2017 is 1.2, then 1980 becomes 1.0/1.2 = 0.83
            if ref_values[param] != 0:
                normalized_facs[param] = round(val / ref_values[param], 4)
            else:
                normalized_facs[param] = 1.0

        final_factors[str(year)] = normalized_facs

    return final_factors


def main():
    logging.info("--- Building Genetic Factors from OFFICIAL BSA/IfZ DATA ---")

    input_csv = config.DATA_DIR / '01_raw/official_genetic_trends.csv'
    output_json = config.PROCESSED_DATA_DIR / 'GeneticGainFactors.json'

    # We start calculations from 1980 data, but normalize everything so 2017 is 1.0
    factors = calculate_factors_from_data(
        input_csv,
        dataset_start_year=1980,
        model_reference_year=2017
    )

    with open(output_json, 'w') as f:
        json.dump(factors, f, indent=4)

    logging.info(f"✓ Saved normalized genetic factors to {output_json}")
    logging.info(f"  Reference Year (2017) Check: {factors['2017']}")
    logging.info(f"  1980 Check (Should be < 1.0): {factors['1980']}")
    logging.info(f"  2024 Check (Should be > 1.0): {factors['2024']}")


if __name__ == "__main__":
    main()