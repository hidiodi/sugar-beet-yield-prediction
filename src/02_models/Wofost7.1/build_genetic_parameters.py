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
EFF_SENSITIVITY = 0.15
AMAX_SENSITIVITY = 0.80
CVO_SENSITIVITY = 0.10
ROOT_SENSITIVITY = 2.00


def apply_trend_shaping(year, raw_delta_pct):
    """
    Forces Absolute Flat Start and Steep End.
    """
    # 1. ABSOLUTE FLAT START (1980-2000)
    if year <= 2000:
        return raw_delta_pct * 0.01

    # 2. STEEPENED MODERN GAIN (2014-2024)
    elif year >= 2014:
        return raw_delta_pct * 1.6

    # 3. TRANSITION
    else:
        return raw_delta_pct


def calculate_factors_from_data(input_csv, dataset_start_year=1980, model_reference_year=2017):
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"FATAL: Official genetic data not found at {input_csv}")
        sys.exit(1)

    base_row = df[df['year'] == dataset_start_year]
    base_yield_idx = base_row['relative_sugar_yield_index'].values[0]
    base_sugar_pol = base_row['sugar_content_pol'].values[0]

    raw_factors = {}

    for _, row in df.iterrows():
        year = int(row['year'])
        raw_delta_yield = (row['relative_sugar_yield_index'] - base_yield_idx) / 100.0

        shaped_delta = apply_trend_shaping(year, raw_delta_yield)

        delta_pol = row['sugar_content_pol'] - base_sugar_pol

        eff_raw = 1.0 + (shaped_delta * EFF_SENSITIVITY)
        cvo_raw = 1.0 + (delta_pol * CVO_SENSITIVITY)
        amax_raw = 1.0 + (shaped_delta * AMAX_SENSITIVITY)
        root_raw = 1.0 + (shaped_delta * ROOT_SENSITIVITY)

        raw_factors[year] = {
            'EFF_FACTOR': eff_raw,
            'CVO_FACTOR': cvo_raw,
            'AMAX_FACTOR': amax_raw,
            'ROOT_FACTOR': root_raw,
            'TSUM1_FACTOR': 1.0
        }

    ref_values = raw_factors[model_reference_year]
    final_factors = {}

    for year, facs in raw_factors.items():
        normalized_facs = {}
        for param, val in facs.items():
            if ref_values[param] != 0:
                normalized_facs[param] = round(val / ref_values[param], 4)
            else:
                normalized_facs[param] = 1.0
        final_factors[str(year)] = normalized_facs

    return final_factors


def main():
    logging.info("--- Building Genetic Factors (Steeper End) ---")
    input_csv = config.DATA_DIR / '01_raw/official_genetic_trends.csv'
    output_json = config.PROCESSED_DATA_DIR / 'GeneticGainFactors.json'
    factors = calculate_factors_from_data(input_csv, 1980, 2017)
    with open(output_json, 'w') as f: json.dump(factors, f, indent=4)


if __name__ == "__main__":
    main()