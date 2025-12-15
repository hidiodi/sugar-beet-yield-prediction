# File: src/02_models/Wofost7.1/validation_dashboard.py
# Description: detailed tabular analysis of the WOFOST Ensemble.
#              Focuses on identifying "Hidden Signals" (Risk/Spread) in the logs.

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# --- Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# Set logging to print purely clean tables
logging.basicConfig(level=logging.INFO, format='%(message)s')


def analyze_ensemble_statistics():
    results_path = config.WOFOST_CONFIG['FILE_PATHS']['OUTPUT_DIR'] / "forecast_ensemble_results_raw.csv"
    yield_path = config.WOFOST_CONFIG['FILE_PATHS']['YIELD_DATA']

    if not results_path.exists():
        logging.error(f"FATAL: Simulation results not found at {results_path}")
        return

    logging.info(f"Loading Simulation Results...")
    df_sim = pd.read_csv(results_path)

    logging.info(f"Loading Actual Yields...")
    df_actual = pd.read_csv(yield_path)
    # Ensure matching district format
    df_actual['district_no'] = df_actual['district_no'].astype(str).str.zfill(5)

    # 1. Calculate National Average Actual Yield (Benchmark)
    actual_yearly = df_actual.groupby('year')['yield'].mean().reset_index()
    actual_yearly.rename(columns={'yield': 'actual_mean'}, inplace=True)

    # 2. Calculate Ensemble Statistics (The "Mind" of the Model)
    # We aggregate across ALL districts and ALL members for a given year to see the "National Signal"
    # This answers: "Did the model generally feel nervous about this year?"

    stats = df_sim.groupby('year')['yield_water_limited'].agg(
        sim_mean='mean',
        sim_median='median',
        sim_std='std',
        sim_min='min',
        sim_p05=lambda x: x.quantile(0.05),  # Extreme Downside (1-in-20 worst case)
        sim_p25=lambda x: x.quantile(0.25),
        sim_p75=lambda x: x.quantile(0.75),
        sim_p95=lambda x: x.quantile(0.95)  # Extreme Upside
    ).reset_index()

    # 3. Merge
    merged = pd.merge(stats, actual_yearly, on='year', how='left')

    # 4. Filter for requested range if possible
    merged = merged.sort_values('year')

    # --- OUTPUT TABLE ---
    headers = [
        "YEAR", "ACTUAL", "SIM_AVG", "BIAS",
        "SPREAD (Std)", "P05 (Crash)", "P95 (Boom)", "SKEW (Avg-Med)"
    ]

    logging.info("\n" + "=" * 115)
    logging.info(f" WOFOST ENSEMBLE FORENSICS: Risk & Distribution Analysis (1982-2024)")
    logging.info(f" Strategy: Static Genetics (Weather Sensor Only)")
    logging.info("=" * 115)
    logging.info(
        f"{headers[0]:<6} | {headers[1]:<8} | {headers[2]:<8} | {headers[3]:<6} | {headers[4]:<12} | {headers[5]:<11} | {headers[6]:<11} | {headers[7]:<12}")
    logging.info("-" * 115)

    for _, row in merged.iterrows():
        year = int(row['year'])
        act = row['actual_mean']
        sim = row['sim_mean']
        bias = sim - act if pd.notnull(act) else 0
        std = row['sim_std']
        p05 = row['sim_p05']
        p95 = row['sim_p95']

        # Skew: Positive = Tail is to the right (Boom possible). Negative = Tail is to the left (Crash possible).
        skew = row['sim_mean'] - row['sim_median']

        # --- Diagnostic Flags ---
        flags = []
        if pd.notnull(act):
            # Did we miss a crash? (Actual is way below Mean, check if P05 caught it)
            if (act < sim - 50) and (act < p05):
                flags.append("MISS_LOW")
            elif (act < sim - 50) and (act >= p05):
                flags.append("CAUGHT_IN_TAIL")  # The mean missed, but the risk model knew!

            # Did we miss a boom?
            if (act > sim + 50) and (act > p95):
                flags.append("MISS_HIGH")

        # Is the ensemble "Nervous"? (High Spread)
        if std > 40: flags.append("HIGH_UNCERTAINTY")

        flag_str = " ".join(flags)

        log_line = (f"{year:<6} | {act:8.1f} | {sim:8.1f} | {bias:6.1f} | "
                    f"{std:12.1f} | {p05:11.1f} | {p95:11.1f} | {skew:12.1f}  {flag_str}")

        logging.info(log_line)

    logging.info("=" * 115)
    logging.info("KEY INDICATORS TO LOOK FOR:")
    logging.info(
        "1. CAUGHT_IN_TAIL: The Mean was wrong, but the P05 (Risk) predicted the disaster. XGBoost loves this.")
    logging.info("2. SPREAD (Std): If 2018 has higher Std than 2017, the model 'sensed' the instability.")
    logging.info("3. SKEW: If Skew is negative, the model knows a crash is more likely than a boom.")
    logging.info("=" * 115)


if __name__ == "__main__":
    analyze_ensemble_statistics()