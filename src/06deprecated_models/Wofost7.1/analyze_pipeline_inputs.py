# File: src/02_models/Wofost7.1/analyze_pipeline_inputs.py
# Description: The "Single Source of Truth" Input Auditor.
#              Verifies Data Topology, Physical Plausibility, and Signal Validity.
#              Output: LOG ONLY (No images).

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import logging

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')

PATHS = {
    'static_site': config.PROCESSED_DATA_DIR / 'StaticSiteData.csv',
    'initial_conditions': config.PROCESSED_DATA_DIR / 'InitialConditions.csv',
    'genetic_params': config.PROCESSED_DATA_DIR / 'GeneticGainFactors.json'
}


def audit_topology(df_static, df_init):
    """Checks if the datasets match (Districts/Years)."""
    logging.info("\n" + "=" * 80)
    logging.info("      [1] DATASET TOPOLOGY AUDIT")
    logging.info("=" * 80)

    dist_static = set(df_static['district_no'].unique())
    dist_init = set(df_init['district_no'].unique())

    common = dist_static.intersection(dist_init)
    missing_in_init = dist_static - dist_init
    missing_in_static = dist_init - dist_static

    logging.info(f"Static Site Districts:      {len(dist_static)}")
    logging.info(f"Initial Cond Districts:     {len(dist_init)}")
    logging.info(f"Ready for Simulation:       {len(common)}")

    if missing_in_static:
        logging.warning(f"⚠️  WARNING: {len(missing_in_static)} districts have Weather but NO SOIL data.")
        logging.warning(f"    WOFOST will crash for these districts. Run 'clean_inputs.py' logic recommended.")

    if missing_in_init:
        logging.warning(f"⚠️  WARNING: {len(missing_in_init)} districts have Soil but NO WEATHER data.")


def audit_soil_physics(df_static):
    """Checks if the soil parameters are physically realistic."""
    logging.info("\n" + "=" * 80)
    logging.info("      [2] SOIL PHYSICS AUDIT (The 'Tank Size')")
    logging.info("=" * 80)

    # PAW = Plant Available Water (Volumetric)
    # Tank Size = PAW * Root Depth
    df = df_static[['district_no', 'SMFCF', 'SMW', 'RDMSOL']].drop_duplicates().copy()
    df['PAW_vol'] = df['SMFCF'] - df['SMW']
    df['Tank_Size_cm'] = df['PAW_vol'] * df['RDMSOL']

    # Stats
    logging.info("Distribution of Root Zone Water Capacity (cm):")
    logging.info(df['Tank_Size_cm'].describe().to_string())

    # Plausibility Checks
    # < 5cm is basically concrete/sand dune -> Instant death
    # > 40cm is a swamp/peat -> Unlimited water
    too_low = len(df[df['Tank_Size_cm'] < 5.0])
    too_high = len(df[df['Tank_Size_cm'] > 40.0])

    if too_low > 0:
        logging.warning(f"⚠️  CRITICAL: {too_low} districts have < 5cm water capacity (Likely bad soil data).")
    if too_high > 0:
        logging.warning(f"⚠️  NOTE: {too_high} districts have massive water capacity (>40cm).")

    logging.info(f"Avg Soil Quality: {df['Tank_Size_cm'].mean():.1f} cm of water storage.")


def audit_sowing_validity(df_init):
    """Checks if Sowing Dates makes sense across Time and Space."""
    logging.info("\n" + "=" * 80)
    logging.info("      [3] SOWING DATE VALIDITY AUDIT")
    logging.info("=" * 80)

    df = df_init.copy()
    df['sowing_date'] = pd.to_datetime(df['sowing_date'])
    df['doy'] = df['sowing_date'].dt.dayofyear

    # 1. Global Stats
    logging.info("Sowing Day of Year (DOY) Statistics:")
    logging.info(df['doy'].describe().to_string())

    # 2. Risk of Frost (Too Early) vs Yield Loss (Too Late)
    # March 1st = DOY 60. May 1st = DOY 121.
    too_early = (df['doy'] < 60).sum()
    too_late = (df['doy'] > 120).sum()
    total = len(df)

    logging.info(f"\nRisk Analysis:")
    logging.info(f"  > Pre-March 1st (Frost Risk): {too_early} / {total} ({too_early / total:.1%})")
    logging.info(f"  > Post-May 1st (Yield Loss):  {too_late} / {total} ({too_late / total:.1%})")

    # 3. Dynamic Response Check (The "Smart" Logic)
    # We aggregate by year to see the "National Sowing Date"
    yearly = df.groupby('year')['doy'].mean().sort_values()

    logging.info("\nTop 5 EARLIEST Sowing Years (Model Logic):")
    for y, doy in yearly.head(5).items():
        logging.info(f"  {y}: DOY {doy:.1f}")

    logging.info("\nTop 5 LATEST Sowing Years (Model Logic):")
    for y, doy in yearly.tail(5).items():
        logging.info(f"  {y}: DOY {doy:.1f}")

    # Standard Deviation Check
    std_dev = yearly.std()
    logging.info(f"\nInter-Annual Variability: +/- {std_dev:.1f} days")
    if std_dev < 3.0:
        logging.warning("⚠️  FAIL: Sowing dates are static. The model is ignoring winter temperatures.")
    else:
        logging.info("✅ PASS: Sowing dates show significant yearly variation.")


def audit_wav_validity(df_init):
    """Checks Soil Memory (WAV)."""
    logging.info("\n" + "=" * 80)
    logging.info("      [4] INITIAL SOIL WATER (WAV) AUDIT")
    logging.info("=" * 80)

    yearly = df_init.groupby('year')['WAV'].mean().sort_values()

    logging.info("Top 5 DRIEST Starts (March 1st):")
    for y, wav in yearly.head(5).items():
        logging.info(f"  {y}: {wav:.2f} cm")

    logging.info("\nTop 5 WETTEST Starts (March 1st):")
    for y, wav in yearly.tail(5).items():
        logging.info(f"  {y}: {wav:.2f} cm")

    # Check 2018 specifically
    if 2018 in yearly.index:
        wav_18 = yearly[2018]
        rank_18 = yearly.index.get_loc(2018) + 1
        logging.info(f"\n2018 Analysis: {wav_18:.2f} cm (Rank {rank_18}/{len(yearly)} Driest)")

    std_wav = yearly.std()
    logging.info(f"Inter-Annual Variability: +/- {std_wav:.2f} cm")
    if std_wav < 0.5:
        logging.warning("⚠️  FAIL: Initial Soil Water is static. ERA5 data might be missing/ignored.")
    else:
        logging.info("✅ PASS: Initial Soil Water responds to winter precipitation.")


def check_genetics():
    logging.info("\n" + "=" * 80)
    logging.info("      [5] GENETICS CONFIG CHECK")
    logging.info("=" * 80)
    if not PATHS['genetic_params'].exists():
        logging.error("Genetic file missing.")
        return

    with open(PATHS['genetic_params'], 'r') as f:
        gen = json.load(f)

    vals = [v['AMAX_FACTOR'] for v in gen.values()]
    if min(vals) == max(vals):
        logging.info(f"✅ Genetics are CONSTANT (Value: {vals[0]}).")
    else:
        logging.warning(f"⚠️  Genetics are TRENDING ({min(vals)} -> {max(vals)}). This complicates interpretation.")


def main():
    logging.info("--- DEEP INPUT AUDIT ---")

    if not PATHS['static_site'].exists() or not PATHS['initial_conditions'].exists():
        logging.error("Missing critical input files.")
        return

    df_static = pd.read_csv(PATHS['static_site'], dtype={'district_no': str})
    df_init = pd.read_csv(PATHS['initial_conditions'], dtype={'district_no': str})

    audit_topology(df_static, df_init)
    audit_soil_physics(df_static)
    audit_sowing_validity(df_init)
    audit_wav_validity(df_init)
    check_genetics()

    logging.info("\n" + "=" * 80)
    logging.info("AUDIT COMPLETE")


if __name__ == "__main__":
    main()