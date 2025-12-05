# File: src/03_analysis/hybrid_model_analysis/analyze_input_features.py
# REFACTORED (v3): Physics & Context Analyzer
# Focus: Verifying the new WOFOST signals and their correlation with Yield Deviations.

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import warnings

# --- Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

FEATURES_FILE = config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']['OUTPUT_FILE']


def log_header(title):
    logging.info("\n" + "=" * 60)
    logging.info(f" {title}")
    logging.info("=" * 60)


def check_physics_integrity(df):
    log_header("1. NEW PHYSICS INTEGRITY CHECK")

    new_cols = ['anoxia_events', 'prob_sowing_failure',
                'harvest_respiration_risk', 'prob_terminal_freeze']

    for col in new_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            zeros = (df[col] == 0).sum()
            max_val = df[col].max()
            mean_val = df[col].mean()

            logging.info(
                f"Feature: {col:<25} | Max: {max_val:<6.2f} | Mean: {mean_val:<6.2f} | % Zeros: {zeros / len(df):.1%}")

            if mean_val == 0:
                logging.error(f"  ❌ ERROR: {col} is entirely ZERO. Merge failed or data missing.")
            elif missing > 0:
                logging.warning(f"  ⚠️ WARNING: {col} has {missing} missing values.")
        else:
            logging.error(f"  ❌ CRITICAL: {col} NOT FOUND in dataset!")


def compare_years_2014_2018(df):
    log_header("2. REALITY CHECK: 2014 (Wet) vs 2018 (Dry)")

    # We compare specific features to see if they react correctly
    # 2014 should have HIGH Anoxia, LOW Heat Stress
    # 2018 should have LOW Anoxia, HIGH Heat Stress

    features_to_check = [
        'anoxia_events',
        'summer_water_balance_anomaly',
        'heat_stress_sq',
        'prob_sowing_failure',
        'effective_winter_water'
    ]

    available_feats = [f for f in features_to_check if f in df.columns]

    df_14 = df[df['year'] == 2014]
    df_18 = df[df['year'] == 2018]

    if df_14.empty or df_18.empty:
        logging.warning("Cannot compare years. Missing data for 2014 or 2018.")
        return

    logging.info(f"{'Feature':<30} | {'2014 (Avg)':<12} | {'2018 (Avg)':<12} | {'Delta':<10}")
    logging.info("-" * 70)

    for feat in available_feats:
        m14 = df_14[feat].mean()
        m18 = df_18[feat].mean()
        delta = m18 - m14
        logging.info(f"{feat:<30} | {m14:<12.4f} | {m18:<12.4f} | {delta:<10.4f}")

    logging.info("\nINTERPRETATION GUIDE:")
    logging.info(" - 'anoxia_events' should be HIGHER in 2014 (Positive Delta is BAD)")
    logging.info(" - 'heat_stress_sq' should be HIGHER in 2018 (Positive Delta is GOOD)")
    logging.info(" - 'summer_water_balance' should be NEGATIVE in 2018")


def analyze_correlations_with_anomaly(df):
    log_header("3. CORRELATION WITH YIELD ANOMALY")

    # We want to know: Does this feature explain why yield deviated from the trend?
    if 'stage1_forecast' in df.columns and 'kreisYield' in df.columns:
        df['yield_anomaly'] = df['kreisYield'] - df['stage1_forecast']
        target = 'yield_anomaly'
    else:
        logging.warning("Cannot calculate Yield Anomaly. Correlating with Raw Yield.")
        target = 'kreisYield'

    features = [
        # The New Physics
        'anoxia_events', 'prob_sowing_failure', 'harvest_respiration_risk',
        # The Context
        'effective_winter_water', 'summer_water_balance_anomaly',
        'heat_stress_sq', 'flash_drought_index',
        # Interactions
        'summer_heat_x_water_balance', 'winter_buffer_x_summer_heat',
        'optimal_growth_index'
    ]

    valid_feats = [f for f in features if f in df.columns]

    corrs = df[valid_feats + [target]].corr(method='spearman')[target].drop(target)
    corrs = corrs.sort_values(ascending=False)

    logging.info(f"Target: {target} (Spearman Correlation)")
    logging.info("-" * 40)
    for feat, val in corrs.items():
        logging.info(f"{feat:<30} : {val:.4f}")

    logging.info("\nEXPECTATIONS:")
    logging.info(" - Negative: anoxia_events, heat_stress_sq, flash_drought_index")
    logging.info(" - Positive: effective_winter_water, optimal_growth_index")


def main():
    try:
        df = pd.read_csv(FEATURES_FILE)
        logging.info(f"Loaded {len(df)} rows from {FEATURES_FILE}")
    except Exception as e:
        logging.error(f"Load failed: {e}")
        return

    # 1. Check if we have the new columns
    check_physics_integrity(df)

    # 2. Check if the physics make sense
    compare_years_2014_2018(df)

    # 3. Check if they predict anything
    analyze_correlations_with_anomaly(df)

    # 4. Critical Missing Data Check
    if 'stage1_forecast' in df.columns:
        missing_trend = df['stage1_forecast'].isna().sum()
        if missing_trend > 0:
            log_header("WARNING: MISSING TREND DATA")
            logging.warning(f"Trend (stage1_forecast) is missing in {missing_trend} rows.")
            # Breakdown by year
            logging.info(df[df['stage1_forecast'].isna()].groupby('year').size().head())


if __name__ == "__main__":
    main()