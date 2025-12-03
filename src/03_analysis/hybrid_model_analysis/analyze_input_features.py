# File: src/03_analysis/hybrid_model_analysis/analyze_input_features.py
# REFACTORED (v2): Debugger Mode
# Focus: Identifying WHY columns are missing or have 0 correlation.

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


def analyze_missing_by_year(df, col_name):
    """Checks if missing values are clustered in specific years."""
    if col_name not in df.columns:
        return

    missing = df[df[col_name].isna()]
    if missing.empty:
        return

    missing_by_year = missing.groupby('year').size()
    total_by_year = df.groupby('year').size()

    logging.info(f"\n--- Missing '{col_name}' Breakdown ---")
    # Show first 5 years with missing data and last 5
    years_with_missing = missing_by_year.index.tolist()

    if len(years_with_missing) > 10:
        display_years = years_with_missing[:5] + ['...'] + years_with_missing[-5:]
    else:
        display_years = years_with_missing

    logging.info(f"Years affected: {display_years}")

    # Check recent years (critical for backtest)
    recent_missing = missing[missing['year'] >= 2020].shape[0]
    if recent_missing > 0:
        logging.warning(f"⚠️  CRITICAL: {recent_missing} rows missing in 2020-2024!")


def analyze_feature_integrity(df):
    log_header("FEATURE INTEGRITY SCAN")

    # 1. Check WOFOST Root Depth (Why is correlation NaN?)
    col = 'wofost_root_depth'
    if col in df.columns:
        n_nan = df[col].isna().sum()
        variance = df[col].var()
        logging.info(f"Feature: {col}")
        logging.info(f"  - Missing: {n_nan} ({n_nan / len(df):.1%})")
        logging.info(f"  - Variance: {variance:.4f}")
        logging.info(f"  - Unique Values: {df[col].nunique()}")
        if n_nan == len(df):
            logging.error("  ❌ ERROR: Merge failed. Column is entirely empty.")
        elif variance == 0:
            logging.warning("  ⚠️ WARNING: Feature is constant (no variance). Cannot correlate.")
    else:
        logging.error(f"  ❌ ERROR: {col} not found in dataframe.")

    # 2. Check Summer Temp (Name mismatch?)
    target_name = 'summer_temp_anomaly_forecast'
    if target_name in df.columns:
        logging.info(f"Feature: {target_name} -> FOUND")
    else:
        logging.error(f"Feature: {target_name} -> NOT FOUND")
        # Suggest alternatives
        matches = [c for c in df.columns if 'summer' in c and 'temp' in c]
        if matches:
            logging.info(f"  Did you mean: {matches}?")


def analyze_correlations_corrected(df):
    log_header("CORRELATION SCAN (Fixed)")

    # Define the exact column names we expect
    target = 'trend_residual'

    # Ensure residual exists
    if 'stat_trend_forecast' in df.columns and 'kreisYield' in df.columns:
        df[target] = df['kreisYield'] - df['stat_trend_forecast']
    else:
        logging.error("Cannot calc residual. Missing yield/trend columns.")
        return

    features = [
        'trend_vs_phys_gap',
        'wofost_sowing_doy',
        'wofost_initial_wav',
        'wofost_root_depth',
        'summer_temp_anomaly_forecast'  # Fixed name
    ]

    logging.info(f"{'Feature':<35} | {'Corr (r)':<10} | {'N Samples':<10}")
    logging.info("-" * 65)

    for feat in features:
        if feat in df.columns:
            # Drop NaNs specifically for this pair to get a valid N
            tmp = df[[feat, target]].dropna()
            if len(tmp) > 50:
                corr = tmp.corr().iloc[0, 1]
                logging.info(f"{feat:<35} | {corr:.3f}      | {len(tmp)}")
            else:
                logging.info(f"{feat:<35} | TOO FEW ({len(tmp)})")
        else:
            logging.info(f"{feat:<35} | NOT FOUND")


def main():
    try:
        df = pd.read_csv(FEATURES_FILE)
        logging.info(f"Loaded {len(df)} rows.")
    except Exception as e:
        logging.error(f"Load failed: {e}")
        return

    analyze_feature_integrity(df)

    # Check why 23% of trend is missing
    analyze_missing_by_year(df, 'stat_trend_forecast')

    # Check why 6% of WOFOST is missing
    analyze_missing_by_year(df, 'wofost_esp_mean')

    analyze_correlations_corrected(df)


if __name__ == "__main__":
    main()