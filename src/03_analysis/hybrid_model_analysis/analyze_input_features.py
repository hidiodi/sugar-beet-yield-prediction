import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# Project Setup
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')


def analyze_feature_signal():
    logging.info("--- STARTING DEEP FEATURE INSPECTION (The 'Z-Score' Audit) ---")

    # 1. Load Data
    data_path = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
    df = pd.read_csv(data_path)

    # 2. Define the Target: "Trend-Adjusted Residual"
    # This is exactly what the XGBoost model is trying to learn.
    # If stage1_forecast is missing, approximate it or drop.
    if 'stage1_forecast' not in df.columns:
        logging.error("stage1_forecast missing. Cannot calculate residuals.")
        return

    df['yield_residual'] = df['kreisYield'] - df['stage1_forecast']

    # 3. Calculate Z-Scores for EVERYTHING
    # This puts all features on the same scale (Standard Deviations from Mean).
    # A Z-Score of +2.0 means "Top 2.5% extreme event".
    feature_cols = [c for c in df.columns if
                    pd.api.types.is_numeric_dtype(df[c]) and c not in ['district_no', 'year', 'kreisYield',
                                                                       'yield_residual']]

    z_df = df[['district_no', 'year']].copy()

    stats = {}
    for col in feature_cols:
        mu = df[col].mean()
        sigma = df[col].std()
        if sigma == 0: continue
        z_df[f'Z_{col}'] = (df[col] - mu) / sigma
        stats[col] = (mu, sigma)

    # 4. The "Smoking Gun" Analysis for Key Years
    target_years = [2014, 2018, 2024]

    for year in target_years:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"🚨 FORENSIC AUDIT: YEAR {year} 🚨")
        logging.info(f"{'=' * 60}")

        # Get data for this year
        yr_data = df[df['year'] == year]
        yr_z = z_df[z_df['year'] == year]

        avg_yield_resid = yr_data['yield_residual'].mean()
        resid_z = (avg_yield_resid - df['yield_residual'].mean()) / df['yield_residual'].std()

        logging.info(f"Yield Anomaly: {avg_yield_resid:.2f} dt/ha (Z-Score: {resid_z:.2f} sigma)")
        if abs(resid_z) < 1.0:
            logging.info("NOTE: This year was NOT a global outlier. Regional analysis recommended.")

        # Find which features were MOST EXTREME this year (High Z-Score Magnitude)
        # We average the Z-scores across all districts to find the "Global Signal" of the year.
        mean_z_scores = yr_z.drop(columns=['district_no', 'year']).mean().sort_values(ascending=False)

        logging.info("\n>>> WHAT WAS ABNORMAL THIS YEAR? (Top Positive Anomalies)")
        logging.info(f"{'Feature':<50} | {'Z-Score':<10} | {'Interpretation'}")
        logging.info("-" * 80)
        for name, score in mean_z_scores.head(10).items():
            logging.info(
                f"{name.replace('Z_', ''):<50} | {score:>6.2f}     | {'Extreme High' if score > 2 else 'High'}")

        logging.info("\n>>> WHAT WAS ABNORMAL THIS YEAR? (Top Negative Anomalies)")
        logging.info(f"{'Feature':<50} | {'Z-Score':<10} | {'Interpretation'}")
        logging.info("-" * 80)
        for name, score in mean_z_scores.tail(10).sort_values().items():
            logging.info(f"{name.replace('Z_', ''):<50} | {score:>6.2f}     | {'Extreme Low' if score < -2 else 'Low'}")

        # 5. Correlation Check
        # Did these extreme features actually correlate with the residual IN THIS YEAR?
        # (i.e., did districts with *more* of this feature do *better/worse*?)
        logging.info("\n>>> SIGNAL CHECK: Did these features drive the variance WITHIN the year?")
        corrs = yr_data[feature_cols].corrwith(yr_data['yield_residual']).sort_values(key=abs, ascending=False).head(5)
        for name, corr in corrs.items():
            logging.info(f"{name:<50} | Correlation: {corr:.2f}")


if __name__ == "__main__":
    analyze_feature_signal()