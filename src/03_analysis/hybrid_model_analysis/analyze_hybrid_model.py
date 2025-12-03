# File: src/03_analysis/hybrid_model_analysis/analyze_hybrid_model.py
# Description: Diagnostics for the Trained Hybrid Model.
#              Checks Feature Importance and Correlations.

import pandas as pd
import joblib
import logging
from pathlib import Path
import sys

# --- Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
MODEL_PATH = config.XGBOOST_TRAINING_CONFIG['MEDIAN_MODEL_PATH']


def main():
    if not MODEL_PATH.exists():
        logging.error("Model not found.")
        return

    model = joblib.load(MODEL_PATH)

    # Extract Feature Importance
    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')

    # Sort
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    logging.info("\n" + "=" * 60)
    logging.info("      HYBRID MODEL BRAIN SCAN (Top 15 Features)")
    logging.info("=" * 60)
    logging.info(f"{'Rank':<5} | {'Feature Name':<35} | {'Gain (Impact)':<10}")
    logging.info("-" * 60)

    for i, (feat, gain) in enumerate(sorted_imp[:15]):
        logging.info(f"{i + 1:<5} | {feat:<35} | {gain:.1f}")

    logging.info("-" * 60)

    # Interpretation Helper
    top_feats = [x[0] for x in sorted_imp[:5]]

    if 'trend_vs_phys_gap' in top_feats or 'wofost_trend_ratio' in top_feats:
        logging.info("✅ SUCCESS: The model is listening to the Physics (Gap/Ratio).")
    else:
        logging.info("⚠️ WARNING: The model is ignoring the Physics Gap.")

    if 'national_avg_yield_lag1' in top_feats or 'global_yield_lag_ratio' in top_feats:
        logging.info("ℹ️ INFO: The model relies heavily on Inertia (Lag).")


if __name__ == "__main__":
    main()