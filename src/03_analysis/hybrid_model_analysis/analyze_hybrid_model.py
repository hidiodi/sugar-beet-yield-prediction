# File: src/03_analysis/hybrid_model_analysis/analyze_hybrid_model.py
# Description: Diagnostics for the Trained Hybrid & Standalone Models.
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


def analyze_model(model_path, model_name):
    if not model_path.exists():
        logging.error(f"{model_name} not found at {model_path}")
        return

    model = joblib.load(model_path)

    # Extract Feature Importance
    try:
        booster = model.get_booster()
        importance = booster.get_score(importance_type='gain')
    except Exception as e:
        logging.warning(f"Could not extract feature importance for {model_name}: {e}")
        return

    # Sort
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    logging.info("\n" + "=" * 60)
    logging.info(f"      {model_name.upper()} BRAIN SCAN (Top 15 Features)")
    logging.info("=" * 60)
    logging.info(f"{'Rank':<5} | {'Feature Name':<35} | {'Gain (Impact)':<10}")
    logging.info("-" * 60)

    for i, (feat, gain) in enumerate(sorted_imp[:15]):
        logging.info(f"{i + 1:<5} | {feat:<35} | {gain:.1f}")

    logging.info("-" * 60)

    # Interpretation Helper
    top_feats = [x[0] for x in sorted_imp[:5]]

    # Updated list of "Good" Physics features based on recent engineering
    physics_signals = [
        'trend_vs_phys_gap', 'wofost_trend_ratio',
        'optimal_growth_index', 'trend_x_winter_water',
        'solar_capture_potential', 'effective_winter_water'
    ]

    found_physics = [f for f in top_feats if f in physics_signals]

    if found_physics:
        logging.info(f"✅ SUCCESS: The model is listening to Physics features: {found_physics}")
    else:
        logging.info("⚠️ WARNING: The model is ignoring key Physics interaction terms in the top 5.")

    if 'year' in top_feats and model_name == "Hybrid Model":
        logging.info("ℹ️ INFO: Hybrid Model is heavily relying on 'year' (Trend-following behavior).")


def main():
    # 1. Analyze Hybrid Model
    analyze_model(config.XGBOOST_TRAINING_CONFIG['MEDIAN_MODEL_PATH'], "Hybrid Model")

    # 2. Analyze Standalone Model
    analyze_model(config.STANDALONE_XGB_CONFIG['MEDIAN_MODEL_PATH'], "Standalone Model")


if __name__ == "__main__":
    main()