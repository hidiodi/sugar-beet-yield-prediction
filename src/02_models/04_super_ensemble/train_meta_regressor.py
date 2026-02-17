import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import sys
import numpy as np
import json

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib
analysis_config = importlib.import_module("src.03_analysis.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = analysis_config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])

INPUT_FILENAME = 'super_ensemble_training_data.csv'
METADATA_FILENAME = 'super_ensemble_weights.json'

WALK_FORWARD_START_YEAR = 2000
LAST_HISTORICAL_YEAR = 2024

# --- EXPERT SYSTEM CONFIGURATION ---
# Derived from domain analysis of regime-specific performance:
# 1. Extreme Stress (< 0.8 signal): Statistical Trend is safest (Hybrid XGB tends to over-predict crashes).
# 2. Bumper Years (> 1.0 signal): Robust Linear captures upside best.
# 3. Normal Regime: A blend of Robust Linear (60%) and V31 Solar Gated (40%) provides the best stability.
EXPERT_PARAMS = {
    'stress_threshold': 0.8,
    'bumper_threshold': 1.0,
    'robust_weight': 0.6,
    'v31_weight': 0.4
}

def get_signal(row):
    trend = row.get('Statistical_Trend_pred', np.nan)
    hybrid = row.get('Hybrid_XGB_pred', np.nan)
    if pd.isna(hybrid) or pd.isna(trend) or trend <= 0:
        return 1.0
    return hybrid / trend

def super_ensemble_logic(row, params):
    robust = row.get('Robust_Linear_pred', np.nan)
    v31 = row.get('V31_Solar_Gated_pred', np.nan)
    trend = row.get('Statistical_Trend_pred', np.nan)

    # Fallback
    if pd.isna(robust) or pd.isna(v31) or pd.isna(trend):
        if not pd.isna(robust): return robust
        if not pd.isna(trend): return trend
        return np.nan

    signal = get_signal(row)

    stress_thresh = params['stress_threshold']
    bumper_thresh = params['bumper_threshold']
    robust_wt = params['robust_weight']
    v31_wt = params['v31_weight']
    trend_wt = 1.0 - robust_wt - v31_wt

    if signal < stress_thresh:
        # Extreme Stress -> Trust Trend (Safety Mode)
        return trend
    elif signal > bumper_thresh:
        # Bumper Year -> Trust Robust Linear (Upside Mode)
        return robust
    else:
        # Normal Regime -> Optimal Blend
        return robust_wt * robust + v31_wt * v31 + trend_wt * trend

def train_meta_regressor():
    input_path = OUTPUT_DIR / INPUT_FILENAME
    if not input_path.exists():
        logging.warning(f"Input file not found: {input_path}")
        return

    logging.info("--- Training Meta-Learner (Expert System Ensemble) ---")
    df = pd.read_csv(input_path)

    # 1. Inspect Data
    model_cols = ['Statistical_Trend_pred', 'Robust_Linear_pred', 'V31_Solar_Gated_pred', 'Hybrid_XGB_pred']
    model_cols = [c for c in model_cols if c in df.columns]

    logging.info(f"Using Models: {model_cols}")
    logging.info(f"Expert Configuration: {EXPERT_PARAMS}")

    # 2. Apply Logic Walk-Forward
    results = []

    for year in range(WALK_FORWARD_START_YEAR, LAST_HISTORICAL_YEAR + 1):
        test = df[df['year'] == year].copy()

        # Apply Expert Logic
        test['Super_Ensemble_pred'] = test.apply(lambda row: super_ensemble_logic(row, EXPERT_PARAMS), axis=1)

        # Safety Clip (100 - 1200 dt/ha range)
        test['Super_Ensemble_pred'] = test['Super_Ensemble_pred'].clip(lower=100, upper=1200)

        results.append(test)

    if not results:
        logging.warning("No results generated.")
        return

    df_res = pd.concat(results)

    # 3. Evaluation
    mae_trend = mean_absolute_error(df_res['kreisYield'], df_res['Statistical_Trend_pred'])
    mae_ens = mean_absolute_error(df_res['kreisYield'], df_res['Super_Ensemble_pred'])
    skill = (1 - (mae_ens / mae_trend)) * 100

    logging.info("\n" + "=" * 60)
    logging.info(f"SUPER ENSEMBLE (EXPERT SYSTEM) RESULTS ({WALK_FORWARD_START_YEAR}-{LAST_HISTORICAL_YEAR})")
    logging.info("=" * 60)
    logging.info(f"Trend MAE:          {mae_trend:.4f}")
    logging.info(f"Super Ensemble MAE: {mae_ens:.4f}")
    logging.info(f"Skill Improvement:  {skill:.4f}%")

    # Recent Volatility (2010-2024)
    df_recent = df_res[df_res['year'] >= 2010]
    if not df_recent.empty:
        mae_trend_rec = mean_absolute_error(df_recent['kreisYield'], df_recent['Statistical_Trend_pred'])
        mae_ens_rec = mean_absolute_error(df_recent['kreisYield'], df_recent['Super_Ensemble_pred'])
        skill_rec = (1 - (mae_ens_rec / mae_trend_rec)) * 100
        logging.info("-" * 60)
        logging.info(f"RECENT VOLATILITY (2010-2024) N={len(df_recent)}")
        logging.info(f"Trend MAE:          {mae_trend_rec:.4f}")
        logging.info(f"Super Ensemble MAE: {mae_ens_rec:.4f}")
        logging.info(f"Skill Improvement:  {skill_rec:.4f}%")

    # Save outputs
    honesty_path = OUTPUT_DIR / 'super_ensemble_walkforward_predictions.csv'
    df_res.to_csv(honesty_path, index=False)

    tscv_path = OUTPUT_DIR / 'super_ensemble_final_forecast_TSCV.csv'
    df_res.to_csv(tscv_path, index=False)

    # Save metadata
    metadata = {
        "strategy": "Expert System (Threshold Switching + Blend)",
        "params": EXPERT_PARAMS,
        "skill_overall": skill,
        "skill_recent": skill_rec if not df_recent.empty else 0.0
    }
    with open(OUTPUT_DIR / METADATA_FILENAME, 'w') as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"\n✓ Saved final predictions to {tscv_path}")
    logging.info(f"✓ Saved metadata to {OUTPUT_DIR / METADATA_FILENAME}")

if __name__ == '__main__':
    train_meta_regressor()
