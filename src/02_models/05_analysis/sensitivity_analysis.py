import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import sys
import itertools

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Input path
SUPER_ENSEMBLE_WALKFORWARD_PATH = project_root / 'reports/figures/final_model_comparison/super_ensemble_walkforward_predictions.csv'

THRESHOLDS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
DOWNSIDE_TRUSTS = [0.1, 0.3, 0.5, 0.8, 1.0]
UPSIDE_TRUSTS = [0.0, 0.2, 0.5]

def load_data():
    if not SUPER_ENSEMBLE_WALKFORWARD_PATH.exists():
        logging.error(f"Input file not found: {SUPER_ENSEMBLE_WALKFORWARD_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(SUPER_ENSEMBLE_WALKFORWARD_PATH)

    rename_map = {
        'Statistical_Trend_pred': 'trend_pred',
        'Hybrid_XGB_pred': 'xgb_pred',
        'kreisYield': 'actual'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def apply_gate(row, threshold, downside_trust, upside_trust):
    trend = row['trend_pred']
    xgb = row['xgb_pred']

    if pd.isna(trend) or pd.isna(xgb):
        return trend

    delta_pct = (xgb - trend) / trend

    if delta_pct < -threshold:
        return trend + ((xgb - trend) * downside_trust)
    elif delta_pct > threshold:
        return trend + ((xgb - trend) * upside_trust)
    else:
        return trend

def run_grid_search(df):
    mask = (df['year'] >= 2005) & (df['year'] <= 2024)
    subset = df[mask].copy()

    if subset.empty:
        return

    trend_mae = mean_absolute_error(subset['actual'], subset['trend_pred'])
    logging.info(f"Baseline Trend MAE (2005-2024): {trend_mae:.4f}")

    best_mae = float('inf')
    best_params = None

    results = []

    for thresh, dt, ut in itertools.product(THRESHOLDS, DOWNSIDE_TRUSTS, UPSIDE_TRUSTS):
        preds = subset.apply(lambda row: apply_gate(row, thresh, dt, ut), axis=1)
        mae = mean_absolute_error(subset['actual'], preds)
        skill = (1 - (mae / trend_mae)) * 100

        results.append((thresh, dt, ut, mae, skill))

        if mae < best_mae:
            best_mae = mae
            best_params = (thresh, dt, ut)

    # Sort by MAE
    results.sort(key=lambda x: x[3])

    logging.info("\nTop 10 Configurations:")
    logging.info(f"{'Thresh':<8} {'DT':<5} {'UT':<5} {'MAE':<10} {'Skill (%)'}")
    for res in results[:10]:
        logging.info(f"{res[0]:<8.2f} {res[1]:<5.1f} {res[2]:<5.1f} {res[3]:<10.4f} {res[4]:.2f}")

    logging.info(f"\nBest Params: Thresh={best_params[0]}, DT={best_params[1]}, UT={best_params[2]}")

def main():
    df = load_data()
    if df.empty:
        return
    run_grid_search(df)

if __name__ == '__main__':
    main()
