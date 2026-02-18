import pandas as pd
import logging
from pathlib import Path
import json
import sys
import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib

analysis_config = importlib.import_module("src.03_analysis.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = analysis_config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])
INPUT_FILENAME = 'super_ensemble_training_data.csv'
# Changed: Now loading weights/params instead of a binary model
METADATA_FILENAME = 'super_ensemble_weights.json'
FINAL_FORECAST_FILENAME = 'super_ensemble_final_forecast_TSCV.csv'

TEST_YEAR_START = 2020


def super_ensemble_logic(row, params):
    """
    Returns (prediction, regime_label)
    """
    robust = row.get('Robust_Linear_pred', np.nan)
    v31 = row.get('V31_Solar_Gated_pred', np.nan)
    trend = row.get('Statistical_Trend_pred', np.nan)
    hybrid = row.get('Hybrid_XGB_pred', np.nan)

    if pd.isna(robust) or pd.isna(v31) or pd.isna(trend):
        return trend if not pd.isna(trend) else robust, "Fallback"

    signal = hybrid / trend if trend > 0 else 1.0

    if signal < params['stress_threshold']:
        return trend, "Statistical Trend (Stress Mode)"
    elif signal > params['bumper_threshold']:
        return robust, "Robust Linear (Bumper Mode)"
    else:
        # Blend
        pred = (params['robust_weight'] * robust +
                params['v31_weight'] * v31 +
                (1.0 - params['robust_weight'] - params['v31_weight']) * trend)
        return pred, "Expert Blend (Normal)"

def generate_forecast():
    input_path = OUTPUT_DIR / INPUT_FILENAME
    meta_path = OUTPUT_DIR / METADATA_FILENAME

    if not meta_path.exists():
        logging.error(f"Metadata {METADATA_FILENAME} not found. Run train_meta_regressor.py first.")
        return

    logging.info("--- Generating Expert-System Super-Ensemble Forecast ---")
    df = pd.read_csv(input_path)

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    params = metadata['params']
    logging.info(f"Loaded Strategy: {metadata['strategy']}")
    logging.info(f"Parameters: {params}")

    # Apply Inference Logic
    results = df.apply(lambda row: super_ensemble_logic(row, params), axis=1)
    df['Super_Ensemble_pred'] = [r[0] for r in results]
    df['Predicted_Best_Model'] = [r[1] for r in results]  # Restore the missing column

    df['Super_Ensemble_pred'] = df['Super_Ensemble_pred'].clip(lower=100, upper=1200)

    # Logging and Comparison
    df_test = df[df['year'] >= TEST_YEAR_START].copy()
    if not df_test.empty and 'kreisYield' in df_test.columns:
        from sklearn.metrics import mean_absolute_error
        mae_trend = mean_absolute_error(df_test['kreisYield'], df_test['Statistical_Trend_pred'])
        mae_ens = mean_absolute_error(df_test['kreisYield'], df_test['Super_Ensemble_pred'])

        logging.info("\n" + "=" * 80)
        logging.info(f"TEST PERIOD ({TEST_YEAR_START}-2024) EVALUATION")
        logging.info("=" * 80)
        logging.info(f"Trend MAE:                     {mae_trend:.4f}")
        logging.info(f"Super Ensemble MAE:            {mae_ens:.4f}")
        logging.info(f"Skill Improvement:             {((1 - mae_ens / mae_trend) * 100):.2f}%")

    # Final Output Formatting
    out_cols = ['year', 'district_no', 'kreisYield', 'Super_Ensemble_pred', 'Predicted_Best_Model']
    component_cols = [c for c in df.columns if c.endswith('_pred') and c != 'Super_Ensemble_pred']

    final_df = df[out_cols + component_cols]
    final_df.to_csv(OUTPUT_DIR / FINAL_FORECAST_FILENAME, index=False)
    logging.info(f"\n✓ Saved final forecast to {OUTPUT_DIR / FINAL_FORECAST_FILENAME}")


if __name__ == '__main__':
    generate_forecast()