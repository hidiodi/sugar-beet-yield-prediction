import pandas as pd
import logging
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
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
MODEL_FILENAME = 'super_ensemble_meta_learner_TSCV.json'
LABEL_MAP_FILENAME = 'meta_learner_label_map_TSCV.json'
FINAL_FORECAST_FILENAME = 'super_ensemble_final_forecast_TSCV.csv'

TEST_YEAR_START = 2020


def generate_forecast():
    input_path = OUTPUT_DIR / INPUT_FILENAME
    model_path = OUTPUT_DIR / MODEL_FILENAME
    map_path = OUTPUT_DIR / LABEL_MAP_FILENAME

    if not model_path.exists(): return

    logging.info("--- Generating Consensus-Aware Super-Ensemble Forecast ---")
    df = pd.read_csv(input_path)

    clf = XGBClassifier()
    clf.load_model(model_path)

    with open(map_path, 'r') as f:
        label_map = json.load(f)

    sorted_models = sorted(label_map.keys(), key=lambda x: label_map[x])

    # Feature Selection (Must match Training!)
    # FIX: Explicitly exclude Regret_Weight and Median_Error so they aren't used as features
    exclude_cols = [
        'year', 'district_no', 'kreisYield', 'Best_Model', 'Oracle_Error',
        'Predicted_Model', 'Switch_Prediction', 'Target_Encoded',
        'Is_Garbage_Data', 'Raw_Bias', 'Regret_Weight', 'Median_Error'
    ]

    pred_cols = [c for c in df.columns if c.endswith('_pred') and c != 'Statistical_Trend_pred']
    feature_cols = [c for c in df.columns if c not in exclude_cols + pred_cols]

    logging.info(f"Inference Features ({len(feature_cols)}): {feature_cols}")

    # Predict Probas
    probas = clf.predict_proba(df[feature_cols])

    # Soft Voting
    weighted_preds = np.zeros(len(df))
    for idx, model_name in enumerate(sorted_models):
        pred_col = f"{model_name}_pred"
        model_prob = probas[:, idx]
        weighted_preds += (df[pred_col] * model_prob)
        df[f'Prob_{model_name}'] = model_prob

    df['Super_Ensemble_pred'] = weighted_preds
    df['Predicted_Best_Model'] = [sorted_models[i] for i in np.argmax(probas, axis=1)]

    # Analysis
    df_test = df[df['year'] >= TEST_YEAR_START].copy()
    if not df_test.empty:
        if 'Oracle_Error' not in df_test.columns: pass

        mae_trend = mean_absolute_error(df_test['kreisYield'], df_test['Statistical_Trend_pred'])
        mae_ens = mean_absolute_error(df_test['kreisYield'], df_test['Super_Ensemble_pred'])

        logging.info("\n" + "=" * 80)
        logging.info(f"TEST PERIOD ({TEST_YEAR_START}-2024)")
        logging.info("=" * 80)
        logging.info(f"Trend MAE:          {mae_trend:.4f}")
        logging.info(f"Super Ensemble MAE: {mae_ens:.4f} (Consensus-Aware)")
        logging.info(f"Gain vs Trend:      {mae_trend - mae_ens:+.4f}")

    df[['year', 'district_no', 'kreisYield', 'Super_Ensemble_pred', 'Predicted_Best_Model'] + [c for c in df.columns if
                                                                                               c.startswith(
                                                                                                   'Prob_')]].to_csv(
        OUTPUT_DIR / FINAL_FORECAST_FILENAME, index=False)
    logging.info(f"Saved to {OUTPUT_DIR / FINAL_FORECAST_FILENAME}")


if __name__ == '__main__':
    generate_forecast()