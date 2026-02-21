import pandas as pd
import logging
from pathlib import Path
import json
import sys
import numpy as np
import joblib

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
MODEL_FILENAME = 'super_ensemble_meta_model.joblib'
FINAL_FORECAST_FILENAME = 'super_ensemble_production_forecast.csv'

TEST_YEAR_START = 2020


def generate_forecast():
    logging.info("--- Generating Classifier Super-Ensemble Forecast ---")

    input_path = OUTPUT_DIR / INPUT_FILENAME
    meta_path = OUTPUT_DIR / METADATA_FILENAME
    model_path = OUTPUT_DIR / MODEL_FILENAME

    if not model_path.exists(): return

    df = pd.read_csv(input_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # 1. Reconstruct Features
    feature_cols = metadata['features']
    logging.info(f"Loaded Strategy: {metadata['strategy']}")

    # Re-create Diff Features
    base_models = ['Hybrid_XGB_pred', 'Robust_Linear_pred', 'V31_Solar_Gated_pred']
    for model in base_models:
        if model in df.columns and 'Statistical_Trend_pred' in df.columns:
            df[model] = df[model].fillna(df['Statistical_Trend_pred'])
            diff_col = f"{model}_diff"
            df[diff_col] = df[model] - df['Statistical_Trend_pred']

    # Fill Context NaNs
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # 2. Predict Probabilities
    meta_model = joblib.load(model_path)
    probs = meta_model.predict_proba(df[feature_cols])
    classes = meta_model.classes_

    # 3. Soft Voting Blend
    final_pred = np.zeros(len(df))
    for i, model_name in enumerate(classes):
        if model_name in df.columns:
            final_pred += probs[:, i] * df[model_name].values
        else:
            final_pred += probs[:, i] * df['Statistical_Trend_pred'].values

    df['Super_Ensemble_pred'] = final_pred

    # Add Predicted Label (Highest Prob)
    best_idx = np.argmax(probs, axis=1)
    df['Predicted_Best_Model'] = [classes[i] for i in best_idx]

    # 4. Evaluation
    df_test = df[df['year'] >= TEST_YEAR_START].copy()
    if not df_test.empty and 'kreisYield' in df_test.columns:
        from sklearn.metrics import mean_absolute_error

        df_test = df_test.dropna(subset=['kreisYield', 'Super_Ensemble_pred'])
        if not df_test.empty:
            mae_trend = mean_absolute_error(df_test['kreisYield'], df_test['Statistical_Trend_pred'])
            mae_ens = mean_absolute_error(df_test['kreisYield'], df_test['Super_Ensemble_pred'])
            skill = ((1 - mae_ens / mae_trend) * 100)

            logging.info("\n" + "=" * 80)
            logging.info(f"PRODUCTION CHECK ({TEST_YEAR_START}-2024)")
            logging.info("=" * 80)
            logging.info(f"Trend MAE:          {mae_trend:.4f}")
            logging.info(f"Super Ensemble MAE: {mae_ens:.4f}")
            logging.info(f"Skill:              {skill:.2f}%")

    # 5. Save Output
    final_df = df[['year', 'district_no', 'kreisYield', 'Super_Ensemble_pred', 'Predicted_Best_Model']]
    final_df.to_csv(OUTPUT_DIR / FINAL_FORECAST_FILENAME, index=False)
    logging.info(f"\n✓ Saved production forecast to {OUTPUT_DIR / FINAL_FORECAST_FILENAME}")


if __name__ == '__main__':
    generate_forecast()