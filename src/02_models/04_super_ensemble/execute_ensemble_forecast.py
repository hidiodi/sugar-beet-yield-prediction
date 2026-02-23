import pandas as pd
import logging
import numpy as np
from pathlib import Path
import sys
import joblib

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib

analysis_config = importlib.import_module("src.03_analysis.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
OUTPUT_DIR = Path(analysis_config.MODEL_COMPARISON_CONFIG['OUTPUT_DIR'])

MODEL_FILENAME = 'super_ensemble_meta_model.joblib'
FEATURES_FILENAME = 'meta_features.joblib'
FINAL_FORECAST_FILENAME = 'super_ensemble_production_forecast.csv'


def generate_forecast():
    logging.info("--- Generating Stacking Super-Ensemble Forecast ---")

    model_path = OUTPUT_DIR / MODEL_FILENAME
    feat_path = OUTPUT_DIR / FEATURES_FILENAME

    if not model_path.exists():
        logging.error("Meta-model not found. Train it first.")
        return

    meta_model = joblib.load(model_path)
    features = joblib.load(feat_path)

    input_path = OUTPUT_DIR / 'super_ensemble_final_forecast_TSCV.csv'
    if not input_path.exists():
        logging.error("Test data not found.")
        return

    df = pd.read_csv(input_path)

    for col in features:
        if col not in df.columns:
            logging.warning(f"Feature {col} missing in production data. Fallback to trend.")
            df[col] = df['trend_pred']
        else:
            df[col] = df[col].fillna(df['trend_pred'])

    # Predict final yield directly using exactly what the analyzer expects
    df['Super_Ensemble_pred'] = meta_model.predict(df[features])

    # Assign Predicted_Best_Model
    base_preds = df[features].values
    ens_preds = df['Super_Ensemble_pred'].values.reshape(-1, 1)
    closest_idx = np.argmin(np.abs(base_preds - ens_preds), axis=1)
    df['Predicted_Best_Model'] = [features[i] for i in closest_idx]

    # Save Output
    final_df = df[['year', 'district_no', 'kreisYield', 'trend_pred', 'Super_Ensemble_pred', 'Predicted_Best_Model']]
    out_path = OUTPUT_DIR / FINAL_FORECAST_FILENAME
    final_df.to_csv(out_path, index=False)
    logging.info(f"\n✓ Saved production forecast to {out_path}")


if __name__ == '__main__':
    generate_forecast()