# File: src/models/train_final_quantile_model.py
# FINAL VERSION: Removes all heuristics (sample_weights) and reverts to a
#                single, robust hyperparameter set. Relies purely on the
#                quantile objective and a proper CQR calibration step.
# Refactored to use central configuration from src.config

import pandas as pd
from xgboost import XGBRegressor
import os
import joblib
import warnings
import numpy as np
from pathlib import Path
import sys

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src import config

warnings.filterwarnings("ignore")

# Use the XGBOOST_TRAINING_CONFIG dictionary from the central config file
CONFIG = config.XGBOOST_TRAINING_CONFIG

def train_and_save_quantile_models():
    """Trains quantile models using a unified, robust approach."""
    print("--- Starting Final Residual Fitting Pipeline (FINAL - No Heuristics) ---")

    df = pd.read_csv(CONFIG['DATA_PATH'])
    df.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df['forecast_residual'] = df['kreisYield'] - df['stage1_forecast']
    df.dropna(subset=['stage1_forecast', 'forecast_residual'], inplace=True)
    df.dropna(subset=CONFIG['FEATURE_COLS'], inplace=True)

    X_train = df[CONFIG['FEATURE_COLS']]
    y_train = df['forecast_residual']
    print(f"\nTraining on {len(X_train)} samples to predict the forecast residuals.")

    os.makedirs(CONFIG['MODEL_OUTPUT_DIR'], exist_ok=True)

    for name, alpha in CONFIG['QUANTILES'].items():
        print(f"\n--- Training {name.upper()} Residual Model (Quantile: {alpha}) ---")

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            **CONFIG['BEST_PARAMS']
        )

        model.fit(X_train, y_train)

        model_path = os.path.join(CONFIG['MODEL_OUTPUT_DIR'], f'final_quantile_model_{name}.joblib')
        joblib.dump(model, model_path)
        print(f"✅ {name.upper()} model saved to {model_path}")

    print("\n--- All Residual Models Trained and Saved Successfully ---")


if __name__ == "__main__":
    train_and_save_quantile_models()
