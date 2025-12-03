# File: src/02_models/XGBoost/regression_model/ModelScripts/train_standalone_xgb_model.py
# REFACTORED (v19.0): End-to-End Prediction (Raw Yield Target)
# Strategy: Train on 1980-2024. Use Trend as a feature, not a baseline.

import pandas as pd
from xgboost import XGBRegressor
import joblib
import sys
from pathlib import Path
import logging

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TRAIN_CONFIG = config.STANDALONE_XGB_CONFIG


def train_standalone_model(train_config):
    logging.info("--- Starting Standalone Training (Raw Yield Target) ---")

    try:
        df = pd.read_csv(train_config['DATA_PATH'])
    except FileNotFoundError:
        logging.error("Data file not found.")
        sys.exit(1)

    target_col = train_config['TARGET_COL']  # 'kreisYield'
    feature_cols = train_config['FEATURE_COLS']

    # 1. MAXIMIZE DATA: Filter only where Target is missing
    # We allow 'stat_trend_forecast' to be NaN (XGBoost handles missing features)
    initial_len = len(df)
    df_train = df.dropna(subset=[target_col])

    logging.info(f"Target: {target_col}")
    logging.info(f"Data Retention: {len(df_train)}/{initial_len} rows used for training.")
    logging.info(f"Note: 1980s data included. Missing features handled by XGBoost.")

    # 2. Features
    valid_features = [c for c in feature_cols if c in df_train.columns]
    X_train = df_train[valid_features]
    y_train = df_train[target_col]

    # 3. Train
    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} model...")
        params = train_config[f'BEST_PARAMS_{name.upper()}']

        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile, **params)
        model.fit(X_train, y_train)

        out_path = train_config[f'{name.upper()}_MODEL_PATH']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_path)

    logging.info("--- Standalone Training Complete ---")


if __name__ == "__main__":
    train_standalone_model(TRAIN_CONFIG)