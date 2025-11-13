# File: src/02_models/XGBoost/regression_model/ModelScripts/train_standalone_xgb_model.py
# Description: (HYBRID MODEL V3.1) Trains the final hybrid model.
#              This version trains on the *actual yield* (kreisYield) and
#              uses the full feature set (including stat_trend_forecast).
#              All detrending logic has been REMOVED.
#              FIX 3.1: Corrected .UPPER() typo to .upper()

import pandas as pd
from xgboost import XGBRegressor
import joblib
import sys
from pathlib import Path
import logging

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Use the main XGB config, as this is now the main hybrid model
TRAIN_CONFIG = config.XGBOOST_TRAINING_CONFIG


def train_hybrid_model(train_config):
    logging.info("--- Starting Hybrid XGBoost Model Training (Target: kreisYield) ---")

    # 1. Load FINAL feature data
    try:
        df = pd.read_csv(train_config['DATA_PATH'])
        logging.info(f"Loaded {len(df)} rows from {train_config['DATA_PATH']}.")
    except FileNotFoundError:
        logging.error(f"FATAL: Input data not found at {train_config['DATA_PATH']}")
        sys.exit(1)

    # 2. Define features and target
    # Use the 'FEATURE_COLS' from the config
    feature_cols = [col for col in train_config['FEATURE_COLS'] if col in df.columns]

    # --- LOGIC CHANGE: Train on the REAL yield ---
    target_col = 'kreisYield'

    if target_col not in df.columns:
        logging.error(f"FATAL: Target column '{target_col}' not found in data.")
        sys.exit(1)

    # Drop rows where target or key features are missing
    # (e.g., spin-up period for stat_trend_forecast)
    critical_cols = [target_col, 'stat_trend_forecast']
    df.dropna(subset=critical_cols, inplace=True)

    logging.info(f"Training on {len(df)} complete samples.")

    X_train = df[feature_cols]
    y_train = df[target_col]

    # 3. Train models on the actual yield target
    for name, quantile in train_config['QUANTILES'].items():
        # --- FIX: Changed .UPPER() to .upper() ---
        logging.info(f"Training {name.upper()} model on target '{target_col}'...")

        params = train_config[f'BEST_PARAMS_{name.upper()}']
        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile, **params)
        model.fit(X_train, y_train)

        output_path = train_config[f'{name.upper()}_MODEL_PATH']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)

        # --- FIX: Changed .UPPER() to .upper() ---
        logging.info(f"✓ {name.upper()} model saved to {output_path}")

    logging.info("--- All hybrid models have been trained successfully. ---")


if __name__ == "__main__":
    train_hybrid_model(TRAIN_CONFIG)