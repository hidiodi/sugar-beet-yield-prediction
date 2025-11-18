# File: src/02_models/XGBoost/regression_model/ModelScripts/train_standalone_xgb_model.py
# REFACTORED (v5.0): Standalone Model using Robust Risk Features
# Description:
#   Trains a standalone XGBoost model to predict the RESIDUAL from the Rolling Trend.
#   (Previously it predicted raw yield, but residual prediction is mathematically superior here).

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
TRAIN_CONFIG = config.STANDALONE_XGB_CONFIG


def train_hybrid_model(train_config):
    logging.info("--- Starting Standalone XGBoost Training (Risk-Based Strategy) ---")

    # 1. Load Data
    try:
        df = pd.read_csv(train_config['DATA_PATH'])
        logging.info(f"Loaded {len(df)} rows from {train_config['DATA_PATH']}.")
    except FileNotFoundError:
        logging.error(f"FATAL: Input data not found at {train_config['DATA_PATH']}")
        sys.exit(1)

    # 2. Target Engineering: Rolling Trend Residual
    # Even for the "Standalone" model, predicting the deviation from a rolling mean
    # is much easier than predicting raw yield levels (which shift due to tech/varieties).
    baseline_col = 'yield_rolling_trend'
    target_col = 'trend_residual'

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # 5-Year Rolling Average (Lagged to prevent leak)
    df[baseline_col] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=3).mean().shift(1)
    )

    # Calculate Target
    df[target_col] = df['kreisYield'] - df[baseline_col]

    # 3. Feature Selection
    # Use the same robust features as the main model
    feature_cols = [col for col in train_config['FEATURE_COLS'] if col in df.columns]

    # Drop rows where target or features are missing
    # We need 'stat_trend_forecast' because it's in the feature list
    df.dropna(subset=[target_col, baseline_col] + feature_cols, inplace=True)

    logging.info(f"Training on {len(df)} complete samples.")
    logging.info(f"Features: {feature_cols}")

    X_train = df[feature_cols]
    y_train = df[target_col]

    # 4. Train Models
    # We use the hyperparameters from the config (which now point to the regularized v5 params)
    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} model (alpha={quantile})...")

        # Retrieve params
        params = train_config[f'BEST_PARAMS_{name.upper()}']

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=quantile,
            **params
        )
        model.fit(X_train, y_train)

        output_path = train_config[f'{name.upper()}_MODEL_PATH']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)

        logging.info(f"✓ {name.upper()} model saved to {output_path}")

    logging.info("--- Standalone Models Trained Successfully. ---")


if __name__ == "__main__":
    train_hybrid_model(TRAIN_CONFIG)