# File: src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py
# REFACTORED (v20.0): The Champion Training Logic (Forecast Residual)

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
TRAIN_CONFIG = config.XGBOOST_TRAINING_CONFIG


def train_and_save_models(train_config):
    logging.info("--- Starting Champion Training (Forecast Residual Target) ---")

    try:
        df = pd.read_csv(train_config['DATA_PATH'])
    except Exception as e:
        logging.error(f"Failed to load data: {e}");
        sys.exit(1)

    # DEFINE TARGET: Yield - Stage1_Forecast (WOFOST/Trend Blend)
    # The 'stage1_forecast' column comes from the builder, derived from 'stat_trend_forecast'
    target_col = 'forecast_residual'
    df[target_col] = df['kreisYield'] - df['stage1_forecast']

    # Filter
    df.dropna(subset=[target_col], inplace=True)

    # Feature Selection
    feature_cols = train_config['FEATURE_COLS']
    valid_features = [c for c in feature_cols if c in df.columns]

    # Log Missing Features
    missing = set(feature_cols) - set(valid_features)
    if missing: logging.warning(f"⚠️ Missing Features: {missing}")

    X_train = df[valid_features]
    y_train = df[target_col]

    logging.info(f"Training on {len(X_train)} samples using {len(valid_features)} features.")

    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()}...")
        params = train_config[f'BEST_PARAMS_{name.upper()}']

        # XGBoost requires a tuple of constraints in the order of columns
        monotone_constraints = []
        constraints_dict = train_config.get('MONOTONE_CONSTRAINTS', {})

        for feature in valid_features:
            # Default to 0 (no constraint) if not specified
            constraint = constraints_dict.get(feature, 0)
            monotone_constraints.append(constraint)

        # Convert to tuple format for XGBoost
        # Note: sklearn API uses 'monotone_constraints' parameter which can take a dict or string
        # But passing the tuple to 'monotone_constraints' is the most robust way across versions.

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=quantile,
            monotone_constraints=tuple(monotone_constraints),
            **params
        )
        model.fit(X_train, y_train)

        out_path = train_config[f'{name.upper()}_MODEL_PATH']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_path)

    logging.info("--- Champion Training Complete ---")


if __name__ == "__main__":
    train_and_save_models(TRAIN_CONFIG)