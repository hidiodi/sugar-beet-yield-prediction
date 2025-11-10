# File: src/02_models/XGBoost/regression_model/ModelScripts/train_standalone_xgb_model.py
# Description: (OVERRIDDEN V2) Trains the standalone model. This script now includes
#              the data detrending logic, making it fully self-contained.

import pandas as pd
from xgboost import XGBRegressor
import joblib
import sys
from pathlib import Path
import logging
import numpy as np
from sklearn.linear_model import LinearRegression

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TRAIN_CONFIG = config.STANDALONE_XGB_CONFIG


def detrend_yield_by_district(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a linear trend for each district and creates a detrended yield column."""
    detrended_dfs = []
    for district_id, group in df.groupby('district_no'):
        group = group.sort_values('year').copy()
        # Require at least 5 years of data to fit a stable trend
        if len(group) < 5 or group['kreisYield'].isnull().any():
            group['yield_trend'] = np.nan
            group['yield_detrended'] = np.nan
        else:
            X = group[['year']]
            y = group['kreisYield']
            model = LinearRegression()
            model.fit(X, y)
            group['yield_trend'] = model.predict(X)
            group['yield_detrended'] = group['kreisYield'] - group['yield_trend']
        detrended_dfs.append(group)
    return pd.concat(detrended_dfs, ignore_index=True)


def train_standalone_model(train_config):
    logging.info("--- Starting Detrended Standalone XGBoost Model Training ---")

    # 1. Load ORIGINAL feature data
    df_raw = pd.read_csv(train_config['DATA_PATH'])
    logging.info(f"Loaded {len(df_raw)} rows from {train_config['DATA_PATH']}.")

    # 2. Perform detrending ON-THE-FLY
    logging.info("Performing in-memory yield detrending...")
    df = detrend_yield_by_district(df_raw)

    feature_cols = [col for col in train_config['FEATURE_COLS'] if col in df.columns]
    target_col = train_config['TARGET_COL']

    df.dropna(subset=[target_col], inplace=True)
    logging.info(f"Detrending complete. Training on {len(df)} samples.")

    X_train = df[feature_cols]
    y_train = df[target_col]

    # 3. Train models on the detrended target
    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} model on target '{target_col}'...")
        params = train_config[f'BEST_PARAMS_{name.upper()}']
        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile, **params)
        model.fit(X_train, y_train)

        output_path = train_config[f'{name.upper()}_MODEL_PATH']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        logging.info(f"✓ {name.upper()} model saved to {output_path}")

    logging.info("--- All detrended standalone models have been trained successfully. ---")


if __name__ == "__main__":
    train_standalone_model(TRAIN_CONFIG)