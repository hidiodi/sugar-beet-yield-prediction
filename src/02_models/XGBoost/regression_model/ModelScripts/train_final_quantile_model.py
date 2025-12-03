# File: src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py
# REFACTORED (v16.0): Naive Baseline for Early Years (No Leaks)

import pandas as pd
import numpy as np
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


def fill_missing_trend_causally(df, trend_col, yield_col):
    """
    Fills missing trends in early years using an Expanding Mean of previous yields.
    Strictly Causal: Baseline for Year X is mean(Yields < X).
    """
    df_clean = df.copy()

    # Identify rows where Trend is missing
    missing_mask = df_clean[trend_col].isna()
    if missing_mask.sum() == 0:
        return df_clean

    logging.info(f"⚡ Filling {missing_mask.sum()} missing trends using Expanding Mean (Causal)...")

    # Calculate Expanding Mean per District (Shifted by 1 to avoid using current year)
    # Min periods=1 ensures we get a value as soon as we have 1 historical year
    naive_trend = df_clean.sort_values('year').groupby('district_no')[yield_col] \
        .expanding().mean().shift(1).reset_index(level=0, drop=True)

    # Fill only the missing values
    df_clean.loc[missing_mask, trend_col] = naive_trend.loc[missing_mask]

    # If first year is still NaN (because no history), drop it (can't be helped)
    remaining_nan = df_clean[trend_col].isna().sum()
    if remaining_nan > 0:
        logging.warning(f"  Dropping {remaining_nan} rows (First year of history, no baseline possible).")
        df_clean = df_clean.dropna(subset=[trend_col])

    return df_clean


def train_and_save_models(train_config):
    logging.info("--- Starting Hybrid Training (Causal Baseline Mode) ---")

    data_path = train_config['DATA_PATH']
    if not data_path.exists(): sys.exit(1)
    df = pd.read_csv(data_path)

    baseline_col = 'stat_trend_forecast'

    # 1. Fill Gaps Causally
    df = fill_missing_trend_causally(df, baseline_col, 'kreisYield')

    # 2. Define Target
    target_col = 'trend_residual'
    df[target_col] = df['kreisYield'] - df[baseline_col]

    feature_cols = train_config['FEATURE_COLS']
    valid_features = [c for c in feature_cols if c in df.columns]

    # 3. Filter Targets ONLY
    initial_len = len(df)
    df_train = df.dropna(subset=[target_col])

    logging.info(f"Data Retention: {len(df_train)}/{initial_len} rows used for training.")

    X_train = df_train[valid_features]
    y_train = df_train[target_col]

    # 4. Train
    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} model...")
        params = train_config[f'BEST_PARAMS_{name.upper()}']

        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile, **params)
        model.fit(X_train, y_train)

        out_path = train_config[f'{name.upper()}_MODEL_PATH']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_path)

    logging.info("--- Hybrid Training Complete ---")


if __name__ == "__main__":
    train_and_save_models(TRAIN_CONFIG)