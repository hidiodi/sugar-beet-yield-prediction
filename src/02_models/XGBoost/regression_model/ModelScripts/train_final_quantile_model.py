# File: src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py
# Description: Trains a RESIDUAL model. Predicts the deviation from the statistical trend.
# VERSION: 4.0 (Stat-Trend Residual Model)

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

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TRAIN_CONFIG = config.XGBOOST_TRAINING_CONFIG


def train_and_save_models(train_config):
    """
    Loads data, trains three separate quantile models on the RESIDUAL of the
    statistical trend forecast, and saves the final model artifacts.
    """
    logging.info("--- Starting STAT-TREND RESIDUAL Quantile Model Training ---")

    # --- 1. Load and Prepare Data ---
    data_path = train_config['DATA_PATH']
    logging.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error(f"FATAL: Input data not found at {data_path}. Aborting.")
        sys.exit(1)

    # --- LOGIC CHANGE: Define the target as the residual from the 5-year rolling average ---
    baseline_feature = 'yield_trend'  # This will be our new baseline feature name
    target_col = 'trend_residual'
    logging.info(f"Calculating baseline '{baseline_feature}' as 5-year rolling average of kreisYield.")

    # Sort to ensure correct rolling window application
    df.sort_values(by=['district_no', 'year'], inplace=True)

    # Calculate the 5-year TRAILING average. shift(1) is crucial to prevent data leakage.
    df[baseline_feature] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=3).mean().shift(1)
    )

    logging.info(f"Calculating target variable '{target_col}' as (kreisYield - {baseline_feature}).")
    df[target_col] = df['kreisYield'] - df[baseline_feature]

    # Drop rows where the baseline or the target can't be calculated (e.g., early years)
    df.dropna(subset=[baseline_feature, target_col], inplace=True)

    # --- LOGIC CHANGE: Define the feature set for the residual model ---
    # Start with the master list of clean features from the config.
    base_feature_cols = train_config['FEATURE_COLS']

    # CRITICAL: DO NOT use the baseline as a feature to predict its own residual.
    # This would be a perfect data leak.
    if baseline_feature in base_feature_cols:
        residual_feature_cols = [col for col in base_feature_cols if col != baseline_feature]
        logging.info(f"Removed '{baseline_feature}' from feature list to prevent data leakage.")
    else:
        residual_feature_cols = base_feature_cols

    # Final validation of features
    missing_features = set(residual_feature_cols) - set(df.columns)
    if missing_features:
        logging.error(f"FATAL: The following required features are missing from the dataset: {missing_features}")
        sys.exit(1)

    X_train = df[residual_feature_cols]
    y_train = df[target_col]
    logging.info(f"Data prepared. Training on {len(X_train)} samples with {len(residual_feature_cols)} features.")

    # --- 2. Load Hyperparameters from Config (Unchanged) ---
    params_lower = train_config['BEST_PARAMS_LOWER']
    params_median = train_config['BEST_PARAMS_MEDIAN']
    params_upper = train_config['BEST_PARAMS_UPPER']
    quantiles = train_config['QUANTILES']

    model_configs = {
        'lower': {'alpha': quantiles['lower'], 'params': params_lower, 'path': train_config['LOWER_MODEL_PATH']},
        'median': {'alpha': quantiles['median'], 'params': params_median, 'path': train_config['MEDIAN_MODEL_PATH']},
        'upper': {'alpha': quantiles['upper'], 'params': params_upper, 'path': train_config['UPPER_MODEL_PATH']}
    }

    # --- 3. Train and Save Each Model (Unchanged) ---
    for name, cfg in model_configs.items():
        logging.info(f"Training {name.upper()} model on target '{target_col}' (alpha={cfg['alpha']})...")

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=cfg['alpha'],
            **cfg['params']
        )

        model.fit(X_train, y_train)
        logging.info(f"Training for {name.upper()} model complete.")

        # Save the trained model
        output_path = cfg['path']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        logging.info(f"✓ {name.upper()} model saved to {output_path}")

    logging.info("--- All RESIDUAL models have been trained and saved successfully. ---")


if __name__ == "__main__":
    train_and_save_models(TRAIN_CONFIG)