# File: src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py
# Description: Trains the final set of quantile regression models using individually
#              tuned hyperparameters for each quantile.
# VERSION: 3.0 (Separate Hyperparameter Sets)

import pandas as pd
from xgboost import XGBRegressor
import joblib
import sys
from pathlib import Path
import logging

# --- Project Setup ---
# Add project root to the Python path to allow for module imports
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TRAIN_CONFIG = config.XGBOOST_TRAINING_CONFIG


def train_and_save_models(train_config):
    """
    Loads data, trains three separate quantile models with their own optimized
    hyperparameters, and saves the final model artifacts.
    """
    logging.info("--- Starting Final Quantile Model Training (Separate Hyperparameters) ---")

    # --- 1. Load and Prepare Data ---
    data_path = train_config['DATA_PATH']
    logging.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error(f"FATAL: Input data not found at {data_path}. Aborting.")
        sys.exit(1)

    # Prepare data consistently with the tuning script
    df.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df['forecast_residual'] = df['kreisYield'] - df['stage1_forecast']
    df.dropna(subset=['stage1_forecast', 'forecast_residual'], inplace=True)

    feature_cols = train_config['FEATURE_COLS']
    target_col = 'forecast_residual'

    # Final validation of features
    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        logging.error(f"FATAL: The following required features are missing from the dataset: {missing_features}")
        sys.exit(1)

    X_train = df[feature_cols]
    y_train = df[target_col]
    logging.info(f"Data prepared. Training on {len(X_train)} samples with {len(feature_cols)} features.")

    # --- 2. Load Hyperparameters from Config ---
    params_lower = train_config['BEST_PARAMS_LOWER']
    params_median = train_config['BEST_PARAMS_MEDIAN']
    params_upper = train_config['BEST_PARAMS_UPPER']
    quantiles = train_config['QUANTILES']

    model_configs = {
        'lower': {'alpha': quantiles['lower'], 'params': params_lower, 'path': train_config['LOWER_MODEL_PATH']},
        'median': {'alpha': quantiles['median'], 'params': params_median, 'path': train_config['MEDIAN_MODEL_PATH']},
        'upper': {'alpha': quantiles['upper'], 'params': params_upper, 'path': train_config['UPPER_MODEL_PATH']}
    }

    # --- 3. Train and Save Each Model ---
    for name, cfg in model_configs.items():
        logging.info(f"Training {name.upper()} model (alpha={cfg['alpha']})...")

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

    logging.info("--- All final models have been trained and saved successfully. ---")


if __name__ == "__main__":
    train_and_save_models(TRAIN_CONFIG)