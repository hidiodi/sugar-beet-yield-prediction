# File: src/models/train_final_ngboost_model.py
# Description: Trains a definitive NGBoost model by fitting it on the RESIDUALS
#              of the primary time-series forecast. This version is refactored
#              to use the central config file and a robust, leak-proof feature set.

import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
import os
import joblib
import warnings
from pathlib import Path
import sys

# --- START OF REFACTOR ---

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]  # Adjust path depth as needed
sys.path.insert(0, str(project_root))

from src import config

warnings.filterwarnings("ignore")

# Use the XGBOOST_TRAINING_CONFIG dictionary from the central config file
# NGBoost will use the same data path and feature list as XGBoost.
CONFIG = config.XGBOOST_TRAINING_CONFIG

MODEL_OUTPUT_PATH = CONFIG['MODEL_OUTPUT_DIR'] / 'final_ngboost_model.joblib'


def train_and_save_ngboost_model():
    """Trains a single NGBoost model to predict the distribution of residuals."""
    print("--- Starting NGBoost Residual Fitting Pipeline ---")

    try:
        df = pd.read_csv(CONFIG['DATA_PATH'])
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {CONFIG['DATA_PATH']}.")
        return

    print("\n--- Calculating Forecast Residuals ---")
    # Use the consistent WOFOST column name
    wofost_col = 'wofost_forecast_yield_fresh_dt'
    df['forecast_residual'] = df['kreisYield'] - df[wofost_col]

    # Drop rows where essential data for the target is missing.
    df.dropna(subset=[wofost_col, 'forecast_residual', 'kreisYield'], inplace=True)
    print(" -> Target variable (forecast_residual) created.")

    # --- START OF FIX: Leak-proof feature selection ---
    # Dynamically create a clean list of predictors to prevent target leakage.
    cols_to_exclude = [wofost_col, 'stage1_forecast', 'kreisYield']

    actual_training_features = [
        col for col in CONFIG['FEATURE_COLS']
        if col in df.columns and col not in cols_to_exclude
    ]

    missing_features = [col for col in CONFIG['FEATURE_COLS'] if col not in df.columns]
    if missing_features:
        print(
            f"WARNING: The following features from config were not found in the dataset and will be ignored: {missing_features}")

    # NGBoost can handle NaNs in features, so we don't drop them from X.
    X = df[actual_training_features]
    y = df['forecast_residual']
    # --- END OF FIX ---

    print(f"\nTotal samples available: {len(X)}")

    # Create a time-series-aware validation set for early stopping.
    train_end_idx = int(len(X) * 0.85)
    X_train, y_train = X[:train_end_idx], y[:train_end_idx]
    X_val, y_val = X[train_end_idx:], y[train_end_idx:]
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    # Initialize the NGBoost model with a validation set for early stopping
    ngb_model = NGBRegressor(
        Dist=Normal,
        n_estimators=500,
        learning_rate=0.05,
        verbose=True,
        random_state=42,
        minibatch_frac=0.8
    )

    print("\n--- Training NGBoost Model with Validation Set ---")
    ngb_model.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=20)

    # Save the final model
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(ngb_model, MODEL_OUTPUT_PATH)
    print(f"\n✅ NGBoost model saved to {MODEL_OUTPUT_PATH}")
    print("\n--- NGBoost Model Trained and Saved Successfully ---")


if __name__ == "__main__":
    train_and_save_ngboost_model()