# File: src/models/train_final_quantile_model.py
# FINAL VERSION (Corrected): Fixes KeyError from renaming and removes
#                            target leakage by dynamically creating a
#                            clean feature set for residual prediction.

import pandas as pd
from xgboost import XGBRegressor
import os
import joblib
import warnings
from pathlib import Path
import sys

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from src import config

warnings.filterwarnings("ignore")

# Use the XGBOOST_TRAINING_CONFIG dictionary from the central config file
CONFIG = config.XGBOOST_TRAINING_CONFIG


def train_and_save_quantile_models():
    """Trains quantile models using a unified, robust approach."""
    print("--- Starting Final Residual Fitting Pipeline (FINAL - No Heuristics) ---")

    df = pd.read_csv(CONFIG['DATA_PATH'])
    # 1. Calculate the residual using the correct, consistent column name.
    #    DO NOT rename the column.
    wofost_col = 'wofost_forecast_yield_fresh_dt'
    df['forecast_residual'] = df['kreisYield'] - df[wofost_col]
    df.dropna(subset=[wofost_col, 'forecast_residual'], inplace=True)

    # 2. Create a clean list of predictors.
    #    This is critical to prevent target leakage. We remove the WOFOST forecast
    #    itself from the list of features used to predict the WOFOST residual.
    #    Also remove any other non-predictor columns that might be in the master list.
    cols_to_exclude = [wofost_col, 'stage1_forecast', 'kreisYield']

    # Check for features that were created in the feature builder but might be missing
    # in the FEATURE_COLS list from the config, or vice versa.
    actual_training_features = [
        col for col in CONFIG['FEATURE_COLS']
        if col in df.columns and col not in cols_to_exclude
    ]

    missing_features = [col for col in CONFIG['FEATURE_COLS'] if col not in df.columns]
    if missing_features:
        print(
            f"WARNING: The following features from config were not found in the dataset and will be ignored: {missing_features}")

    # 3. Drop NaNs based ONLY on the columns that will actually be used for training.
    df.dropna(subset=actual_training_features, inplace=True)
    X_train = df[actual_training_features]
    y_train = df['forecast_residual']

    print(f"\nTraining on {len(X_train)} samples to predict the forecast residuals.")
    print(f"Using {len(actual_training_features)} features.")

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