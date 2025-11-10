# File: src/models/tune_xgboost_for_intervals.py
# Description: A hyperparameter tuning script that optimizes each quantile model (lower,
#              median, upper) independently to find the best possible parameters for each task.
#              This script is fully integrated with the project's config file.
#
# REVISED VERSION v8: Implements separate tuning for each quantile model.

import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import warnings
import optuna
import sys
from pathlib import Path

# --- Project Setup ---
warnings.filterwarnings("ignore")
# Ensure the project root is in the Python path for module imports
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

# --- Load Configuration ---
TUNE_CONFIG = config.XGBOOST_TUNING_CONFIG
TRAIN_CONFIG = config.XGBOOST_TRAINING_CONFIG


# --- METRIC FUNCTION ---
def pinball_loss(y_true, y_pred, alpha):
    """Calculates the pinball loss, the correct metric for quantile regression."""
    delta = y_true - y_pred
    loss = np.maximum(alpha * delta, (alpha - 1) * delta)
    return np.mean(loss)


def load_and_prepare_data(train_config):
    """
    Loads data using paths from the config file and prepares the target variable (forecast residuals).
    It also validates the feature set against the config.
    """
    file_path = train_config['DATA_PATH']
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The input file was not found at {file_path}. Please run the feature engineering script first.")
        return None, None, None

    # Use the same residual-fitting approach as the final model
    df.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df['forecast_residual'] = df['kreisYield'] - df['stage1_forecast']
    df.dropna(subset=['stage1_forecast', 'forecast_residual'], inplace=True)

    # Get feature list directly from the training configuration
    feature_cols = train_config['FEATURE_COLS']

    # Validate that all configured features are present in the dataframe
    missing_in_df = set(feature_cols) - set(df.columns)
    if missing_in_df:
        print(
            f"CRITICAL WARNING: The following features from the config are MISSING from the input data: {missing_in_df}")
        feature_cols = [col for col in feature_cols if col in df.columns]
        print(f"Proceeding with {len(feature_cols)} available features.")

    print("Data loaded and prepared successfully.")
    return df, feature_cols, 'forecast_residual'


def objective_quantile(trial, df, feature_cols, target_col, alpha, tune_config):
    """
    Generic objective function for Optuna. Trains a SINGLE quantile model and optimizes
    its pinball loss.
    """
    # Define the hyperparameter search space for this trial
    params = {
        'objective': 'reg:quantileerror',
        'quantile_alpha': alpha,
        'n_estimators': trial.suggest_int('n_estimators', 400, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.5, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': -1
    }

    # Get validation years from config
    validation_start = tune_config['VALIDATION_START_YEAR']
    validation_end = tune_config['VALIDATION_END_YEAR']

    all_losses = []
    for year_to_predict in range(validation_start, validation_end + 1):
        train_df = df[df['year'] < year_to_predict]
        test_df = df[df['year'] == year_to_predict]
        if test_df.empty or train_df.empty:
            continue

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Predict the residual
        y_pred_residual = model.predict(X_test)

        # We evaluate the loss on the RESIDUAL, as that's what the model is trained on.
        loss = pinball_loss(y_test, y_pred_residual, alpha)
        all_losses.append(loss)

    if not all_losses:
        return float('inf')

    return np.mean(all_losses)


if __name__ == "__main__":
    data, feature_cols, target_col = load_and_prepare_data(TRAIN_CONFIG)

    if data is not None and feature_cols:
        quantiles = TRAIN_CONFIG['QUANTILES']
        storage_name = f"sqlite:///{TUNE_CONFIG['STORAGE_DB_NAME']}"
        all_best_params = {}

        for name, alpha in quantiles.items():
            study_name = TUNE_CONFIG['STUDY_NAMES'][name]
            study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name,
                                        load_if_exists=True)

            print("\n" + "=" * 50)
            print(f"--- Starting SEPARATE hyperparameter tuning for {name.upper()} model (alpha={alpha}) ---")
            print(f"Using study: {study_name} in database {TUNE_CONFIG['STORAGE_DB_NAME']}")
            print("=" * 50)

            study.optimize(
                lambda trial: objective_quantile(trial, data, feature_cols, target_col, alpha, TUNE_CONFIG),
                n_trials=TUNE_CONFIG['N_TRIALS_PER_MODEL'],
                show_progress_bar=True
            )
            all_best_params[name] = study.best_params

        print("\n\n" + "=" * 60)
        print("      SEPARATE QUANTILE MODEL TUNING FINISHED!")
        print("=" * 60)

        for name, params in all_best_params.items():
            print(f"\n--- Best Hyperparameters for the {name.upper()} Model ---")
            print(f"BEST_PARAMS_{name.upper()} = {{")
            for key, value in params.items():
                if isinstance(value, str):
                    print(f"    '{key}': '{value}',")
                elif isinstance(value, float):
                    print(f"    '{key}': {value:.6f},")
                else:
                    print(f"    '{key}': {value},")
            print("    'random_state': 42,")
            print("    'n_jobs': -1")
            print("}")

        print("\n\nUpdate your training script to use these separate parameter sets for each respective model.")