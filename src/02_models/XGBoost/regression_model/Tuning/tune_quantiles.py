# File: src/02_models/XGBoost/regression_model/Tuning/tune_direct_model.py
# Description: A robust hyperparameter tuning script for the DIRECT XGBoost model.
#              - Optimizes each quantile model (lower, median, upper) independently.
#              - Uses a rolling-origin backtest for validation within each Optuna trial.
#              - Optimizes for pinball loss, the correct metric for quantile regression.
#              - Target variable is the actual yield ('kreisYield').
# VERSION: 1.0

import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import warnings
import optuna
import sys
from pathlib import Path

# --- Project Setup ---
warnings.filterwarnings("ignore")
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
    Loads data and validates the feature set against the config.
    The target variable is the direct yield ('kreisYield').
    """
    file_path = train_config['DATA_PATH']
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"FATAL: The input file was not found at {file_path}.")
        return None, None, None

    # Define target and get feature list from the training configuration
    target_col = 'kreisYield'
    feature_cols = train_config['FEATURE_COLS']

    # Validate that all configured features are present in the dataframe
    missing_in_df = set(feature_cols) - set(df.columns)
    if missing_in_df:
        print(f"FATAL: The following features from the config are MISSING from the data: {missing_in_df}")
        return None, None, None

    print("Data loaded and prepared successfully.")
    return df, feature_cols, target_col


def objective_quantile(trial, df, feature_cols, target_col, alpha, tune_config):
    """
    Generic objective function for Optuna. Trains a SINGLE quantile model using a robust
    rolling-origin backtest for validation and optimizes its pinball loss.
    """
    # Define the hyperparameter search space for this trial
    params = {
        'objective': 'reg:quantileerror',
        'quantile_alpha': alpha,
        'n_estimators': trial.suggest_int('n_estimators', 400, 2500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'random_state': 42,
        'n_jobs': -1
    }

    validation_start = tune_config['VALIDATION_START_YEAR']
    validation_end = tune_config['VALIDATION_END_YEAR']

    all_losses = []
    # --- ROBUST ROLLING-ORIGIN VALIDATION LOOP ---
    for year_to_predict in range(validation_start, validation_end + 1):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()

        # Clean data *after* splitting to prevent data leakage
        # Drop rows where critical features (like stat_trend) are missing
        critical_cols = ['stat_trend_forecast', target_col]
        train_df.dropna(subset=critical_cols, inplace=True)
        test_df.dropna(subset=critical_cols, inplace=True)

        if test_df.empty or train_df.empty:
            continue

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        loss = pinball_loss(y_test.values, y_pred, alpha)
        all_losses.append(loss)

    if not all_losses:
        return float('inf')  # Return a high error if validation fails

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

            print("\n" + "=" * 60)
            print(f"--- Starting ROBUST tuning for {name.upper()} model (alpha={alpha}) ---")
            print(f"--- Target: '{target_col}' ---")
            print(f"--- Study: '{study_name}' in database '{TUNE_CONFIG['STORAGE_DB_NAME']}' ---")
            print(f"--- Validation: Rolling backtest from {TUNE_CONFIG['VALIDATION_START_YEAR']} to {TUNE_CONFIG['VALIDATION_END_YEAR']} ---")
            print("=" * 60)

            study.optimize(
                lambda trial: objective_quantile(trial, data, feature_cols, target_col, alpha, TUNE_CONFIG),
                n_trials=TUNE_CONFIG['N_TRIALS_PER_MODEL'],
                show_progress_bar=True
            )
            all_best_params[name] = study.best_params

        print("\n\n" + "=" * 60)
        print("      ROBUST HYPERPARAMETER TUNING FINISHED!")
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

        print("\n\nUpdate your config.py with these new parameter sets and re-run the main pipeline.")