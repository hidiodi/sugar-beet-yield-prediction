# File: src/02_models/XGBoost/regression_model/Tuning/tune_quantiles.py
# Description: A robust hyperparameter tuning script for the HYBRID (Residual) XGBoost model.
#              - Optimizes each quantile model (lower, median, upper) independently.
#              - Uses a rolling-origin backtest for validation within each Optuna trial.
#              - Optimizes for pinball loss on the RESIDUALS.
#              - Target variable is (Actual Yield - Stage 1 Forecast).
#              - ROBUST: Auto-detects available features and ignores missing ones.
# VERSION: 3.0 (Robust Feature Loading)

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
    Loads data, calculates the residual target, and validates features.
    Target: 'forecast_residual' = 'kreisYield' - 'stage1_forecast'

    ROBUSTNESS:
    - If configured features are missing in the CSV, it WARNS but CONTINUES.
    - It strictly filters out rows where the Target cannot be calculated.
    """
    file_path = train_config['DATA_PATH']
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"FATAL: The input file was not found at {file_path}.")
        return None, None, None

    # --- HYBRID SPECIFIC: Calculate Residual Target ---
    if 'stage1_forecast' not in df.columns or 'kreisYield' not in df.columns:
        print("FATAL: Data missing 'kreisYield' or 'stage1_forecast'. Cannot calculate residual.")
        return None, None, None

    target_col = 'forecast_residual'
    df[target_col] = df['kreisYield'] - df['stage1_forecast']

    # Filter out rows where we can't calculate the target
    df.dropna(subset=[target_col], inplace=True)

    # Get feature list from the training configuration
    desired_features = train_config['FEATURE_COLS']

    # --- ROBUST FEATURE SELECTION (Soft Fail) ---
    # Intersection of Config and CSV
    available_features = [c for c in desired_features if c in df.columns]
    missing_features = set(desired_features) - set(available_features)

    if missing_features:
        print("\n" + "!" * 60)
        print(f"WARNING: {len(missing_features)} features from config are MISSING in the data.")
        print(f"Missing: {missing_features}")
        print("Proceeding with the available features only.")
        print("!" * 60 + "\n")

    if not available_features:
        print("FATAL: No valid features found in the dataset!")
        return None, None, None

    print(f"Data loaded. Target '{target_col}' calculated.")
    print(f"Training with {len(available_features)} features on {len(df)} rows.")

    return df, available_features, target_col


def objective_quantile(trial, df, feature_cols, target_col, alpha, tune_config):
    """
    Generic objective function for Optuna. Trains a SINGLE quantile model using a robust
    rolling-origin backtest for validation and optimizes its pinball loss on the RESIDUAL.
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
        # Time-based split to prevent leakage
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()

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
    # Load data and calculate the residual target
    # Uses the robust loader to avoid crashing on missing econ features
    data, feature_cols, target_col = load_and_prepare_data(TRAIN_CONFIG)

    if data is not None and feature_cols:
        quantiles = TRAIN_CONFIG['QUANTILES']
        # Ensure DB name is valid
        db_name = TUNE_CONFIG['STORAGE_DB_NAME']
        storage_name = f"sqlite:///{db_name}" if not db_name.startswith("sqlite:///") else db_name

        all_best_params = {}

        for name, alpha in quantiles.items():
            study_name = TUNE_CONFIG['STUDY_NAMES'][name]

            # Create or load study
            study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name,
                                        load_if_exists=True)

            print("\n" + "=" * 60)
            print(f"--- Starting HYBRID tuning for {name.upper()} model (alpha={alpha}) ---")
            print(f"--- Target: '{target_col}' (Residual) ---")
            print(f"--- Study: '{study_name}' in database '{db_name}' ---")
            print(f"--- Features: {len(feature_cols)} active features ---")
            print("=" * 60)

            study.optimize(
                lambda trial: objective_quantile(trial, data, feature_cols, target_col, alpha, TUNE_CONFIG),
                n_trials=TUNE_CONFIG['N_TRIALS_PER_MODEL'],
                show_progress_bar=True
            )
            all_best_params[name] = study.best_params

        print("\n\n" + "=" * 60)
        print("      HYBRID HYPERPARAMETER TUNING FINISHED!")
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

        print("\n\nUpdate XGBOOST_TRAINING_CONFIG in config.py with these new parameter sets.")