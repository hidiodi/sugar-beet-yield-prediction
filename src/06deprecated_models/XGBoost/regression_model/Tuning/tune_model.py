# File: src/02_models/XGBoost/regression_model/Tuning/tune_model.py
# REFACTORED (v5): XGBoost 2.0 Compatibility Fix
# - Removed early_stopping_rounds from .fit() to prevent TypeError.
# - Retains robust feature selection.

import optuna
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import sys
import logging
import warnings

# --- Setup ---
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def load_data():
    path = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
    if not path.exists():
        logging.error(f"CRITICAL: Data not found at {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    target_col = 'kreisYield'

    if target_col not in df.columns:
        logging.error(f"CRITICAL: Target '{target_col}' not found in CSV!")
        sys.exit(1)

    df = df.dropna(subset=[target_col])
    return df, target_col


def get_valid_features_and_constraints(df):
    config_features = config.XGBOOST_TRAINING_CONFIG['FEATURE_COLS']
    config_constraints = config.XGBOOST_TRAINING_CONFIG.get('MONOTONE_CONSTRAINTS', {})

    valid_features = []
    valid_constraints_list = []
    missing = []

    for feat in config_features:
        if feat in df.columns:
            valid_features.append(feat)
            c = config_constraints.get(feat, 0)
            valid_constraints_list.append(c)
        else:
            missing.append(feat)

    if missing:
        logging.warning(f"⚠️  TUNER DROPPING {len(missing)} MISSING FEATURES: {missing}")

    return valid_features, tuple(valid_constraints_list)


def objective(trial):
    df, target_col = load_data()
    features, constraints = get_valid_features_and_constraints(df)

    val_years = [2014, 2015, 2016, 2017, 2018, 2019]
    train_df = df[~df['year'].isin(val_years)]
    val_df = df[df['year'].isin(val_years)]

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_val = val_df[features]
    y_val = val_df[target_col]

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
        'gamma': trial.suggest_float('gamma', 0.1, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'early_stopping_rounds': 50  # Moved to constructor for XGBoost 2.0+ compliance
    }

    # Initialize Model
    # Note: 'early_stopping_rounds' in constructor requires eval_set in fit()
    model = XGBRegressor(
        objective='reg:absoluteerror',
        n_jobs=-1,
        random_state=42,
        monotone_constraints=constraints,
        **params
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    return mae


def main():
    logging.info("--- Starting Crash-Proof Tuning Sprint (v5) ---")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    logging.info("--- Tuning Complete ---")
    logging.info(f"Best MAE on Validation Set: {study.best_value:.4f}")
    logging.info("Best Params:")
    print(study.best_params)
    logging.info("\n>>> COPY THESE PARAMS TO 'BEST_PARAMS_MEDIAN' IN CONFIG.PY <<<")


if __name__ == "__main__":
    main()