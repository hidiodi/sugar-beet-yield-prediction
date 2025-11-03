# File: src/models/tune_xgboost_for_intervals.py
# Description: An advanced hyperparameter tuning script that optimizes for the overall
#              quality of the prediction interval, not just individual quantiles.
#
# REVISED VERSION v6: The objective function now trains all three quantile models
#                     simultaneously and minimizes a combined metric of the Interval Score
#                     and the Median Pinball Loss to combat overconfidence.

import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import os
import warnings
import optuna

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
N_TRIALS = 100  # Increased trials for a more complex search space
STUDY_NAME = "xgb_yield_INTERVAL_tuning_v6"
VALIDATION_START_YEAR = 2007
VALIDATION_END_YEAR = 2014
QUANTILES = {'lower': 0.025, 'median': 0.5, 'upper': 0.975}


# --- METRIC FUNCTIONS ---
def pinball_loss(y_true, y_pred, alpha):
    """Calculates the pinball loss, the correct metric for quantile regression."""
    delta = y_true - y_pred
    loss = np.maximum(alpha * delta, (alpha - 1) * delta)
    return np.mean(loss)


def interval_score(y_true, lower, upper, alpha):
    """
    Calculates the Winkler Interval Score. A holistic metric for interval quality.
    Penalizes wide intervals and heavily penalizes intervals that miss the true value. Lower is better.
    """
    width = upper - lower
    # Penalty for when the true value is below the lower bound
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    # Penalty for when the true value is above the upper bound
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return np.mean(width + penalty_lower + penalty_upper)


def load_and_prepare_data():
    """Loads data and prepares the target variable (forecast residuals)."""
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: Please run the updated feature engineering script first.")
        return None, None, None

    # --- Use the same residual-fitting approach as the final model ---
    df.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df['forecast_residual'] = df['kreisYield'] - df['stage1_forecast']
    df.dropna(subset=['stage1_forecast', 'forecast_residual'], inplace=True)

    # Define the feature set (ensure this matches your training script)
    feature_cols = [
        'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
        # ... (Include the FULL list of features from your training script)
        'is_drought_high_clay_in_state_11'
    ]

    # For simplicity in this example, ensure all feature columns from your training script are listed here.
    # A robust way is to load them from a shared config file.

    # Quick check for missing features in the dataframe
    df_feature_cols = [col for col in feature_cols if col in df.columns]
    if len(df_feature_cols) != len(feature_cols):
        missing_in_df = set(feature_cols) - set(df.columns)
        print(f"Warning: The following features are defined but not in the dataframe: {missing_in_df}")

    print("Data loaded and prepared successfully.")
    return df, df_feature_cols, 'forecast_residual'


def objective(trial, df, feature_cols, target_col):
    """
    Objective function for Optuna. Trains all three models and optimizes a combined
    score for interval quality and median accuracy.
    """
    # These hyperparameters will be shared across all three models
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 1, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
        'random_state': 42,
        'n_jobs': -1
    }

    all_predictions = []
    for year_to_predict in range(VALIDATION_START_YEAR, VALIDATION_END_YEAR + 1):
        train_df = df[df['year'] < year_to_predict]
        test_df = df[df['year'] == year_to_predict]
        if test_df.empty or train_df.empty:
            continue

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test = test_df[feature_cols]

        # --- Train a model for each quantile using the same hyperparameters ---
        models = {}
        for name, alpha in QUANTILES.items():
            model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, **params)
            model.fit(X_train, y_train)
            models[name] = model

        # --- Generate predictions for all quantiles ---
        fold_results = test_df[['kreisYield', 'stage1_forecast']].copy()
        # Predict the residual and then add back the Stage 1 forecast
        fold_results['pred_lower'] = models['lower'].predict(X_test) + test_df['stage1_forecast']
        fold_results['pred_median'] = models['median'].predict(X_test) + test_df['stage1_forecast']
        fold_results['pred_upper'] = models['upper'].predict(X_test) + test_df['stage1_forecast']
        all_predictions.append(fold_results)

    if not all_predictions:
        return float('inf')

    results_df = pd.concat(all_predictions)

    # --- Calculate the final combined loss metric ---
    # 1. The quality of the interval
    alpha_for_score = 1 - (QUANTILES['upper'] - QUANTILES['lower'])
    score = interval_score(results_df['kreisYield'], results_df['pred_lower'], results_df['pred_upper'],
                           alpha_for_score)

    # 2. The accuracy of the median
    median_loss = pinball_loss(results_df['kreisYield'], results_df['pred_median'], QUANTILES['median'])

    # Return a combined score to balance both goals
    return score + median_loss


if __name__ == "__main__":
    data, feature_cols, target_col = load_and_prepare_data()

    if data is not None:
        storage_name = f"sqlite:///{STUDY_NAME}.db"
        study = optuna.create_study(direction='minimize', study_name=STUDY_NAME, storage=storage_name,
                                    load_if_exists=True)

        print(f"\n--- Starting unified hyperparameter tuning for INTERVAL QUALITY ---")
        print(f"Using study: {STUDY_NAME}")

        study.optimize(
            lambda trial: objective(trial, data, feature_cols, target_col),
            n_trials=N_TRIALS,
            show_progress_bar=True
        )

        print("\n\n" + "=" * 50)
        print("      INTERVAL TUNING FINISHED!")
        print("=" * 50)

        print("\nBest Hyperparameters for the Interval Model:")
        print(f"-> USE THIS DICTIONARY FOR ALL THREE (LOWER, MEDIAN, UPPER) MODELS <-\n")
        print(f"BEST_INTERVAL_PARAMS = {{")
        for key, value in study.best_params.items():
            if isinstance(value, str):
                print(f"    '{key}': '{value}',")
            elif isinstance(value, float):
                print(f"    '{key}': {value:.6f},")
            else:
                print(f"    '{key}': {value},")
        print("    'random_state': 42,")
        print("    'n_jobs': -1")
        print("}")
        print("\nCopy this dictionary into your final training script and replace 'BEST_PARAMS'.")