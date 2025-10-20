# File: tune_model_robust.py
# Description: A hyperparameter tuning script that uses a rolling-origin
#              backtest for validation within each Optuna trial. This ensures
#              that the hyperparameters are optimized for generalizability over time,
#              aligning the tuning process with the final backtesting evaluation.

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import warnings
import optuna
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
N_TRIALS = 75  # Reduced trials because each is now much more computationally expensive.
STUDY_NAME = "xgb_yield_prediction_robust_v3"

# Define the validation period for the backtest inside each trial.
# This should be a representative multi-year period.
VALIDATION_START_YEAR = 2007
VALIDATION_END_YEAR = 2014  # Same period as the old validation set, but now used in a rolling fashion.


def load_and_prepare_data():
    """
    Loads the feature data and applies the correct, leak-proof causal detrending once.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {file_path}. Please run the feature engineering script first.")
        return None

    print("\n--- Applying Causal (Trailing Mean) Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)

    # Use a trailing (causal) window. shift(1) makes it a pure lookback.
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    # Handle NaNs created by the shift and rolling window using ONLY past data (forward fill)
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.ffill())
    # For districts with few data points at the start, fill remaining NaNs with the first valid trend value
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x
    )
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    # Get the list of features (must be done after loading)
    # The list is copied from your base_model.py to ensure consistency
    feature_cols = [
        'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
        'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
        'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
        'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
        'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
        'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast', 'spring_temp_prob_warm_forecast',
        'spring_precip_prob_wet_forecast', 'spring_solar_rad_prob_wet_forecast', 'spring_evaporation_prob_wet_forecast',
        'spring_runoff_prob_wet_forecast', 'spring_soil_temp_l1_prob_warm_forecast',
        'spring_snowfall_prob_wet_forecast',
        'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast', 'summer_solar_rad_prob_wet_forecast',
        'summer_evaporation_prob_wet_forecast', 'summer_runoff_prob_wet_forecast',
        'summer_soil_temp_l1_prob_warm_forecast',
        'summer_snowfall_prob_wet_forecast', 'lat', 'lon', 'avg_elevation', 'avg_slope', 'avg_bdod_0_30cm',
        'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'avg_bdod_0_100cm',
        'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm', 'avg_phh2o_0_100cm', 'winter_cropland_ndvi_mean',
        'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days', 'national_avg_yield_lag1', 'producer_price_index_lag1',
        'seed_price_index_lag1', 'energy_price_index_lag1', 'fertilizer_price_index_lag1',
        'plant_protection_price_index_lag1', 'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
        'producer_price_index_lag1_anomaly', 'seed_price_index_lag1_anomaly', 'energy_price_index_lag1_anomaly',
        'fertilizer_price_index_lag1_anomaly', 'plant_protection_price_index_lag1_anomaly',
        'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme', 'is_summer_forecast_dry',
        'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip', 'antecedent_gdd_sum_anomaly_sq',
        'summer_heat_x_profit_margin', 'summer_precip_x_input_costs', 'spring_temp_prob_warm_forecast_sq',
        'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq', 'summer_precip_prob_wet_forecast_sq'
    ]

    print("Data loaded and prepared successfully.")
    return df, feature_cols


def objective(trial, df, feature_cols):
    """
    The objective function for Optuna. For each trial, it performs a full
    rolling-origin backtest on the validation period and returns the overall RMSE.
    """
    # Define the hyperparameter search space with constraints to prevent overfitting
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 5),  # Constrained to prevent overfitting
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': -1
    }

    all_predictions = []

    # --- ROBUST BACKTESTING VALIDATION ---
    # This loop mimics the logic of your backtesting.py script
    for year_to_predict in range(VALIDATION_START_YEAR, VALIDATION_END_YEAR + 1):
        train_df = df[df['year'] < year_to_predict]
        test_df = df[df['year'] == year_to_predict]

        if test_df.empty or train_df.empty:
            continue

        X_train, y_train = train_df[feature_cols], train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        # A new model is instantiated and trained for each fold of the backtest
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        predicted_detrended = model.predict(X_test)
        final_predictions = predicted_detrended + test_df['yield_trend']

        fold_results = test_df[['kreisYield']].copy()
        fold_results['predicted_yield'] = final_predictions
        all_predictions.append(fold_results)

    if not all_predictions:
        return float('inf')  # Return a very high error if no predictions could be made

    # Combine results from all years and calculate a single performance metric
    results_df = pd.concat(all_predictions)
    rmse = np.sqrt(mean_squared_error(results_df['kreisYield'], results_df['predicted_yield']))

    return rmse


if __name__ == "__main__":
    data, feature_cols = load_and_prepare_data()

    if data is not None:
        # Create an Optuna study object. We want to 'minimize' the backtested RMSE.
        study = optuna.create_study(direction='minimize', study_name=STUDY_NAME)

        print(f"\n🚀 Starting ROBUST hyperparameter tuning with {N_TRIALS} trials...")
        print(f"   Each trial will run a backtest from {VALIDATION_START_YEAR} to {VALIDATION_END_YEAR}.")
        print("   This will take significantly longer than the previous script. Please be patient.")

        # Use a lambda function to pass the dataframe and feature columns to the objective
        study.optimize(
            lambda trial: objective(trial, data, feature_cols),
            n_trials=N_TRIALS,
            show_progress_bar=True
        )

        print("\n🎉 Tuning finished!")
        print(f"  Best trial number: {study.best_trial.number}")
        print(f"  Best validation backtest RMSE: {study.best_value:.4f}")

        print("\n📋 Best Hyperparameters Found:")
        best_params = study.best_params
        for key, value in best_params.items():
            if isinstance(value, float):
                print(f"    '{key}': {value:.6f},")
            else:
                print(f"    '{key}': {value},")

        print(
            "\n➡️ Next Step: Copy these parameters into 'src/models/base_model.py' and run your backtesting.py script to see the final performance.")