# File: src/models/train_hybrid_model.py
# Description: Trains and saves the two-stage hybrid model.
# Stage 1: Predicts the median detrended yield.
# Stage 2: Predicts the absolute error of the Stage 1 model.

import pandas as pd
from xgboost import XGBRegressor
import os
import warnings
import joblib
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# --- Configuration ---
# Define paths for the two new models
MODEL_MEDIAN_PATH = os.path.join('src/models', 'hybrid_model_stage1_median.joblib')
MODEL_ERROR_PATH = os.path.join('src/models', 'hybrid_model_stage2_error.joblib')

DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
TRAIN_END_YEAR = 2014  # Use the same split as your static test for consistency

# Use the same hyperparameters for both models for consistency
BEST_PARAMS = {
    'colsample_bytree': 0.8223306320976561,
    'learning_rate': 0.020282652208788696,
    'max_depth': 4,
    'n_estimators': 448,
    'subsample': 0.8049549320516778
}
feature_cols = [  # Ensure this matches your final feature set
    'avg_elevation', 'avg_soil_pawc', 'lon', 'lat', 'profit_margin_proxy_lag1',
    'cost_of_inputs_lag1', 'producer_price_index_lag1_anomaly',
    'seed_price_index_lag1_anomaly', 'energy_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly', 'plant_protection_price_index_lag1_anomaly',
    'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast',
    'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast',
    'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',
    'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
    'spring_temp_anomaly_forecast_sq', 'summer_temp_anomaly_forecast_sq',
    'spring_precip_anomaly_forecast_sq', 'summer_precip_anomaly_forecast_sq'
]
target_col = 'kreisYield'
detrended_target_col = 'kreisYield_detrended'


def main():
    """Trains and saves both stages of the hybrid model."""
    print("--- Starting Hybrid Model Training Pipeline ---")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}.")
        return

    # --- 1. Causal Detrending (Hardened against data leakage) ---
    print("\n--- Applying Causal (Trailing Mean) Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')[target_col].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x)
    df.dropna(subset=['yield_trend'], inplace=True)
    df[detrended_target_col] = df[target_col] - df['yield_trend']
    print(" -> Detrending complete.")

    # --- 2. Split Data for Training ---
    train_df = df[df['year'] <= TRAIN_END_YEAR].copy()
    X_train = train_df[feature_cols]
    y_train_detrended = train_df[detrended_target_col]
    print(f"\nTraining data defined as all years up to {TRAIN_END_YEAR}. Training set size: {len(X_train)} samples.")

    # --- 3. Train and Save Stage 1: Median Model ---
    print("\n--- Training Stage 1: Median Prediction Model ---")
    model_median = XGBRegressor(objective='reg:squarederror', **BEST_PARAMS, random_state=42, n_jobs=-1)
    model_median.fit(X_train, y_train_detrended)

    os.makedirs(os.path.dirname(MODEL_MEDIAN_PATH), exist_ok=True)
    joblib.dump(model_median, MODEL_MEDIAN_PATH)
    print(f"Stage 1 Median Model saved to {MODEL_MEDIAN_PATH}")

    # --- 4. Generate Targets for Stage 2 Model ---
    print("\n--- Generating Targets for Stage 2: Error Model ---")
    # Predict on the training data to get the errors the model makes
    predictions_on_train = model_median.predict(X_train)
    # The target is the absolute error
    y_train_error = abs(y_train_detrended - predictions_on_train)
    print(" -> Error targets generated successfully.")

    # --- 5. Train and Save Stage 2: Error Model ---
    print("\n--- Training Stage 2: Error Prediction Model ---")
    model_error = XGBRegressor(objective='reg:squarederror', **BEST_PARAMS, random_state=42, n_jobs=-1)
    model_error.fit(X_train, y_train_error)

    os.makedirs(os.path.dirname(MODEL_ERROR_PATH), exist_ok=True)
    joblib.dump(model_error, MODEL_ERROR_PATH)
    print(f"Stage 2 Error Model saved to {MODEL_ERROR_PATH}")
    print("\n--- Hybrid Model Training Complete ---")


if __name__ == "__main__":
    main()