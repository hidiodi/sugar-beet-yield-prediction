# File: src/models/base_model.py
# Description: MODIFIED to use a time-based split for validation, holding out the last 5 years of data.

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion.joblib')


def train_and_validate_with_holdout():
    """
    Loads pre-season data, splits it into a training set (all data except the last 5 years)
    and a validation set (the last 5 years), trains the champion XGBoost model,
    evaluates it, and then retrains the final model on all available training data.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    # --- This is our champion feature set ---
    feature_cols = [
        #'district_no',
        #'year',  # Note: 'year' as a direct feature can introduce time trends, be mindful.
        'precip_total_peak_growth',
        'temp_mean_peak_growth',
        'heat_stress_days_peak_growth',
        'solar_rad_peak_growth',
        'DTR_accumulation_phase',
        'temp_min_peak_growth',
        'temp_max_peak_growth',
        'spring_freezing_days',
        'spring_temp_anomaly_hybrid',
        'spring_precip_anomaly_hybrid',
        'summer_temp_anomaly_hybrid',
        'summer_precip_anomaly_hybrid',
        'avg_elevation',
        'avg_soil_pawc',
        #'producer_price_index',  # Contemporaneous feature
        #'energy_price_index',
        #'fertilizer_price_index',
        'lon',
        'lat',
        'winter_temp_anomaly',
        'winter_precip_anomaly',
        #'national_avg_yield',  # Contemporaneous feature
        'national_avg_yield_lag1',
        'producer_price_index_lag1'
    ]
    target_col = 'kreisYield'

    # --- Data Integrity Check ---
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from the input file: {missing_cols}")
        return

    # --- Time-Based Split ---
    last_year = df['year'].max()
    validation_start_year = last_year - 5

    print(f"--- Using Last 5 Years for Validation ---")
    print(f"Training data will be from years before {validation_start_year + 1}")
    print(f"Validation data will be from years {validation_start_year + 1} to {last_year}")

    train_df = df[df['year'] <= validation_start_year]
    validation_df = df[df['year'] > validation_start_year]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_validation = validation_df[feature_cols]
    y_validation = validation_df[target_col]

    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_validation)} samples")

    # --- Train and Evaluate the XGBoost Model ---
    xgb = XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.03,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_validation)

    r2 = r2_score(y_validation, y_pred_xgb)
    rmse = np.sqrt(mean_squared_error(y_validation, y_pred_xgb))

    print("\n--- Validation Performance ---")
    print(f"  R-squared (R2): {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- Final Model Training on the defined training data ---
    print("\n--- Training Final Model on Data Before the Holdout Period for Deployment ---")
    final_model = XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.03,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    final_model.fit(X_train, y_train)  # Fitting on the training data only

    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(final_model, MODEL_PATH)
        print(
            f"\n✅ Final XGBoost model successfully trained on data up to {validation_start_year} and saved to {MODEL_PATH}")
    except Exception as e:
        print(f"\n❌ Warning: Could not save the final model. Error: {e}")


if __name__ == "__main__":
    train_and_validate_with_holdout()