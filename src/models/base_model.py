# File: src/models/train_final_model.py
# Description: MODIFIED to use a random split of districts within each year for evaluation.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join('src/models', 'final_xgb_model_random_split.joblib')


def train_with_random_split():
    """
    Loads pre-season data, performs a random split of districts within each year,
    evaluates models, and trains a final model on all data.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    feature_cols = [
        'district_no',
        'avg_elevation',
        'avg_soil_pawc',
        'winter_temp_anomaly',
        'winter_precip_anomaly',
        'national_avg_yield_lag1',
        'producer_price_index_lag1'
    ]
    target_col = 'kreisYield'

    # --- Random Split of Districts Within Each Year ---
    print("--- Performing random 80/20 split of districts WITHIN each year ---")

    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    # Loop through each year in the dataset
    for year in df['year'].unique():
        df_year = df[df['year'] == year]

        X_year = df_year[feature_cols]
        y_year = df_year[target_col]

        # Perform a standard random split on this year's data
        X_train_yr, X_test_yr, y_train_yr, y_test_yr = train_test_split(
            X_year, y_year, test_size=0.2, random_state=42
        )

        X_train_list.append(X_train_yr)
        X_test_list.append(X_test_yr)
        y_train_list.append(y_train_yr)
        y_test_list.append(y_test_yr)

    # Concatenate the splits from all years into final training and testing sets
    X_train = pd.concat(X_train_list)
    X_test = pd.concat(X_test_list)
    y_train = pd.concat(y_train_list)
    y_test = pd.concat(y_test_list)

    print(f"Total training samples: {len(X_train)}")
    print(f"Total testing samples: {len(X_test)}")

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train and Evaluate Ridge Regression ---
    print("\n--- Training Models ---")
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)

    # --- Train and Evaluate XGBoost ---
    xgb = XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.03,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # --- Print Results ---
    print("\n--- Model Performance (Random Split Evaluation) ---")
    print("\nModel: Ridge Regression")
    print(f"  R-squared (R2): {r2_score(y_test, y_pred_ridge):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.2f} dt/ha")

    print("\nModel: XGBoost")
    print(f"  R-squared (R2): {r2_score(y_test, y_pred_xgb):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.2f} dt/ha")

    # --- Final Model Training on ALL data ---
    print("\n--- Training Final Model on All Available Data for Deployment ---")
    final_model = XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.03,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    final_model.fit(df[feature_cols], df[target_col])

    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(final_model, MODEL_PATH)
        print(f"\n✅ Final XGBoost model successfully trained on all data and saved to {MODEL_PATH}")
    except Exception as e:
        print(f"\n❌ Warning: Could not save the final model. Error: {e}")


if __name__ == "__main__":
    train_with_random_split()