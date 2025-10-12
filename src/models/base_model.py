# File: src/models/base_model.py
# Description: FINALIZED to use a robust 10-fold cross-validation to build confidence in the champion model's performance.

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion.joblib')


def perform_robust_cross_validation():
    """
    Loads pre-season data, performs a 10-fold cross-validation with different random splits
    to robustly evaluate the champion XGBoost model, and then trains one final model on all data.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    # --- This is our champion feature set ---
    feature_cols = [
        'district_no',
        'avg_elevation',
        'avg_soil_pawc',
        'winter_temp_anomaly',
        'winter_precip_anomaly',
        'national_avg_yield_lag1',
        'producer_price_index_lag1',
        'spring_temp_anomaly_hybrid',
        'spring_precip_anomaly_hybrid',
        'summer_temp_anomaly_hybrid',
        'summer_precip_anomaly_hybrid',
        # Unused economic features
        'fertilizer_price_index', 'energy_price_index',

        # --- CRITICAL: Remove all peak-growth (summer) weather features ---
        #'precip_total_peak_growth', 'temp_mean_peak_growth', 'heat_stress_days_peak_growth', 'solar_rad_peak_growth',
        #'DTR_accumulation_phase', 'temp_min_peak_growth', 'temp_max_peak_growth', 'spring_freezing_days',

        # --- CRITICAL: Remove contemporaneous features that are replaced by lagged versions ---
        #'national_avg_yield', 'producer_price_index'
    ]
    target_col = 'kreisYield'

    # --- Data Integrity Check ---
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from the input file: {missing_cols}")
        return

    print(f"--- Starting 10-Fold Cross-Validation for Champion XGBoost Model ---")
    print(f"Using {len(feature_cols)} features.")

    # Lists to store the performance metrics from each fold
    r2_scores = []
    rmse_scores = []

    N_FOLDS = 10
    for i in range(N_FOLDS):
        print(f"\n--- FOLD {i + 1}/{N_FOLDS} ---")

        # We use the same yearly split logic, but change the random_state for each fold
        # to ensure we are training and testing on different subsets of the data.
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []

        for year in df['year'].unique():
            df_year = df[df['year'] == year]
            X_year = df_year[feature_cols]
            y_year = df_year[target_col]

            # The only change is here: random_state=i
            X_train_yr, X_test_yr, y_train_yr, y_test_yr = train_test_split(
                X_year, y_year, test_size=0.2, random_state=i
            )
            X_train_list.append(X_train_yr)
            X_test_list.append(X_test_yr)
            y_train_list.append(y_train_yr)
            y_test_list.append(y_test_yr)

        X_train = pd.concat(X_train_list)
        X_test = pd.concat(X_test_list)
        y_train = pd.concat(y_train_list)
        y_test = pd.concat(y_test_list)

        # --- Train and Evaluate the XGBoost Model for this Fold ---
        xgb = XGBRegressor(
            objective='reg:squarederror', n_estimators=500, learning_rate=0.03,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)

        fold_r2 = r2_score(y_test, y_pred_xgb)
        fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

        r2_scores.append(fold_r2)
        rmse_scores.append(fold_rmse)

        print(f"  R-squared (R2): {fold_r2:.4f}")
        print(f"  RMSE: {fold_rmse:.2f} dt/ha")

    # --- Print Final Cross-Validation Results ---
    print("\n-------------------------------------------------")
    print("--- Final Cross-Validation Performance Summary ---")
    print(f"Average R-squared (R2): {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.2f} +/- {np.std(rmse_scores):.2f} dt/ha")
    print("-------------------------------------------------")

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
    perform_robust_cross_validation()