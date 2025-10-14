# File: src/models/walk_forward_model.py
# Description: Implements a robust walk-forward validation to predict unseen future data
#              and builds confidence in the champion model's performance over time.

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion_walkforward.joblib')


def perform_walk_forward_validation():
    """
    Loads pre-season data, performs walk-forward validation to robustly evaluate
    the champion XGBoost model on unseen future years, and then trains one final model on all data.
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
        'year',  # Note: 'year' as a direct feature can introduce time trends, be mindful.
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

    print(f"--- Starting Walk-Forward Validation for Champion XGBoost Model ---")
    print(f"Using {len(feature_cols)} features.")

    # Sort data by year to ensure correct temporal split
    df = df.sort_values(by='year').reset_index(drop=True)
    unique_years = df['year'].unique()

    # Define minimum number of years required for initial training
    MIN_TRAIN_YEARS = 30
    if len(unique_years) < MIN_TRAIN_YEARS + 1:
        print(
            f"Error: Not enough unique years ({len(unique_years)}) for walk-forward validation with MIN_TRAIN_YEARS={MIN_TRAIN_YEARS}.")
        return

    # Lists to store the performance metrics from each fold
    r2_scores = []
    rmse_scores = []

    # Iterate through years for walk-forward validation
    # Each iteration, train on all data up to year 'y-1' and test on year 'y'
    for i in range(MIN_TRAIN_YEARS, len(unique_years)):
        test_year = unique_years[i]
        train_years_end = unique_years[i - 1]  # Train on data up to the year before the test year

        print(f"\n--- Walk-Forward Step: Training on Years <= {train_years_end}, Testing on Year {test_year} ---")

        train_df = df[df['year'] <= train_years_end]
        test_df = df[df['year'] == test_year]

        if train_df.empty or test_df.empty:
            print(f"  Skipping year {test_year} due to insufficient training or testing data.")
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        # --- Train and Evaluate the XGBoost Model for this Step ---
        xgb = XGBRegressor(
            objective='reg:squarederror', n_estimators=500, learning_rate=0.02,
            max_depth=10, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)

        step_r2 = r2_score(y_test, y_pred_xgb)
        step_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

        r2_scores.append(step_r2)
        rmse_scores.append(step_rmse)

        print(f"  R-squared (R2) for {test_year}: {step_r2:.4f}")
        print(f"  RMSE for {test_year}: {step_rmse:.2f} dt/ha")

    # --- Print Final Walk-Forward Validation Results ---
    print("\n-------------------------------------------------")
    print("--- Final Walk-Forward Validation Performance Summary ---")
    if r2_scores:
        print(f"Average R-squared (R2): {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")
        print(f"Average RMSE: {np.mean(rmse_scores):.2f} +/- {np.std(rmse_scores):.2f} dt/ha")
    else:
        print("No walk-forward validation steps were completed.")
    print("-------------------------------------------------")

    # --- Final Model Training on ALL data for deployment ---
    print("\n--- Training Final Model on All Available Data for Deployment ---")
    final_model = XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.02,
        max_depth=10, subsample=0.8, colsample_bytree=0.8,
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
    perform_walk_forward_validation()