# File: src/models/best_base_model.py
# Description: FINALIZED to use a robust 10-fold cross-validation to build confidence in the champion model's performance
# and to find the best feature combination.

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib
from itertools import combinations
import time

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion.joblib')


def perform_robust_cross_validation_with_feature_selection():
    """
    Loads pre-season data, iterates through all possible feature combinations,
    performs a 10-fold cross-validation for each combination to robustly
    evaluate the champion XGBoost model, identifies the best feature set,
    and then trains one final model on all data using the best features.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return
    """
        !!!! this runs for 13 hours !!!!
        --- Best Feature Combination ---
        Features: ['district_no', 'avg_elevation', 'avg_soil_pawc', 'winter_temp_anomaly', 'spring_precip_anomaly_hybrid', 'summer_temp_anomaly_hybrid', 'summer_precip_anomaly_hybrid', 'fertilizer_price_index', 'energy_price_index']
        Average R-squared (R2): 0.8808
        Average RMSE: 46.95 dt/ha
        -------------------------------------------------
        
        --- Training Final Model on All Available Data with Best Features for Deployment ---
        
        ✅ Final XGBoost model successfully trained on all data with the best features and saved to src/models\final_xgb_model_champion.joblib
    """

    # --- This is our pool of potential features ---
    all_feature_cols = [
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
        'fertilizer_price_index',
        'energy_price_index',
    ]
    target_col = 'kreisYield'

    # --- Data Integrity Check ---
    missing_cols = [col for col in all_feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from the input file: {missing_cols}")
        return

    results = []
    best_r2 = -np.inf
    best_features = None
    best_rmse = np.inf

    # --- Iterate through all possible feature combinations ---
    for i in range(1, len(all_feature_cols) + 1):
        for feature_combination in combinations(all_feature_cols, i):
            feature_cols = list(feature_combination)
            print(f"--- Testing Feature Combination: {feature_cols} ---")

            # Lists to store the performance metrics from each fold
            r2_scores = []
            rmse_scores = []
            N_FOLDS = 10

            start_time = time.time()

            for j in range(N_FOLDS):
                X_train_list, X_test_list = [], []
                y_train_list, y_test_list = [], []

                for year in df['year'].unique():
                    df_year = df[df['year'] == year]
                    X_year = df_year[feature_cols]
                    y_year = df_year[target_col]

                    X_train_yr, X_test_yr, y_train_yr, y_test_yr = train_test_split(
                        X_year, y_year, test_size=0.2, random_state=j
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

            avg_r2 = np.mean(r2_scores)
            avg_rmse = np.mean(rmse_scores)
            std_r2 = np.std(r2_scores)
            std_rmse = np.std(rmse_scores)
            elapsed_time = time.time() - start_time

            print(f"  Average R-squared (R2): {avg_r2:.4f} +/- {std_r2:.4f}")
            print(f"  Average RMSE: {avg_rmse:.2f} +/- {std_rmse:.2f} dt/ha")
            print(f"  Time taken: {elapsed_time:.2f} seconds")

            results.append({
                'features': feature_cols,
                'avg_r2': avg_r2,
                'std_r2': std_r2,
                'avg_rmse': avg_rmse,
                'std_rmse': std_rmse
            })

            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_features = feature_cols
                best_rmse = avg_rmse

    # --- Print and save the results ---
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='avg_r2', ascending=False)
    results_df.to_csv('feature_selection_results.csv', index=False)

    print("\n-------------------------------------------------")
    print("--- Feature Selection Results ---")
    print(results_df)
    print("\n--- Best Feature Combination ---")
    print(f"Features: {best_features}")
    print(f"Average R-squared (R2): {best_r2:.4f}")
    print(f"Average RMSE: {best_rmse:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- Final Model Training on ALL data with the best features ---
    print("\n--- Training Final Model on All Available Data with Best Features for Deployment ---")
    final_model = XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.03,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    final_model.fit(df[best_features], df[target_col])

    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(final_model, MODEL_PATH)
        print(f"\n✅ Final XGBoost model successfully trained on all data with the best features and saved to {MODEL_PATH}")
    except Exception as e:
        print(f"\n❌ Warning: Could not save the final model. Error: {e}")


if __name__ == "__main__":
    perform_robust_cross_validation_with_feature_selection()