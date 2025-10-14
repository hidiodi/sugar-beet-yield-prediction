# File: src/models/final_robust_model.py
# Description: This is the definitive model, incorporating all lessons learned.
# It uses the most data, a targeted feature, drops problematic columns, and uses a
# constrained "fine-tuning" search to improve a robust baseline without overfitting.

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion_final.joblib')
IMPORTANCE_PLOT_PATH = os.path.join('reports/figures', 'feature_importance_final.png')


def train_and_validate_final_robust_model():
    """
    Implements the final, robust modeling strategy.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # --- FINAL FEATURE SET: Drop the problematic 'post_quota_era' column ---
    print("--- Dropping 'post_quota_era' feature to improve model robustness ---")
    static_and_lagged_features = [
        # Static & Geographic
        'avg_elevation', 'avg_soil_pawc', 'lon', 'lat',

        # Lagged Economic & Yield
        'national_avg_yield_lag1',

        # === NEW: Using the STABLE ANOMALY versions of economic features ===
        'producer_price_index_lag1_anomaly',
        'seed_price_index_lag1_anomaly',
        'energy_price_index_lag1_anomaly',
        'fertilizer_price_index_lag1_anomaly',
        'plant_protection_price_index_lag1_anomaly',

        # Satellite (Pre-Season)
        'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
        'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days',

        # Evolutionary Trend
        #'year_trend', 'has_satellite_data'
    ]

    # New, powerful weather features from the updated data pipeline
    new_weather_features = [
        # Tier 1: Antecedent Period Indices (from AgERA5 history)
        'antecedent_frost_days_anomaly',
        'antecedent_heavy_precip_days_anomaly',
        'antecedent_gdd_sum_anomaly',

        # Tier 2: Forecast Period Monthly Averages (to match SEAS5)
        'temp_mean_mar_anomaly', 'precip_sum_mar_anomaly', 'srad_mean_mar_anomaly',
        'temp_mean_apr_anomaly', 'precip_sum_apr_anomaly', 'srad_mean_apr_anomaly',
        'temp_mean_may_anomaly', 'precip_sum_may_anomaly', 'srad_mean_may_anomaly',
        'temp_mean_jun_anomaly', 'precip_sum_jun_anomaly', 'srad_mean_jun_anomaly',
        'temp_mean_jul_anomaly', 'precip_sum_jul_anomaly', 'srad_mean_jul_anomaly'
    ]
    polynomial_features = [
        'temp_mean_jul_anomaly_sq',
        'temp_mean_jun_anomaly_sq',
        'precip_sum_jul_anomaly_sq',
        'srad_mean_jul_anomaly_sq'
    ]

    interaction_features = [
        'july_heat_x_profit_margin',
        'june_precip_x_input_costs'
    ]

    # Combine them into the final list of features for the model
    feature_cols = static_and_lagged_features + new_weather_features + interaction_features + polynomial_features
    target_col = 'kreisYield'

    print("\n--- Applying Adaptive (Rolling Mean) Detrending ---")
    df['yield_trend'] = df.groupby('district_no')[target_col].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    )
    # Ensure no NaNs are left in the trend feature after rolling mean
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    detrended_target_col = 'kreisYield_detrended'
    df[detrended_target_col] = df[target_col] - df['yield_trend']
    print(" -> Detrending complete.")

    # --- Use the most logical split: Train on all data before the forecast period ---
    validation_start_year = 2011

    print(
        f"\n--- Using Final Split: Training before {validation_start_year}, Validating from {validation_start_year} ---")
    train_df = df[df['year'] < validation_start_year].copy()
    validation_df = df[df['year'] >= validation_start_year].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[detrended_target_col]
    X_validation = validation_df[feature_cols]
    y_validation_actual = validation_df[target_col]

    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_validation)} samples")

    # --- STRATEGY: Fine-tuning instead of a wide search ---
    # We search in a narrowband around the original model's robust parameters
    xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    param_dist = {
        'n_estimators': randint(400, 800),  # Around 500
        'learning_rate': uniform(0.02, 0.04),  # Around 0.03-0.05
        'max_depth': randint(4, 9),  # Around 5
        'subsample': uniform(0.7, 0.2),  # Around 0.8 (0.7 to 0.9)
        'colsample_bytree': uniform(0.7, 0.2)  # Around 0.8 (0.7 to 0.9)
    }
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        estimator=xgb_base, param_distributions=param_dist, n_iter=40,
        # 40 iterations is sufficient for a narrow search
        cv=tscv, scoring='r2', n_jobs=-1, verbose=1, random_state=42
    )

    print("\n--- Starting Constrained Hyperparameter Search (Fine-tuning) ---")
    random_search.fit(X_train, y_train)

    print(f"\nBest parameters found: {random_search.best_params_}")
    print(f"Best cross-validation R2 score during search: {random_search.best_score_:.4f}")

    best_xgb_model = random_search.best_estimator_
    y_pred_detrended = best_xgb_model.predict(X_validation)
    y_pred_final = y_pred_detrended + validation_df['yield_trend']

    r2 = r2_score(y_validation_actual, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final))

    print("\n--- FINAL ROBUST MODEL Validation Performance ---")
    print(f"  R-squared (R2): {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} dt/ha")
    print("-------------------------------------------------")

    # Plotting and saving...

    # ==============================================================================
    # === NEW: PLOT AND SAVE FEATURE IMPORTANCE ===
    # ==============================================================================
    print("Generating and saving feature importance plot...")
    try:
        # Define a new path for this version of the plot to avoid overwriting
        IMPORTANCE_PLOT_PATH = os.path.join('reports/figures', 'feature_importance_final_with_economics.png')

        # Get feature importances from the best model found by the search
        importance_scores = best_xgb_model.feature_importances_

        # Create a DataFrame for easier sorting and plotting
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance_scores
        }).sort_values(by='Importance', ascending=True)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))  # Adjust size as needed for all features
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        ax.set_title('Feature Importance (Final Model with Advanced Weather & Economic Features)')
        ax.set_xlabel('Feature Importance Score (Gini Importance)')
        ax.set_ylabel('Features')
        plt.tight_layout()

        # Create the directory if it doesn't exist and save the plot
        os.makedirs(os.path.dirname(IMPORTANCE_PLOT_PATH), exist_ok=True)
        plt.savefig(IMPORTANCE_PLOT_PATH, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to {IMPORTANCE_PLOT_PATH}")

    except Exception as e:
        print(f"❌ Warning: Could not save the feature importance plot. Error: {e}")
    # ==============================================================================

    try:
        # Code to plot and save feature importance
        joblib.dump(best_xgb_model, MODEL_PATH)
        print(f"\n✅ Final robust model trained up to {validation_start_year - 1} and saved to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving artifact: {e}")


if __name__ == "__main__":
    train_and_validate_final_robust_model()