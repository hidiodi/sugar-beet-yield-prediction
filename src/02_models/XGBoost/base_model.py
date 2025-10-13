# File: src/models/base_model.py
# Description: DEFINITIVE MODEL. Uses time-based split for validation (holdout from 2011),
# applies adaptive detrending, uses the final comprehensive feature set, and
# implements the best-found hyperparameters from the fine-tuning search.
# This model is optimized for pre-season forecasting (no data leakage).

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Updated path to reflect the 'champion' status of this final robust configuration
MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion_final.joblib')
IMPORTANCE_PLOT_PATH = os.path.join('reports/figures', 'feature_importance_champion_final.png')

# --- BEST HYPERPARAMETERS FOUND IN THE ROBUST MODEL SEARCH ---
BEST_PARAMS = {
    'colsample_bytree': 0.8223306320976561,
    'learning_rate': 0.020282652208788696,
    'max_depth': 4,
    'n_estimators': 448,
    'subsample': 0.8049549320516778
}


def train_and_validate_with_holdout():
    """
    Loads pre-season data, applies adaptive detrending, splits data for a robust
    validation (holdout from 2011), trains the XGBoost model with optimized
    parameters, evaluates it, and saves the final model.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # --- FINAL ROBUST CHAMPION FEATURE SET ---
    # This feature set is chosen for its stability, predictive power, and
    # strict adherence to pre-season availability.
    feature_cols = [
        # Static & Geographic
        'avg_elevation', 'avg_soil_pawc', 'lon', 'lat',

        # Lagged Economic & Yield
        'national_avg_yield_lag1',
        'producer_price_index_lag1_anomaly',
        'seed_price_index_lag1_anomaly',
        'energy_price_index_lag1_anomaly',
        'fertilizer_price_index_lag1_anomaly',
        'plant_protection_price_index_lag1_anomaly',

        # Satellite (Pre-Season)
        'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
        'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days',
        # 'has_satellite_data' # Removed as a direct feature, implicitly handled by data quality/imputation

        # Tier 1: Antecedent Period Indices (from AgERA5 history)
        'antecedent_frost_days_anomaly',
        'antecedent_heavy_precip_days_anomaly',
        'antecedent_gdd_sum_anomaly',

        # Tier 2: Forecast Period Monthly Averages (to match SEAS5)
        'temp_mean_mar_anomaly', 'precip_sum_mar_anomaly', 'srad_mean_mar_anomaly',
        'temp_mean_apr_anomaly', 'precip_sum_apr_anomaly', 'srad_mean_apr_anomaly',
        'temp_mean_may_anomaly', 'precip_sum_may_anomaly', 'srad_mean_may_anomaly',
        'temp_mean_jun_anomaly', 'precip_sum_jun_anomaly', 'srad_mean_jun_anomaly',
        'temp_mean_jul_anomaly', 'precip_sum_jul_anomaly', 'srad_mean_jul_anomaly',

        # Non-linear & Interaction Terms
        'temp_mean_jul_anomaly_sq',
        'temp_mean_jun_anomaly_sq',
        'precip_sum_jul_anomaly_sq',
        'srad_mean_jul_anomaly_sq',
        'july_heat_x_profit_margin', # Interaction
        'june_precip_x_input_costs' # Interaction
    ]
    target_col = 'kreisYield'
    detrended_target_col = 'kreisYield_detrended'

    # --- Data Integrity Check ---
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from the input file: {missing_cols}")
        # Note: You might need to check for 'july_heat_x_profit_margin' and 'june_precip_x_input_costs'
        # if they are not pre-calculated and saved in 'stage1_preseason_features.csv'
        return

    # --- Adaptive (Rolling Mean) Detrending ---
    print("\n--- Applying Adaptive (Rolling Mean) Detrending ---")
    df['yield_trend'] = df.groupby('district_no')[target_col].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    df[detrended_target_col] = df[target_col] - df['yield_trend']
    print(" -> Detrending complete.")


    # --- Time-Based Split: Use the robust 2011 split ---
    validation_start_year = 2011

    print(f"\n--- Using Final Robust Split ---")
    print(f"Training data will be from years before {validation_start_year}")
    print(f"Validation data will be from years {validation_start_year} to {df['year'].max()}")

    train_df = df[df['year'] < validation_start_year].copy()
    validation_df = df[df['year'] >= validation_start_year].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[detrended_target_col]
    X_validation = validation_df[feature_cols]
    y_validation_actual = validation_df[target_col] # Keep actual for final scoring

    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_validation)} samples")

    # --- Train and Evaluate the XGBoost Model with Best Params ---
    xgb = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=BEST_PARAMS['n_estimators'],
        learning_rate=BEST_PARAMS['learning_rate'],
        max_depth=BEST_PARAMS['max_depth'],
        subsample=BEST_PARAMS['subsample'],
        colsample_bytree=BEST_PARAMS['colsample_bytree'],
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    # Re-trend the prediction
    y_pred_detrended = xgb.predict(X_validation)
    y_pred_final = y_pred_detrended + validation_df['yield_trend']

    r2 = r2_score(y_validation_actual, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final))

    print("\n--- FINAL ROBUST MODEL Validation Performance (2011+ Holdout) ---")
    print(f"  R-squared (R2): {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- Plot and Save Feature Importance ---
    try:
        # Get feature importances
        importance_scores = xgb.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance_scores
        }).sort_values(by='Importance', ascending=True)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        ax.set_title('Feature Importance (Final Champion Model)')
        ax.set_xlabel('Feature Importance Score (Gini Importance)')
        ax.set_ylabel('Features')
        plt.tight_layout()

        # Create the directory if it doesn't exist and save the plot
        os.makedirs(os.path.dirname(IMPORTANCE_PLOT_PATH), exist_ok=True)
        plt.savefig(IMPORTANCE_PLOT_PATH, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to {IMPORTANCE_PLOT_PATH}")
    except Exception as e:
        print(f"❌ Warning: Could not save the feature importance plot. Error: {e}")

    # --- Final Model Training and Saving ---
    # The model 'xgb' is already the final model trained on the training split.
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(xgb, MODEL_PATH)
        print(
            f"\n✅ Final XGBoost model (Champion Robust) successfully trained on data up to {validation_start_year - 1} and saved to {MODEL_PATH}")
    except Exception as e:
        print(f"\n❌ Warning: Could not save the final model. Error: {e}")


if __name__ == "__main__":
    train_and_validate_with_holdout()