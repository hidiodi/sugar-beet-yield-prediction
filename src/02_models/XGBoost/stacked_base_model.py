# File: src/models/base_model.py
# Description: DEFINITIVE MODEL. **IMPROVED WITH STACKING REGRESSOR.**
# Applies adaptive detrending, uses the final comprehensive feature set, and
# implements a stacking ensemble approach for robust final predictions.

import pandas as pd
from xgboost import XGBRegressor
# --- New Imports for Stacking ---
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold
# --------------------------------
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt
import sys

warnings.filterwarnings("ignore")

# Updated path to reflect the 'champion' status of this final robust configuration
MODEL_PATH = os.path.join('src/02_models', 'final_stacked_model_champion.joblib')
IMPORTANCE_PLOT_PATH = os.path.join('reports/figures', 'feature_importance_champion_final.png')
STACKING_PLOT_PATH = os.path.join('reports/figures', 'predicted_vs_actual_regression.png')

# --- BEST HYPERPARAMETERS FOUND IN THE ROBUST MODEL SEARCH ---
BEST_PARAMS = {
    'colsample_bytree': 0.8223306320976561,
    'learning_rate': 0.020282652208788696,
    'max_depth': 4,
    'n_estimators': 448,
    'subsample': 0.8049549320516778
}


def plot_predicted_vs_actual(y_actual, y_predicted, r2, rmse, path):
    """Generates and saves a scatter plot of predicted vs. actual values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_predicted, alpha=0.6, s=15)

    # Add the diagonal line for perfect prediction (y=x)
    max_val = max(y_actual.max(), y_predicted.max())
    min_val = min(y_actual.min(), y_predicted.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

    # Add text box for metrics
    textstr = f'R²: {r2:.4f}\nRMSE: {rmse:.2f} dt/ha'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)

    plt.title('Stacked Regression: Predicted vs. Actual Yield')
    plt.xlabel('Actual Yield (dt/ha)')
    plt.ylabel('Predicted Yield (dt/ha)')
    plt.grid(True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ Predicted vs. Actual plot saved to {path}")


def train_and_validate_with_holdout():
    """
    Loads pre-season data, applies adaptive detrending, trains the champion XGBoost model,
    improves it with a Stacking Regressor, evaluates both, and saves the final stacked model.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # --- FINAL ROBUST CHAMPION FEATURE SET (omitted for brevity, copied from original) ---
    feature_cols = [
        # Static & Geographic
        'avg_elevation', 'avg_soil_pawc', 'lon', 'lat',
        # Lagged Economic & Yield
        'national_avg_yield_lag1', 'producer_price_index_lag1_anomaly',
        'seed_price_index_lag1_anomaly', 'energy_price_index_lag1_anomaly',
        'fertilizer_price_index_lag1_anomaly', 'plant_protection_price_index_lag1_anomaly',
        # Satellite (Pre-Season)
        'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
        'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days',
        # Tier 1: Antecedent Period Indices (from AgERA5 history)
        'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly',
        'antecedent_gdd_sum_anomaly',
        # Tier 2: Forecast Period Monthly Averages (to match SEAS5)
        'temp_mean_mar_anomaly', 'precip_sum_mar_anomaly', 'srad_mean_mar_anomaly',
        'temp_mean_apr_anomaly', 'precip_sum_apr_anomaly', 'srad_mean_apr_anomaly',
        'temp_mean_may_anomaly', 'precip_sum_may_anomaly', 'srad_mean_may_anomaly',
        'temp_mean_jun_anomaly', 'precip_sum_jun_anomaly', 'srad_mean_jun_anomaly',
        'temp_mean_jul_anomaly', 'precip_sum_jul_anomaly', 'srad_mean_jul_anomaly',
        # Non-linear & Interaction Terms
        'temp_mean_jul_anomaly_sq', 'temp_mean_jun_anomaly_sq',
        'precip_sum_jul_anomaly_sq', 'srad_mean_jul_anomaly_sq',
        'july_heat_x_profit_margin', 'june_precip_x_input_costs'
    ]
    target_col = 'kreisYield'
    detrended_target_col = 'kreisYield_detrended'

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from the input file: {missing_cols}")
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

    # --- Time-Based Split ---
    validation_start_year = 2011
    train_df = df[df['year'] < validation_start_year].copy()
    validation_df = df[df['year'] >= validation_start_year].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[detrended_target_col]
    X_validation_raw = validation_df[feature_cols]  # Keep the raw validation set reference
    y_validation_actual_raw = validation_df[target_col]
    y_validation_trend_raw = validation_df['yield_trend']

    # Handle NaNs: Clean the training set by dropping NaNs (first year of data in each district)
    train_df_clean = pd.concat([X_train, y_train], axis=1).dropna()
    X_train_clean = train_df_clean[feature_cols]
    y_train_clean = train_df_clean[detrended_target_col]

    # --- CRITICAL FIX: Clean the validation set in the same way ---
    validation_df_clean = pd.concat([X_validation_raw, y_validation_actual_raw, y_validation_trend_raw],
                                    axis=1).dropna()
    X_validation_clean = validation_df_clean[feature_cols]
    y_validation_actual = validation_df_clean[target_col]
    y_validation_trend = validation_df_clean['yield_trend']

    print(f"Training set size (clean): {len(X_train_clean)} samples")
    print(f"Validation set size (clean): {len(X_validation_clean)} samples")  # Check the size after drop

    # --- 1. Train and Evaluate the Base XGBoost Model (for comparison) ---
    xgb_base = XGBRegressor(
        objective='reg:squarederror', random_state=42, n_jobs=-1,
        **BEST_PARAMS
    )
    xgb_base.fit(X_train_clean, y_train_clean)

    # Base Model Prediction (Use the CLEANED validation set)
    y_pred_detrended_base = xgb_base.predict(X_validation_clean)
    y_pred_final_base = y_pred_detrended_base + y_validation_trend

    r2_base = r2_score(y_validation_actual, y_pred_final_base)
    rmse_base = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final_base))

    print("\n--- BASE XGBoost Validation Performance ---")
    print(f"  R-squared (R2): {r2_base:.4f}")
    print(f"  RMSE: {rmse_base:.2f} dt/ha")

    # --- 2. Train and Evaluate the Stacked Model ---
    print("\n--- Training Stacking Regressor (XGBoost + Ridge) ---")

    estimators = [
        ('xgb', xgb_base),  # The tuned model (already fitted for speed)
        ('ridge', Ridge(alpha=1.0, random_state=42))  # Simple, well-behaved linear model
    ]

    stacked_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=0.1, random_state=42),
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )

    stacked_regressor.fit(X_train_clean, y_train_clean)

    # Stacked Model Prediction (Use the CLEANED validation set)
    y_pred_detrended_stack = stacked_regressor.predict(X_validation_clean)
    y_pred_final_stack = y_pred_detrended_stack + y_validation_trend

    r2_stack = r2_score(y_validation_actual, y_pred_final_stack)
    rmse_stack = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final_stack))

    # --- Comparison and Final Decision ---
    print("\n--- STACKED Model Validation Performance ---")
    print(f"  R-squared (R2): {r2_stack:.4f}")
    print(f"  RMSE: {rmse_stack:.2f} dt/ha")
    print("-------------------------------------------------")

    # Decide which model to save based on RMSE (lower is better)
    if rmse_stack < rmse_base:
        final_model = stacked_regressor
        r2_final, rmse_final = r2_stack, rmse_stack
        print(f"✅ Stacking improved performance. Saving Stacking Regressor.")
    else:
        final_model = xgb_base
        r2_final, rmse_final = r2_base, rmse_base
        MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion_final.joblib')
        print(f"⚠️ Stacking did not improve performance. Saving Base XGBoost Regressor.")

    # --- Plot and Save Feature Importance ---
    try:
        # We still plot feature importance from the strongest base model (xgb_base)
        importance_scores = xgb_base.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance_scores
        }).sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        ax.set_title('Feature Importance (Champion XGBoost Base)')
        ax.set_xlabel('Feature Importance Score (Gini Importance)')
        ax.set_ylabel('Features')
        plt.tight_layout()
        os.makedirs(os.path.dirname(IMPORTANCE_PLOT_PATH), exist_ok=True)
        plt.savefig(IMPORTANCE_PLOT_PATH, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to {IMPORTANCE_PLOT_PATH}")
    except Exception as e:
        print(f"❌ Warning: Could not save the feature importance plot. Error: {e}")

    # --- Save Final Artifacts and Plot ---
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(final_model, MODEL_PATH)
        print(f"\n✅ Final Model saved to {MODEL_PATH}")

        # Plot the winning model's performance
        y_plot_actual = y_validation_actual
        y_plot_predicted = y_pred_final_stack if final_model is stacked_regressor else y_pred_final_base

        plot_predicted_vs_actual(y_plot_actual, y_plot_predicted, r2_final, rmse_final, STACKING_PLOT_PATH)

    except Exception as e:
        print(f"\n❌ Warning: Could not save final model or plot. Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    train_and_validate_with_holdout()