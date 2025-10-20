# File: src/02_models/XGBoost/classifier/ModelScripts/stacked_base_model.py
# Description: DEFINITIVE MODEL. **IMPROVED WITH STACKING AND A ROBUST 3-WAY SPLIT.**
# - Train: Before 2011
# - Validate: 2011-2018
# - Test: 2019+
# Applies causal detrending, uses a stacking ensemble, and evaluates on a true holdout test set.

import pandas as pd
from xgboost import XGBRegressor
# --- Imports for Stacking ---
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

# Define paths for the final champion model and its artifacts
MODEL_PATH = os.path.join('src/02_models', 'final_stacked_model_champion.joblib')
IMPORTANCE_PLOT_PATH = os.path.join('reports/figures', 'feature_importance_champion_final.png')
PERFORMANCE_PLOT_PATH = os.path.join('reports/figures', 'predicted_vs_actual_regression_final.png')

# --- BEST HYPERPARAMETERS FOUND IN THE ROBUST MODEL SEARCH ---
BEST_PARAMS = {
    'colsample_bytree': 0.8223306320976561,
    'learning_rate': 0.020282652208788696,
    'max_depth': 4,
    'n_estimators': 448,
    'subsample': 0.8049549320516778
}


def plot_predicted_vs_actual(y_actual, y_predicted, r2, rmse, path, dataset_name="Test"):
    """Generates and saves a scatter plot of predicted vs. actual values for the final test set."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_predicted, alpha=0.6, s=15)

    max_val = max(y_actual.max(), y_predicted.max())
    min_val = min(y_actual.min(), y_predicted.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

    textstr = f'Dataset: {dataset_name}\nR²: {r2:.4f}\nRMSE: {rmse:.2f} dt/ha'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)

    plt.title(f'Final Model Performance: Predicted vs. Actual Yield ({dataset_name} Set)')
    plt.xlabel('Actual Yield (dt/ha)')
    plt.ylabel('Predicted Yield (dt/ha)')
    plt.grid(True)
    plt.legend()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Final performance plot saved to {path}")


def train_validate_and_test_stacked_model():
    """
    Loads data, applies causal detrending, performs a 3-way time-series split,
    trains and compares a base XGBoost and a Stacking model on a validation set,
    evaluates the champion model on a final test set, and saves the artifacts.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # ============================ THE FIX: CORRECTED FEATURE SET ============================
    # This feature list now matches the one from your working train_validate_and_test script.
    feature_cols = [
        'avg_elevation',
        'avg_soil_pawc',
        'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
        'fertilizer_price_index_lag1_anomaly',
        'plant_protection_price_index_lag1_anomaly',
        'antecedent_frost_days_anomaly',
        'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
        'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast',
        'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
        'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast',
        'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',
        'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
        'spring_temp_anomaly_forecast_sq', 'summer_temp_anomaly_forecast_sq',
        'spring_precip_anomaly_forecast_sq', 'summer_precip_anomaly_forecast_sq'
    ]
    # ========================================================================================
    target_col = 'kreisYield'
    detrended_target_col = 'kreisYield_detrended'

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are still missing: {missing_cols}")
        print("Please ensure your feature engineering script is up-to-date and has been run.")
        return

    # --- Causal (Trailing) Rolling Mean Detrending ---
    print("\n--- Applying Causal (Trailing Mean) Detrending ---")
    df['yield_trend'] = df.groupby('district_no')[target_col].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df.dropna(subset=['yield_trend'], inplace=True) # Drop rows where a trend couldn't be computed
    df[detrended_target_col] = df[target_col] - df['yield_trend']
    print(" -> Detrending complete.")

    # --- 3-WAY TIME-BASED SPLIT ---
    validation_start_year = 2011
    test_start_year = 2019
    print(f"\n--- Using Train / Validation / Test Split ---")
    print(f"Training data:   Years < {validation_start_year}")
    print(f"Validation data: Years {validation_start_year} to {test_start_year - 1}")
    print(f"Test data:       Years >= {test_start_year}")

    train_df = df[df['year'] < validation_start_year].copy()
    validation_df = df[(df['year'] >= validation_start_year) & (df['year'] < test_start_year)].copy()
    test_df = df[df['year'] >= test_start_year].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[detrended_target_col]

    X_validation = validation_df[feature_cols]
    y_validation_actual = validation_df[target_col]

    X_test = test_df[feature_cols]
    y_test_actual = test_df[target_col]

    print(f"\nTraining set size:   {len(X_train)} samples")
    print(f"Validation set size: {len(X_validation)} samples")
    print(f"Test set size:       {len(X_test)} samples")

    # --- 1. Train Base XGBoost Model (ON TRAINING DATA ONLY) ---
    xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **BEST_PARAMS)
    xgb_base.fit(X_train, y_train)

    # --- 2. Train Stacked Model (ON TRAINING DATA ONLY) ---
    print("\n--- Training Stacking Regressor (XGBoost + Ridge) ---")
    estimators = [
        ('xgb', xgb_base),
        ('ridge', Ridge(alpha=1.0, random_state=42))
    ]
    stacked_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=0.1, random_state=42),
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    stacked_regressor.fit(X_train, y_train)
    print(" -> Training complete.")

    # --- 3. Evaluate both models on the VALIDATION set ---
    print("\n--- Evaluating Models on VALIDATION Set ---")
    # Base XGBoost
    y_pred_detrended_base_val = xgb_base.predict(X_validation)
    y_pred_final_base_val = y_pred_detrended_base_val + validation_df['yield_trend']
    r2_base_val = r2_score(y_validation_actual, y_pred_final_base_val)
    rmse_base_val = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final_base_val))

    # Stacking Regressor
    y_pred_detrended_stack_val = stacked_regressor.predict(X_validation)
    y_pred_final_stack_val = y_pred_detrended_stack_val + validation_df['yield_trend']
    r2_stack_val = r2_score(y_validation_actual, y_pred_final_stack_val)
    rmse_stack_val = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final_stack_val))

    print(f"  BASE XGBoost (Validation): R2={r2_base_val:.4f}, RMSE={rmse_base_val:.2f} dt/ha")
    print(f"  STACKED Model (Validation): R2={r2_stack_val:.4f}, RMSE={rmse_stack_val:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- 4. Decide champion model based on VALIDATION performance ---
    if rmse_stack_val < rmse_base_val:
        champion_model = stacked_regressor
        model_name = "Stacking Regressor"
        print(f"Stacking improved performance. Selecting it as the champion model.")
    else:
        champion_model = xgb_base
        model_name = "Base XGBoost"
        MODEL_PATH = os.path.join('src/02_models', 'final_xgb_model_champion_final.joblib') # Adjust path if base wins
        print(f"⚠️ Stacking did not improve performance. Selecting Base XGBoost as the champion.")

    # --- 5. Evaluate the CHAMPION model on the TEST set ---
    print(f"\n--- FINAL TEST Performance of Champion ({model_name}) ---")
    y_pred_detrended_test = champion_model.predict(X_test)
    y_pred_final_test = y_pred_detrended_test + test_df['yield_trend']
    r2_test = r2_score(y_test_actual, y_pred_final_test)
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_pred_final_test))
    print(f"  R-squared (R2): {r2_test:.4f}")
    print(f"  RMSE: {rmse_test:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- 6. Save Final Artifacts ---
    # Save the champion model (trained ONLY on pre-2011 data)
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(champion_model, MODEL_PATH)
        print(f"\nChampion Model ({model_name}) saved to {MODEL_PATH}")
    except Exception as e:
        print(f"\n❌ Error: Could not save the final model. Error: {e}")
        sys.exit(1)

    # Plot and save feature importance from the powerful base model
    try:
        importance_scores = xgb_base.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance_scores})
        feature_importance_df.sort_values(by='Importance', ascending=True, inplace=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        ax.set_title('Feature Importance (XGBoost Base Model)')
        ax.set_xlabel('Feature Importance Score')
        plt.tight_layout()
        os.makedirs(os.path.dirname(IMPORTANCE_PLOT_PATH), exist_ok=True)
        plt.savefig(IMPORTANCE_PLOT_PATH)
        print(f"Feature importance plot saved to {IMPORTANCE_PLOT_PATH}")
    except Exception as e:
        print(f"❌ Warning: Could not save the feature importance plot. Error: {e}")

    # Plot the final champion model's performance on the TEST set
    plot_predicted_vs_actual(y_test_actual, y_pred_final_test, r2_test, rmse_test, PERFORMANCE_PLOT_PATH, "Test")

if __name__ == "__main__":
    train_validate_and_test_stacked_model()