# File: src/models/base_model_quantile.py
# Description: DEFINITIVE QUANTILE MODEL. Implements a robust train/validation/test split
# to predict a deviation range for yield instead of a single point.
# - Train: Before 2009
# - Validate: 2009-2018
# - Test: 2019+
# This provides a prediction interval, offering a measure of uncertainty.

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- Define paths for the three models ---
MODEL_PATH_LOWER = os.path.join('src/models', 'final_xgb_model_champion_lower.joblib')
MODEL_PATH_MEDIAN = os.path.join('src/models', 'final_xgb_model_champion_median.joblib')
MODEL_PATH_UPPER = os.path.join('src/models', 'final_xgb_model_champion_upper.joblib')

IMPORTANCE_PLOT_PATH = os.path.join('reports/figures', 'feature_importance_champion_final.png')

# --- BEST HYPERPARAMETERS (UNCHANGED) ---
BEST_PARAMS = {
    'colsample_bytree': 0.8223306320976561,
    'learning_rate': 0.020282652208788696,
    'max_depth': 4,
    'n_estimators': 448,
    'subsample': 0.8049549320516778
}

# --- Quantiles to predict for the deviation range ---
QUANTILES = {'lower': 0.1, 'median': 0.5, 'upper': 0.9}

def train_validate_and_test_quantile():
    """
    Loads data, applies detrending, performs a 3-way time-series split,
    trains three quantile models (lower, median, upper), evaluates them,
    and saves the final models.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # --- FINAL FEATURE SET (UNCHANGED) ---
    feature_cols = [
        'avg_elevation', 'avg_soil_pawc', 'lon', 'lat',
        'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
        'producer_price_index_lag1_anomaly',
        'seed_price_index_lag1_anomaly',
        'energy_price_index_lag1_anomaly', 'fertilizer_price_index_lag1_anomaly',
        'plant_protection_price_index_lag1_anomaly', 'winter_cropland_ndvi_mean',
        'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days', 'has_satellite_data', 'antecedent_frost_days_anomaly',
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

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from the input file: {missing_cols}")
        return

    # --- Adaptive (Rolling Mean) Detrending (UNCHANGED) ---
    print("\n--- Applying Adaptive (Rolling Mean) Detrending ---")
    df['yield_trend'] = df.groupby('district_no')[target_col].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    df[detrended_target_col] = df[target_col] - df['yield_trend']
    print(" -> Detrending complete.")

    # ============================ 3-WAY SPLIT (UNCHANGED) ============================
    validation_start_year = 2000
    test_start_year = 2007

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
    # =================================================================================

    # --- Train Three XGBoost Models for Quantiles ---
    models = {}
    for name, alpha in QUANTILES.items():
        print(f"\n--- Training model for {name} quantile ({alpha:.2f}) ---")
        xgb = XGBRegressor(
            objective='reg:quantileerror',  # Changed for quantile regression
            quantile_alpha=alpha,           # Specify the quantile
            n_estimators=BEST_PARAMS['n_estimators'],
            learning_rate=BEST_PARAMS['learning_rate'],
            max_depth=BEST_PARAMS['max_depth'],
            subsample=BEST_PARAMS['subsample'],
            colsample_bytree=BEST_PARAMS['colsample_bytree'],
            random_state=42, n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        models[name] = xgb

    # --- Evaluate on VALIDATION Set ---
    y_pred_detrended_val_lower = models['lower'].predict(X_validation)
    y_pred_detrended_val_median = models['median'].predict(X_validation)
    y_pred_detrended_val_upper = models['upper'].predict(X_validation)

    y_pred_final_val_lower = y_pred_detrended_val_lower + validation_df['yield_trend']
    y_pred_final_val_median = y_pred_detrended_val_median + validation_df['yield_trend']
    y_pred_final_val_upper = y_pred_detrended_val_upper + validation_df['yield_trend']

    r2_val = r2_score(y_validation_actual, y_pred_final_val_median)
    rmse_val = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final_val_median))
    avg_deviation_val = np.mean(y_pred_final_val_upper - y_pred_final_val_lower)

    print("-------------------------------------------------")
    print(f"Validation data: Years {validation_start_year} to {test_start_year - 1}")
    print(f"  Median R-squared (R2): {r2_val:.4f}")
    print(f"  Median RMSE: {rmse_val:.2f} dt/ha")
    print(f"  Average Deviation (80% Interval): {avg_deviation_val:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- Evaluate on TEST Set ---
    y_pred_detrended_test_lower = models['lower'].predict(X_test)
    y_pred_detrended_test_median = models['median'].predict(X_test)
    y_pred_detrended_test_upper = models['upper'].predict(X_test)

    y_pred_final_test_lower = y_pred_detrended_test_lower + test_df['yield_trend']
    y_pred_final_test_median = y_pred_detrended_test_median + test_df['yield_trend']
    y_pred_final_test_upper = y_pred_detrended_test_upper + test_df['yield_trend']

    r2_test = r2_score(y_test_actual, y_pred_final_test_median)
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_pred_final_test_median))
    avg_deviation_test = np.mean(y_pred_final_test_upper - y_pred_final_test_lower)

    print(f"\n--- FINAL TEST Performance {test_start_year} + Holdout) ---")
    print(f"  Median R-squared (R2): {r2_test:.4f}")
    print(f"  Median RMSE: {rmse_test:.2f} dt/ha")
    print(f"  Average Deviation (80% Interval): {avg_deviation_test:.2f} dt/ha")
    print("-------------------------------------------------")


    # --- Plot and Save Feature Importance (Based on the median model) ---
    try:
        median_model = models['median']
        importance_scores = median_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance_scores
        }).sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        ax.set_title('Feature Importance (Median Prediction Model)')
        ax.set_xlabel('Feature Importance Score (Gini Importance)')
        ax.set_ylabel('Features')
        plt.tight_layout()

        os.makedirs(os.path.dirname(IMPORTANCE_PLOT_PATH), exist_ok=True)
        plt.savefig(IMPORTANCE_PLOT_PATH, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to {IMPORTANCE_PLOT_PATH}")
    except Exception as e:
        print(f"❌ Warning: Could not save the feature importance plot. Error: {e}")

    # --- Final Model Training and Saving ---
    try:
        os.makedirs(os.path.dirname(MODEL_PATH_MEDIAN), exist_ok=True)
        joblib.dump(models['lower'], MODEL_PATH_LOWER)
        joblib.dump(models['median'], MODEL_PATH_MEDIAN)
        joblib.dump(models['upper'], MODEL_PATH_UPPER)
        print(
            f"\n✅ Final quantile models successfully trained and saved.")
    except Exception as e:
        print(f"\n❌ Warning: Could not save the final models. Error: {e}")

if __name__ == "__main__":
    train_validate_and_test_quantile()