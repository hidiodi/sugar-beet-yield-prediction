# File: src/models/base_model_quantile.py
# Description: DEFINITIVE MODEL V3. Uses individually tuned hyperparameters for each quantile
#              and an enhanced feature set to create a robust 95% prediction interval.

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- V3 MODEL PATHS ---
LOWER_BOUND_MODEL_PATH = os.path.join('src/models', 'final_xgb_model_lower.joblib')
MEDIAN_MODEL_PATH = os.path.join('src/models', 'final_xgb_model_median.joblib')
UPPER_BOUND_MODEL_PATH = os.path.join('src/models', 'final_xgb_model_upper.joblib')
IMPORTANCE_PLOT_PATH = os.path.join('reports/figures', 'feature_importance.png')

# --- Time-Series Split---
VALIDATION_START_YEAR = 2011
TEST_START_YEAR = 2019

BEST_PARAMS_0P025 = {
    'n_estimators': 963,
    'learning_rate': 0.012845,
    'max_depth': 9,
    'subsample': 0.885812,
    'colsample_bytree': 0.945145,
    'gamma': 3.591119,
    'min_child_weight': 5,
}

BEST_PARAMS_0P5 = {
    'n_estimators': 506,
    'learning_rate': 0.013730,
    'max_depth': 6,
    'subsample': 0.999079,
    'colsample_bytree': 0.623749,
    'gamma': 9.512946,
    'min_child_weight': 2,
}

BEST_PARAMS_0P975 = {
    'n_estimators': 494,
    'learning_rate': 0.075146,
    'max_depth': 6,
    'subsample': 0.752544,
    'colsample_bytree': 0.789189,
    'gamma': 9.271261,
    'min_child_weight': 8,
}


### <-- MODIFIED STEP 1: Create a map to link quantiles to their specific params -->
PARAMS_MAP = {
    0.025: BEST_PARAMS_0P025,
    0.5: BEST_PARAMS_0P5,
    0.975: BEST_PARAMS_0P975
}


def train_validate_and_test_with_intervals():
    """
    Loads data, applies detrending, performs a 3-way time-series split,
    trains quantile regression models with specific hyperparameters,
    evaluates on validation and test sets, and saves the final models.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # ============================ FEATURE SET ============================
    feature_cols = [
        # --- Original Features ---
        'antecedent_frost_days_anomaly',
        'antecedent_heavy_precip_days_anomaly',
        'antecedent_gdd_sum_anomaly',
        'spring_temp_anomaly_forecast',
        'spring_precip_anomaly_forecast',
        'spring_solar_rad_anomaly_forecast',
        'spring_evaporation_anomaly_forecast',
        'spring_runoff_anomaly_forecast',
        'spring_soil_temp_l1_anomaly_forecast',
        'spring_snowfall_anomaly_forecast',
        'summer_temp_anomaly_forecast',
        'summer_precip_anomaly_forecast',
        'summer_solar_rad_anomaly_forecast',
        'summer_evaporation_anomaly_forecast',
        'summer_runoff_anomaly_forecast',
        'summer_soil_temp_l1_anomaly_forecast',
        'summer_snowfall_anomaly_forecast',
        'spring_temp_prob_warm_forecast',
        'spring_precip_prob_wet_forecast',
        'spring_solar_rad_prob_wet_forecast',
        'spring_evaporation_prob_wet_forecast',
        'spring_runoff_prob_wet_forecast',
        'spring_soil_temp_l1_prob_warm_forecast',
        #'spring_snowfall_prob_wet_forecast',
        #'summer_temp_prob_warm_forecast',
        #'summer_precip_prob_wet_forecast',
        'summer_solar_rad_prob_wet_forecast',
        #'summer_evaporation_prob_wet_forecast',
        'summer_runoff_prob_wet_forecast',
        'summer_soil_temp_l1_prob_warm_forecast',
        #'summer_snowfall_prob_wet_forecast',
        'lat',
        'lon',
        'avg_elevation',
        #'avg_slope',
        'avg_bdod_0_30cm',
        #'avg_clay_0_30cm',
        #'avg_sand_0_30cm',
        #'avg_som_0_30cm',
        #'avg_phh2o_0_30cm',
        #'avg_bdod_0_100cm',
        #'avg_clay_0_100cm',
        #'avg_sand_0_100cm',
        #'avg_som_0_100cm',
        #'avg_phh2o_0_100cm',
        'winter_cropland_ndvi_mean',
        #'winter_cropland_ndvi_anomaly',
        'winter_cropland_LST_mean',
        #'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days',
        'fertilizer_price_index_lag1_anomaly_capped',
        #'is_fertilizer_price_extreme',
        #'is_summer_forecast_dry',
        'gdd_x_fertilizer_price',
        'spring_temp_x_spring_precip',
        'summer_heat_x_profit_margin',
        'summer_precip_x_input_costs',
        #'is_extreme_heat_forecast',
        #'is_extreme_drought_forecast',
        #'drought_x_heat',
        'spring_temp_prob_warm_forecast_sq',
        'summer_temp_prob_warm_forecast_sq',
        #'spring_precip_prob_wet_forecast_sq',
        #'summer_precip_prob_wet_forecast_sq',
        'antecedent_gdd_sum_anomaly_sq',
        'summer_temp_anomaly_forecast_sq',
        'summer_temp_anomaly_forecast_cubed',
        'summer_precip_anomaly_forecast_sq',
        'summer_precip_anomaly_forecast_cubed',
        'antecedent_gdd_sum_anomaly_cubed',

        # --- Advanced Interaction Features ---
        'hot_dry_interaction',
        'lat_x_summer_temp',
        'lon_x_spring_precip',
        'sandy_soil_x_drought',
        'profit_margin_momentum',
        'cost_of_inputs_momentum'
    ]
    # ======================================================================

    target_col = 'kreisYield'
    detrended_target_col = 'kreisYield_detrended'

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: The following feature columns are missing from the input file: {missing_cols}")
        print("💡 Please ensure you have run the latest version of 'build_stage1_features.py'.")
        return

    # --- Causal (Trailing) Rolling Mean Detrending ---
    print("\n--- Applying Causal (Trailing Mean) Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')[target_col].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x)
    df.dropna(subset=['yield_trend'], inplace=True)
    df[detrended_target_col] = df[target_col] - df['yield_trend']
    print(" -> Detrending complete.")

    # --- Time-Series Split ---
    train_df = df[df['year'] < VALIDATION_START_YEAR].copy()
    validation_df = df[(df['year'] >= VALIDATION_START_YEAR) & (df['year'] < TEST_START_YEAR)].copy()
    test_df = df[df['year'] >= TEST_START_YEAR].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[detrended_target_col]
    X_validation = validation_df[feature_cols]
    y_validation_actual = validation_df[target_col]
    X_test = test_df[feature_cols]
    y_test_actual = test_df[target_col]

    print(f"\nTraining set size:   {len(X_train)} samples")
    print(f"Validation set size: {len(X_validation)} samples")
    print(f"Test set size:       {len(X_test)} samples")

    # --- Train the XGBoost Models for Quantile Regression ---

    ### <-- MODIFIED STEP 2: Update the quantiles to match the new 95% interval -->
    quantiles = [0.025, 0.5, 0.975]
    models = {}

    for q in quantiles:
        print(f"\n--- Training New Model for quantile: {q} ---")

        ### <-- MODIFIED STEP 3: Select the correct parameters for the current quantile -->
        current_params = PARAMS_MAP[q]

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            **current_params,  # Use the specific, tuned parameters
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        models[q] = model

    # --- Evaluate on VALIDATION Set ---
    ### <-- MODIFIED STEP 4: Use the new quantile keys (0.025 and 0.975) -->
    y_pred_detrended_val_lower = models[0.025].predict(X_validation)
    y_pred_detrended_val_median = models[0.5].predict(X_validation)
    y_pred_detrended_val_upper = models[0.975].predict(X_validation)

    y_pred_final_val_lower = y_pred_detrended_val_lower + validation_df['yield_trend']
    y_pred_final_val_median = y_pred_detrended_val_median + validation_df['yield_trend']
    y_pred_final_val_upper = y_pred_detrended_val_upper + validation_df['yield_trend']

    r2_val = r2_score(y_validation_actual, y_pred_final_val_median)
    rmse_val = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final_val_median))
    print("-------------------------------------------------")
    print(f"Validation Performance (Median Prediction) (Years {VALIDATION_START_YEAR} to {TEST_START_YEAR - 1})")
    print(f"  R-squared (R2): {r2_val:.4f}")
    print(f"  RMSE: {rmse_val:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- Evaluate on TEST Set ---
    ### <-- MODIFIED STEP 4 (cont.): Use the new quantile keys -->
    y_pred_detrended_test_lower = models[0.025].predict(X_test)
    y_pred_detrended_test_median = models[0.5].predict(X_test)
    y_pred_detrended_test_upper = models[0.975].predict(X_test)

    y_pred_final_test_lower = y_pred_detrended_test_lower + test_df['yield_trend']
    y_pred_final_test_median = y_pred_detrended_test_median + test_df['yield_trend']
    y_pred_final_test_upper = y_pred_detrended_test_upper + test_df['yield_trend']

    r2_test = r2_score(y_test_actual, y_pred_final_test_median)
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_pred_final_test_median))
    print(f"\n--- FINAL TEST Performance (Median Prediction) (Years {TEST_START_YEAR}+ Holdout) ---")
    print(f"  R-squared (R2): {r2_test:.4f}")
    print(f"  RMSE: {rmse_test:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- Example Prediction Intervals for the Test Set ---
    print("\n--- Example Prediction Intervals for the Test Set ---")
    y_pred_lower_np = y_pred_final_test_lower.to_numpy()
    y_pred_median_np = y_pred_final_test_median.to_numpy()
    y_pred_upper_np = y_pred_final_test_upper.to_numpy()
    y_test_actual_np = y_test_actual.to_numpy()
    for i in range(5):
        print(
            f"Sample {i + 1}: Lower Bound = {y_pred_lower_np[i]:.2f}, Median = {y_pred_median_np[i]:.2f}, Upper Bound = {y_pred_upper_np[i]:.2f}, Actual = {y_test_actual_np[i]:.2f}")

    # --- Plot and Save Feature Importance (Based on the median model) ---
    try:
        median_model = models[0.5]
        importance_scores = median_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance_scores}).sort_values(
            by='Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        ax.set_title('Feature Importance (Champion Model - Median Prediction)')
        ax.set_xlabel('Feature Importance Score')
        ax.set_ylabel('Features')
        plt.tight_layout()
        os.makedirs(os.path.dirname(IMPORTANCE_PLOT_PATH), exist_ok=True)
        plt.savefig(IMPORTANCE_PLOT_PATH, bbox_inches='tight')
        print(f"\n Feature importance plot saved to {IMPORTANCE_PLOT_PATH}")
    except Exception as e:
        print(f" Warning: Could not save the feature importance plot. Error: {e}")

    # --- Final Model Saving ---
    try:
        os.makedirs(os.path.dirname(LOWER_BOUND_MODEL_PATH), exist_ok=True)
        ### <-- MODIFIED STEP 4 (cont.): Use the new quantile keys for saving -->
        joblib.dump(models[0.025], LOWER_BOUND_MODEL_PATH)
        joblib.dump(models[0.5], MEDIAN_MODEL_PATH)
        joblib.dump(models[0.975], UPPER_BOUND_MODEL_PATH)
        print(f" Final XGBoost  models for prediction intervals successfully trained and saved.")
    except Exception as e:
        print(f" Warning: Could not save the final models. Error: {e}")


if __name__ == "__main__":
    train_validate_and_test_with_intervals()