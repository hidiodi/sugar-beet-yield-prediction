# File: src/models/base_model.py
# Description: DEFINITIVE MODEL V2. Uses an enhanced feature set based on model analysis to improve robustness.

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- V2 MODEL PATHS ---
# Updated to prevent overwriting the previous champion model.
MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion_v2.joblib')
IMPORTANCE_PLOT_PATH = os.path.join('reports/figures', 'feature_importance_champion_v2.png')

# --- BEST HYPERPARAMETERS (UNCHANGED) ---
# Note: Re-tuning might yield further gains with the new feature set.
BEST_PARAMS = {
    'colsample_bytree': 0.8223306320976561,
    'learning_rate': 0.020282652208788696,
    'max_depth': 4,
    'n_estimators': 448,
    'subsample': 0.8049549320516778
}


def train_validate_and_test():
    """
    Loads data, applies detrending, performs a 3-way time-series split,
    trains the model, evaluates on validation and test sets, and saves the final model.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # ============================ V2 FEATURE SET ============================
    # Updated based on insights from the advanced analysis script.
    feature_cols = [
        # --- Core Geographic & Soil Features ---
        'avg_elevation',
        'avg_soil_pawc',

        # --- Core Economic Drivers ---
        'profit_margin_proxy_lag1',
        'cost_of_inputs_lag1',
        'plant_protection_price_index_lag1_anomaly',

        # --- Core Weather & Climate Features ---
        'antecedent_frost_days_anomaly',
        'antecedent_heavy_precip_days_anomaly',
        'antecedent_gdd_sum_anomaly',
        'spring_temp_anomaly_forecast',
        'spring_precip_anomaly_forecast',
        'summer_temp_anomaly_forecast',
        'summer_precip_anomaly_forecast',
        'spring_temp_prob_warm_forecast',
        'spring_precip_prob_wet_forecast',
        'summer_temp_prob_warm_forecast',
        'summer_precip_prob_wet_forecast',

        # --- Original Interaction & Polynomial Features ---
        'summer_heat_x_profit_margin',
        'summer_precip_x_input_costs',
        'spring_temp_anomaly_forecast_sq',
        'summer_temp_anomaly_forecast_sq',
        'spring_precip_anomaly_forecast_sq',
        'summer_precip_anomaly_forecast_sq',

        # --- REMOVED ---
        # 'fertilizer_price_index_lag1_anomaly', # Removed due to high sensitivity and causing model brittleness.

        # +++ NEW FEATURES FROM ANALYSIS (V2) +++
        # Capped version to reduce impact of extreme outliers.
        'fertilizer_price_index_lag1_anomaly_capped',
        # Binary flag to help model handle extreme price shocks specifically.
        'is_fertilizer_price_extreme',
        # Binary flag to clarify ambiguous summer precipitation signal.
        'is_summer_forecast_dry',
        # Explicit interaction for the powerful 'Good Weather + Cheap Inputs' effect.
        'gdd_x_fertilizer_price',
        # Explicit interaction for the 'Warm & Wet Spring' effect.
        'spring_temp_x_spring_precip',
        # Added to help model better capture the non-linear GDD response.
        'antecedent_gdd_sum_anomaly_sq',
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
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x
    )
    df.dropna(subset=['yield_trend'], inplace=True)
    df[detrended_target_col] = df[target_col] - df['yield_trend']
    print(" -> Detrending complete.")

    # --- Time-Series Split (Consistent with Analysis Script) ---
    validation_start_year = 2007
    test_start_year = 2015

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

    # --- Train the XGBoost Model (ONLY ON TRAINING DATA) ---
    xgb = XGBRegressor(
        objective='reg:squarederror',
        **BEST_PARAMS,  # Unpack the dictionary of hyperparameters
        random_state=42,
        n_jobs=-1,
    )
    print("\n--- Training New V2 Model ---")
    xgb.fit(X_train, y_train)

    # --- Evaluate on VALIDATION Set ---
    y_pred_detrended_val = xgb.predict(X_validation)
    y_pred_final_val = y_pred_detrended_val + validation_df['yield_trend']
    r2_val = r2_score(y_validation_actual, y_pred_final_val)
    rmse_val = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final_val))
    print("-------------------------------------------------")
    print(f"Validation Performance (Years {validation_start_year} to {test_start_year - 1})")
    print(f"  R-squared (R2): {r2_val:.4f}")
    print(f"  RMSE: {rmse_val:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- Evaluate on TEST Set ---
    y_pred_detrended_test = xgb.predict(X_test)
    y_pred_final_test = y_pred_detrended_test + test_df['yield_trend']
    r2_test = r2_score(y_test_actual, y_pred_final_test)
    rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_pred_final_test))
    print(f"\n--- FINAL V2 TEST Performance (Years {test_start_year}+ Holdout) ---")
    print(f"  R-squared (R2): {r2_test:.4f}")
    print(f"  RMSE: {rmse_test:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- Plot and Save Feature Importance (Based on training data) ---
    try:
        importance_scores = xgb.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance_scores
        }).sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 12))  # Increased height for more features
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        ax.set_title('Feature Importance (Champion Model V2)')
        ax.set_xlabel('Feature Importance Score (Gini Importance)')
        ax.set_ylabel('Features')
        plt.tight_layout()

        os.makedirs(os.path.dirname(IMPORTANCE_PLOT_PATH), exist_ok=True)
        plt.savefig(IMPORTANCE_PLOT_PATH, bbox_inches='tight')
        print(f"\n✅ V2 Feature importance plot saved to {IMPORTANCE_PLOT_PATH}")
    except Exception as e:
        print(f"❌ Warning: Could not save the V2 feature importance plot. Error: {e}")

    # --- Final Model Training and Saving ---
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(xgb, MODEL_PATH)
        print(f"✅ Final XGBoost V2 model successfully trained and saved to {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Warning: Could not save the final V2 model. Error: {e}")


if __name__ == "__main__":
    train_validate_and_test()