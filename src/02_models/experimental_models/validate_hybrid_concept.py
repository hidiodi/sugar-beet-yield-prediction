# File: src/models/validate_hybrid_concept.py
# Description: A fast viability test for the hybrid "Residual Modeling" concept.
# It uses a simple weather model as a proxy for a full crop model and tests if
# a more complex ML model can learn to predict and correct its errors.

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.base import clone
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")

# --- Configuration ---
STATIC_FEATURES_PATH = 'data/05_model_input/stage1_preseason_features.csv'
REPORT_DIR = 'reports/figures/diagnostics'

# --- Model Parameters ---
BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2024
# Use simpler params for this quick test to ensure speed
BASE_PARAMS = {
    'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 4,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': -1
}
SIMPLE_MODEL = XGBRegressor(**BASE_PARAMS, random_state=42)
RESIDUAL_MODEL = XGBRegressor(**BASE_PARAMS, random_state=1337)

# --- Feature Definitions ---
# The "LINTUL-Lite" proxy only sees the most fundamental weather drivers
SIMPLE_MODEL_FEATURES = [
    'antecedent_gdd_sum_anomaly',
    'summer_temp_anomaly_forecast',
    'summer_precip_anomaly_forecast'
]

# The "Error Corrector" model sees everything else
# We must exclude the simple features to prevent data leakage
RESIDUAL_MODEL_FEATURES = [
    'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast',
    'spring_solar_rad_anomaly_forecast', 'spring_evaporation_anomaly_forecast',
    'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
    'spring_snowfall_anomaly_forecast', 'summer_solar_rad_anomaly_forecast',
    'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
    'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast',
    'lat', 'lon', 'avg_elevation', 'avg_slope',
    'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm',
    'avg_phh2o_0_30cm', 'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
    'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
    'winter_cropland_snow_cover_days',
    'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',
    'is_summer_forecast_dry', 'gdd_x_fertilizer_price',
    'spring_temp_x_spring_precip', 'antecedent_gdd_sum_anomaly_sq',
    'summer_heat_x_profit_margin', 'summer_precip_x_input_costs'
]


def run_hybrid_viability_backtest(df: pd.DataFrame):
    """
    Performs a rolling backtest of the hybrid concept.
    Returns the true values and the final hybrid predictions.
    """
    print("--- Starting Hybrid Concept Viability Backtest ---")
    all_true_yields = []
    all_hybrid_preds = []
    all_simple_preds = []

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Hybrid Concept"):
        train_df = df[df['year'] < year_to_predict]
        test_df = df[df['year'] == year_to_predict]
        if test_df.empty or train_df.empty: continue

        y_train = train_df['kreisYield_detrended']
        y_test = test_df['kreisYield_detrended']

        # --- 1. Train the "LINTUL-Lite" Simple Model ---
        simple_model = clone(SIMPLE_MODEL)
        simple_model.fit(train_df[SIMPLE_MODEL_FEATURES], y_train)

        # --- 2. Calculate Historical Residuals on the TRAINING data ---
        # This is the signal our second model will learn
        simple_preds_on_train = simple_model.predict(train_df[SIMPLE_MODEL_FEATURES])
        residuals_train = y_train - simple_preds_on_train

        # --- 3. Train the "Error Corrector" Residual Model ---
        residual_model = clone(RESIDUAL_MODEL)
        residual_model.fit(train_df[RESIDUAL_MODEL_FEATURES], residuals_train)

        # --- 4. Make the Hybrid Forecast on the TEST data ---
        # Get the simple model's baseline prediction
        simple_preds_on_test = simple_model.predict(test_df[SIMPLE_MODEL_FEATURES])

        # Get the complex model's predicted correction
        predicted_residuals_on_test = residual_model.predict(test_df[RESIDUAL_MODEL_FEATURES])

        # The final hybrid prediction is the baseline + correction
        hybrid_preds_detrended = simple_preds_on_test + predicted_residuals_on_test

        # Re-add the trend for final evaluation
        hybrid_preds = hybrid_preds_detrended + test_df['yield_trend']
        true_yields = test_df['kreisYield']
        simple_preds_final = simple_preds_on_test + test_df['yield_trend']

        all_true_yields.append(true_yields)
        all_hybrid_preds.append(hybrid_preds)
        all_simple_preds.append(simple_preds_final)

    return (
        pd.concat(all_true_yields, ignore_index=True),
        pd.concat(all_hybrid_preds, ignore_index=True),
        pd.concat(all_simple_preds, ignore_index=True)
    )


def main():
    """Main function to run the test and print the verdict."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("===== STARTING: Hybrid Model Viability Test =====")

    try:
        df = pd.read_csv(STATIC_FEATURES_PATH)
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: Could not find the feature file. Details: {e}")
        return

    # Apply detrending once
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']

    true_yields, hybrid_preds, simple_preds = run_hybrid_viability_backtest(df)

    # --- Final Evaluation and Verdict ---
    r2_simple_model = r2_score(true_yields, simple_preds)
    r2_hybrid_model = r2_score(true_yields, hybrid_preds)

    # This is our current champion's score to beat
    current_champion_r2 = 0.5335

    print("\n\n--- VIABILITY TEST RESULTS ---")
    print(f"  -> R² of 'LINTUL-Lite' Simple Weather Model: {r2_simple_model:.4f}")
    print(f"  -> R² of Final Hybrid Model:                 {r2_hybrid_model:.4f}")
    print(f"  -> R² of Current Champion (Raw Ensemble):    {current_champion_r2:.4f}")

    print("\n--- VERDICT ---")
    if r2_hybrid_model > current_champion_r2:
        improvement = (r2_hybrid_model - current_champion_r2) / current_champion_r2
        print(f"✅ SUCCESS: The hybrid concept is VIABLE and SUPERIOR.")
        print(f"   The final hybrid model showed a {improvement:.2%} improvement over the current champion.")
        print(f"   This provides strong evidence to proceed with implementing a real crop model like LINTUL.")
    elif r2_hybrid_model > r2_simple_model:
        print(f"🟡 PARTIAL SUCCESS: The hybrid concept is VIABLE but not yet superior.")
        print(f"   The error corrector model successfully improved upon the simple model, but the combined result")
        print(f"   did not beat our highly optimized single-step ensemble.")
        print(f"   This suggests a real crop model (which is more accurate than our proxy) is likely to succeed.")
    else:
        print(f"❌ FAILURE: The hybrid concept is NOT VIABLE with this feature set.")
        print(f"   The error corrector model could not improve upon the simple model.")
        print(f"   This indicates the simple model's errors are too random for the other features to predict.")


if __name__ == "__main__":
    main()