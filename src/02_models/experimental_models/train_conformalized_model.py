# File: src/models/train_conformalized_model.py
# Description: Calculates and saves the ADAPTIVE (multiplicative) adjustment factor for CQR+.

import pandas as pd
import joblib
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
LOWER_MODEL_PATH = os.path.join('src/models', 'final_quantile_model_lower.joblib')
UPPER_MODEL_PATH = os.path.join('src/models', 'final_quantile_model_upper.joblib')
# Renamed to reflect it's now a multiplier, not a raw value
ADJUSTMENT_FACTOR_PATH = os.path.join('src/models', 'conformal_multiplier.joblib')

# We can likely lower the target slightly because adaptive is more efficient,
# but let's stick to 97% first to see if we can get high coverage with tighter average intervals.
TARGET_COVERAGE = 0.97
CALIBRATION_YEARS = 8 # Using more years for stability

FEATURE_COLS = [
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly',
    'antecedent_gdd_sum_anomaly', 'spring_temp_anomaly_forecast',
    'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
    'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast',
    'spring_soil_temp_l1_anomaly_forecast', 'spring_snowfall_anomaly_forecast',
    'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast',
    'summer_runoff_anomaly_forecast', 'summer_soil_temp_l1_anomaly_forecast',
    'summer_snowfall_anomaly_forecast', 'spring_temp_prob_warm_forecast',
    'spring_precip_prob_wet_forecast', 'spring_solar_rad_prob_wet_forecast',
    'spring_evaporation_prob_wet_forecast', 'spring_runoff_prob_wet_forecast',
    'spring_soil_temp_l1_prob_warm_forecast', 'spring_snowfall_prob_wet_forecast',
    'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',
    'summer_solar_rad_prob_wet_forecast', 'summer_evaporation_prob_wet_forecast',
    'summer_runoff_prob_wet_forecast', 'summer_soil_temp_l1_prob_warm_forecast',
    'summer_snowfall_prob_wet_forecast', 'lat', 'lon', 'avg_elevation',
    'avg_slope', 'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm',
    'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'avg_bdod_0_100cm',
    'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm',
    'avg_phh2o_0_100cm', 'winter_cropland_ndvi_mean',
    'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean',
    'winter_cropland_LST_anomaly', 'winter_cropland_snow_cover_days',
    'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',
    'is_summer_forecast_dry', 'gdd_x_fertilizer_price',
    'spring_temp_x_spring_precip', 'antecedent_gdd_sum_anomaly_sq',
    'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
    'spring_temp_prob_warm_forecast_sq', 'summer_temp_prob_warm_forecast_sq',
    'spring_precip_prob_wet_forecast_sq', 'summer_precip_prob_wet_forecast_sq'
]

def calculate_and_save_adaptive_factor():
    print("--- Starting ADAPTIVE CQR+ Calibration ---")
    try:
        df = pd.read_csv(DATA_PATH)
        model_lower = joblib.load(LOWER_MODEL_PATH)
        model_upper = joblib.load(UPPER_MODEL_PATH)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    print("\n--- Applying Causal Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    max_year = df['year'].max()
    calibration_start_year = max_year - CALIBRATION_YEARS + 1
    calibration_df = df[df['year'] >= calibration_start_year].copy()
    print(f"\nUsing data from {calibration_start_year}-{max_year} for calibration ({len(calibration_df)} samples).")

    X_calib = calibration_df[FEATURE_COLS]
    y_calib = calibration_df['kreisYield_detrended']

    # 1. Get Raw Predictions
    raw_lower = model_lower.predict(X_calib)
    raw_upper = model_upper.predict(X_calib)

    # 2. Calculate Raw Widths (the normalization factor)
    # Ensure width is at least a small positive number to avoid division by zero
    raw_widths = np.maximum(raw_upper - raw_lower, 1.0)

    # 3. Calculate Raw Conformity Scores
    raw_scores = np.maximum(raw_lower - y_calib, y_calib - raw_upper)

    # 4. Calculate ADAPTIVE (Normalized) Scores
    # This is the key difference: we divide the error by the predicted width.
    normalized_scores = raw_scores / raw_widths

    # 5. Find the Multiplier (q_mult)
    n = len(calibration_df)
    alpha = 1 - TARGET_COVERAGE
    quantile_to_find = np.ceil((1 - alpha) * (n + 1)) / n
    quantile_to_find = min(quantile_to_find, 1.0)

    q_mult = np.quantile(normalized_scores, quantile_to_find, method='higher')

    print(f"\nCalculated ADAPTIVE multiplier (q_mult) for {TARGET_COVERAGE:.0%} target: {q_mult:.4f}")
    print("(This means we will widen the raw interval by this percentage on each side)")

    joblib.dump(q_mult, ADJUSTMENT_FACTOR_PATH)
    print(f"✅ Multiplier saved to {ADJUSTMENT_FACTOR_PATH}")

if __name__ == "__main__":
    calculate_and_save_adaptive_factor()