# File: src/models/train_final_ngboost_model.py
# Description: Trains a definitive NGBoost model by fitting it on the RESIDUALS
#              of the primary time-series forecast. This version includes
#              proper data handling and a validation set for stable training.

import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_OUTPUT_PATH = os.path.join('src/models', 'final_ngboost_model.joblib')

FEATURE_COLS = [
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'spring_temp_anomaly_forecast',
    'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast', 'spring_evaporation_anomaly_forecast',
    'spring_runoff_anomaly_forecast',
    'spring_soil_temp_l1_anomaly_forecast', 'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast',
    'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
    'summer_soil_temp_l1_anomaly_forecast',
    'summer_snowfall_anomaly_forecast', 'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast',
    'summer_temp_prob_warm_forecast',
    'summer_precip_prob_wet_forecast', 'lat', 'lon', 'avg_elevation', 'avg_slope', 'avg_bdod_0_30cm', 'avg_clay_0_30cm',
    'avg_sand_0_30cm',
    'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
    'winter_cropland_LST_mean',
    'winter_cropland_LST_anomaly', 'winter_cropland_snow_cover_days', 'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
    'producer_price_index_lag1_anomaly', 'seed_price_index_lag1_anomaly', 'energy_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly',
    'plant_protection_price_index_lag1_anomaly', 'fertilizer_price_index_lag1_anomaly_capped',
    'is_fertilizer_price_extreme',
    'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip', 'summer_heat_x_profit_margin',
    'summer_precip_x_input_costs',
    'hot_dry_interaction', 'lat_x_summer_temp', 'sandy_soil_x_drought', 'antecedent_gdd_sum_anomaly_sq',
    'spring_temp_prob_warm_forecast_sq',
    'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq', 'summer_precip_prob_wet_forecast_sq',
    'wofost_forecast_x_profit_margin', 'has_wofost_data', 'state_encoded', 'summer_precip_anomaly_forecast_sq',
    'summer_days_precip_gt_20mm',
    'summer_days_tmax_gt_30c', 'is_drought_high_clay_in_state_11', 'state6_precip_interaction', 'nao_winter_avg',
    'sca_winter_avg', 'enso_mei_winter_avg',
    'stage1_forecast'
]


def train_and_save_ngboost_model():
    """Trains a single NGBoost model to predict the distribution of residuals."""
    print("--- Starting NGBoost Residual Fitting Pipeline ---")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {DATA_PATH}.")
        return

    print("\n--- Calculating Forecast Residuals ---")
    df.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df['forecast_residual'] = df['kreisYield'] - df['stage1_forecast']

    # CRITICAL FIX: Only drop rows where essential data for the target is missing.
    # NGBoost can handle NaNs in other feature columns. This prevents massive data loss.
    df.dropna(subset=['stage1_forecast', 'forecast_residual', 'kreisYield'], inplace=True)
    print(" -> Target variable (forecast_residual) created.")

    X = df[FEATURE_COLS]
    y = df['forecast_residual']
    print(f"\nTotal samples available: {len(X)}")

    # CRITICAL FIX: Create a time-series-aware validation set for early stopping.
    # This prevents overfitting and stabilizes training.
    train_end_idx = int(len(X) * 0.85)
    X_train, y_train = X[:train_end_idx], y[:train_end_idx]
    X_val, y_val = X[train_end_idx:], y[train_end_idx:]
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    # Initialize the NGBoost model with a validation set for early stopping
    ngb_model = NGBRegressor(
        Dist=Normal,
        n_estimators=500,
        learning_rate=0.05,
        verbose=True,
        random_state=42,
        minibatch_frac=0.8  # Add regularization
    )

    print("\n--- Training NGBoost Model with Validation Set ---")
    ngb_model.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=20)

    # Save the final model
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(ngb_model, MODEL_OUTPUT_PATH)
    print(f"\n✅ NGBoost model saved to {MODEL_OUTPUT_PATH}")
    print("\n--- NGBoost Model Trained and Saved Successfully ---")


if __name__ == "__main__":
    train_and_save_ngboost_model()