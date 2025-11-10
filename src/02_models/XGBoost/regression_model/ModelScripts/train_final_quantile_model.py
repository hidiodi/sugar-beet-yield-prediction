# File: src/models/train_final_quantile_model.py
# FINAL VERSION: Removes all heuristics (sample_weights) and reverts to a
#                single, robust hyperparameter set. Relies purely on the
#                quantile objective and a proper CQR calibration step.

import pandas as pd
from xgboost import XGBRegressor
import os
import joblib
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_OUTPUT_DIR = 'src/models'

FEATURE_COLS = [
    # --- Original SEAS5 Weather Anomaly Features (Antecedent & Seasonal) ---
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
    'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
    'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
    'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast',

    # --- Original SEAS5 Weather Probability Features ---
    'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast', 'summer_temp_prob_warm_forecast',
    'summer_precip_prob_wet_forecast',

    # --- Static Geographic & Soil Features ---
    'lat', 'lon', 'avg_elevation', 'avg_slope', 'avg_bdod_0_30cm', 'avg_clay_0_30cm',
    'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm',

    # --- Satellite Features (Early Season Condition) ---
    'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean',
    'winter_cropland_LST_anomaly', 'winter_cropland_snow_cover_days',

    # --- Teleconnection Indices ---
    'nao_winter_avg', 'sca_winter_avg', 'enso_mei_winter_avg',

    # --- Lagged Economic Features & Anomalies ---
    'profit_margin_proxy_lag1', 'cost_of_inputs_lag1', 'producer_price_index_lag1_anomaly',
    'seed_price_index_lag1_anomaly', 'energy_price_index_lag1_anomaly',
    #'fertilizer_price_index_lag1_anomaly',
    'plant_protection_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',

    # --- Stage 1 Model & Hybrid Features ---
    'stage1_forecast',  # Note: This is the column name from the file, used as 'stage1_forecast'
    'wofost_forecast_x_profit_margin',
    'has_wofost_data',

    # --- General Regional & Temporal Features ---
    'state_encoded',
    'year_trend',

    # --- Original Interaction & Polynomial Features ---
    'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip', 'summer_heat_x_profit_margin',
    'summer_precip_x_input_costs',
    #'hot_dry_interaction',
    'lat_x_summer_temp', 'sandy_soil_x_drought',
    'antecedent_gdd_sum_anomaly_sq', 'spring_temp_prob_warm_forecast_sq',
    'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq',
    'summer_precip_prob_wet_forecast_sq', 'summer_precip_anomaly_forecast_sq',

    # --- NEW Physiologically-Grounded Features for Extremes ---
    'CASDI_Phase2_Count',  # Compounded Abiotic Stress (Heat & Drought)
    'NMSD_Phase2_Count',  # Nighttime Metabolic Stress Days
    'OSAW_Phase2_Count',  # Optimal Sugar Accumulation Window
    'ECES_Phase1_Cumulative',  # Early Canopy Establishment Stress
    #'summer_days_tmax_gt_30c'  # Retained as a simple, direct measure of heat
]
# Use a single, validated set of parameters for ALL models.
BEST_PARAMS = {
    'n_estimators': 914, 'learning_rate': 0.026114, 'max_depth': 5,
    'subsample': 0.922850, 'colsample_bytree': 0.811573, 'gamma': 1.830853,
    'min_child_weight': 2, 'random_state': 42, 'n_jobs': -1
}

QUANTILES = {'lower': 0.025, 'median': 0.5, 'upper': 0.975}


def train_and_save_quantile_models():
    """Trains quantile models using a unified, robust approach."""
    print("--- Starting Final Residual Fitting Pipeline (FINAL - No Heuristics) ---")

    df = pd.read_csv(DATA_PATH)
    df.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df['forecast_residual'] = df['kreisYield'] - df['stage1_forecast']
    df.dropna(subset=['stage1_forecast', 'forecast_residual'], inplace=True)
    df.dropna(subset=FEATURE_COLS, inplace=True)

    X_train = df[FEATURE_COLS]
    y_train = df['forecast_residual']
    print(f"\nTraining on {len(X_train)} samples to predict the forecast residuals.")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    for name, alpha in QUANTILES.items():
        print(f"\n--- Training {name.upper()} Residual Model (Quantile: {alpha}) ---")

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            **BEST_PARAMS
        )

        # CRITICAL CHANGE: NO SAMPLE WEIGHTS.
        # Let the model learn the quantiles directly from the data distribution.
        model.fit(X_train, y_train)

        model_path = os.path.join(MODEL_OUTPUT_DIR, f'final_quantile_model_{name}.joblib')
        joblib.dump(model, model_path)
        print(f"✅ {name.upper()} model saved to {model_path}")

    print("\n--- All Residual Models Trained and Saved Successfully ---")


if __name__ == "__main__":
    train_and_save_quantile_models()