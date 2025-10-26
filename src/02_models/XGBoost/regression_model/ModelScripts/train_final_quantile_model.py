# File: src/models/train_final_quantile_model.py
# Description: Trains and saves the three definitive XGBoost quantile models (Lower, Median, Upper)
#              for the final prediction interval forecast, using individually tuned hyperparameters.

import pandas as pd
from xgboost import XGBRegressor
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_OUTPUT_DIR = 'src/models'

FEATURE_COLS = [
    # Original Weather Anomaly Features (SEAS5)
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly',
    'antecedent_gdd_sum_anomaly', 'spring_temp_anomaly_forecast',
    'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
    'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast',
    'spring_soil_temp_l1_anomaly_forecast', 'spring_snowfall_anomaly_forecast',
    'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast',
    'summer_runoff_anomaly_forecast', 'summer_soil_temp_l1_anomaly_forecast',
    'summer_snowfall_anomaly_forecast',

    # Original Weather Probability Features (SEAS5)
    'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast',
    'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',

    # Static Geographic & Soil Features
    'lat', 'lon', 'avg_elevation', 'avg_slope',
    'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm',
    'avg_phh2o_0_30cm',

    # Satellite Features
    'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
    'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
    'winter_cropland_snow_cover_days',

    # Lagged Economic Features & Anomalies
    'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
    'producer_price_index_lag1_anomaly', 'seed_price_index_lag1_anomaly',
    'energy_price_index_lag1_anomaly', 'fertilizer_price_index_lag1_anomaly',
    'plant_protection_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',

    # Original V2 Interaction & Polynomial Features
    'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip',
    'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
    'hot_dry_interaction', 'lat_x_summer_temp', 'sandy_soil_x_drought',
    'antecedent_gdd_sum_anomaly_sq', 'spring_temp_prob_warm_forecast_sq',
    'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq',
    'summer_precip_prob_wet_forecast_sq',

    # --- NEWLY AUGMENTED FEATURES (V3) ---

    # 1. WOFOST Hybrid Model Features
    'wofost_forecast_yield_fresh_dt',
    'wofost_forecast_x_profit_margin',
    'has_wofost_data',

    # 2. Diagnostic-Driven Features
    'state_encoded',  # Addresses regional bias
    'summer_precip_anomaly_forecast_sq',  # Addresses "wetness penalty"

    # 3. Granular Weather Features (from daily data)
    'summer_days_precip_gt_20mm',  # Specifically targets extreme wetness
    'summer_days_tmax_gt_30c',     # Specifically targets heatwaves
    'state6_precip_interaction',
    'is_drought_high_clay_in_state_11'
]

# quantile model, based on the results of our final Optuna tuning run.
BEST_PARAMS_LOWER = {
    'n_estimators': 1193, 'learning_rate': 0.035344, 'max_depth': 6,
    'subsample': 0.869864, 'colsample_bytree': 0.664088, 'gamma': 1.689299,
    'min_child_weight': 3, 'random_state': 42, 'n_jobs': -1
}

BEST_PARAMS_MEDIAN = {
    'n_estimators': 648, 'learning_rate': 0.045771, 'max_depth': 5,
    'subsample': 0.736384, 'colsample_bytree': 0.672173, 'gamma': 0.443328,
    'min_child_weight': 8, 'random_state': 42, 'n_jobs': -1
}

BEST_PARAMS_UPPER = {
    'n_estimators': 560, 'learning_rate': 0.098429, 'max_depth': 4,
    'subsample': 0.681244, 'colsample_bytree': 0.722783, 'gamma': 9.998875,
    'min_child_weight': 5, 'random_state': 42, 'n_jobs': -1
}

# Define the quantiles we want to model
QUANTILES = {'lower': 0.025, 'median': 0.5, 'upper': 0.975}


def train_and_save_quantile_models():
    """Loads data, detrends, and trains three separate quantile models."""
    print("--- Starting Final Quantile Model Training Pipeline (with Tuned V5 Params) ---")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {DATA_PATH}.")
        return

    # --- 1. Causal Detrending ---
    print("\n--- Applying Causal Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.ffill().bfill())
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    # Ensure all feature columns exist
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: The following required feature columns are missing in the data file: {missing_cols}")
        return

    X_train = df[FEATURE_COLS]
    y_train = df['kreisYield_detrended']
    print(f"\nTraining on full dataset with {len(X_train)} samples and {len(FEATURE_COLS)} features.")

    # --- 2. Train and Save a Model for Each Quantile ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    for name, alpha in QUANTILES.items():
        print(f"\n--- Training {name.upper()} Model (Quantile: {alpha}) ---")

        # --- CHANGE 3: SELECT THE CORRECT HYPERPARAMETER SET FOR THE CURRENT MODEL ---
        if name == 'lower':
            params = BEST_PARAMS_LOWER
        elif name == 'median':
            params = BEST_PARAMS_MEDIAN
        else:  # upper
            params = BEST_PARAMS_UPPER

        print(f"Using {params['n_estimators']} estimators with a learning rate of {params['learning_rate']:.4f}...")

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            **params
        )

        model.fit(X_train, y_train)

        model_path = os.path.join(MODEL_OUTPUT_DIR, f'final_quantile_model_{name}.joblib')
        joblib.dump(model, model_path)
        print(f"✅ {name.upper()} model saved to {model_path}")

    print("\n--- All Quantile Models Trained and Saved Successfully ---")


if __name__ == "__main__":
    train_and_save_quantile_models()