# File: src/models/train_final_quantile_model.py
# Description: Trains and saves the three definitive XGBoost quantile models (Lower, Median, Upper)
#              for the final prediction interval forecast.

import pandas as pd
from xgboost import XGBRegressor
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_OUTPUT_DIR = 'src/models'

# Use the final V2 feature set for maximum performance
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

# Use the same robust hyperparameters from your champion base model
BEST_PARAMS = {
    'n_estimators': 914, 'learning_rate': 0.026114, 'max_depth': 5,
    'subsample': 0.922850, 'colsample_bytree': 0.811573, 'gamma': 1.830853,
    'min_child_weight': 2, 'random_state': 42, 'n_jobs': -1
}

# Define the quantiles we want to model for a 95% prediction interval
QUANTILES = {'lower': 0.025, 'median': 0.5, 'upper': 0.975}


def train_and_save_quantile_models():
    """Loads data, detrends, and trains three separate quantile models."""
    print("--- Starting Final Quantile Model Training Pipeline ---")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {DATA_PATH}.")
        return

    # --- 1. Causal Detrending ---
    print("\n--- Applying Causal Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    # Use all available data for the final training
    X_train = df[FEATURE_COLS]
    y_train = df['kreisYield_detrended']
    print(f"\nTraining on full dataset with {len(X_train)} samples.")

    # --- 2. Train and Save a Model for Each Quantile ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    for name, alpha in QUANTILES.items():
        print(f"\n--- Training {name.upper()} Model (Quantile: {alpha}) ---")

        # This is the key: Set the objective to quantile regression with the correct alpha
        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            **BEST_PARAMS
        )

        model.fit(X_train, y_train)

        # Save the model
        model_path = os.path.join(MODEL_OUTPUT_DIR, f'final_quantile_model_{name}.joblib')
        joblib.dump(model, model_path)
        print(f"✅ {name.upper()} model saved to {model_path}")

    print("\n--- All Quantile Models Trained and Saved Successfully ---")


if __name__ == "__main__":
    train_and_save_quantile_models()