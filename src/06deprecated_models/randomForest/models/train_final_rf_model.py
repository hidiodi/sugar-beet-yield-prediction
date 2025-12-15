# File: src/models/train_final_rf_model.py
# Description: Trains and saves a single, definitive Random Forest model.
#              Quantiles will be derived from its individual trees during backtesting/prediction.

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_OUTPUT_DIR = 'src/models'
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, 'final_rf_model.joblib')

# Use the exact same V2 feature set
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

# NOTE: These parameters MUST be tuned for RF. These are just a starting point.
# n_estimators: Number of trees. More is usually better but slower. 500 is a good start.
# max_features: 'sqrt' is a common, robust choice.
# min_samples_leaf: Very important for quantile stability. Don't set it to 1.
#                   Try values between 5 and 15 to get stable leaf predictions.
BEST_RF_PARAMS = {
    'n_estimators': 500,
    'max_features': 'sqrt',
    'min_samples_leaf': 10,  # Larger leaf size for more stable quantiles
    'random_state': 42,
    'n_jobs': -1
}

def train_and_save_rf_model():
    """Loads data, detrends, and trains a single Random Forest model."""
    print("--- Starting Final Random Forest Model Training Pipeline ---")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {DATA_PATH}.")
        return

    # --- 1. Causal Detrending (Identical to XGBoost) ---
    print("\n--- Applying Causal Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    X_train = df[FEATURE_COLS]
    y_train = df['kreisYield_detrended']
    print(f"\nTraining on full dataset with {len(X_train)} samples.")

    # --- 2. Train and Save One Model ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    print(f"\n--- Training RANDOM FOREST Model ---")

    model = RandomForestRegressor(**BEST_RF_PARAMS)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✅ RANDOM FOREST model saved to {MODEL_SAVE_PATH}")
    print("\n--- RF Model Trained and Saved Successfully ---")


if __name__ == "__main__":
    train_and_save_rf_model()