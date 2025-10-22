# File: src/models/train_final_qrf_model.py
# Description: Trains and saves a single, definitive Quantile Regression Forest (QRF) model.

import pandas as pd
from quantile_forest import RandomForestQuantileRegressor  # <-- KEY CHANGE
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_OUTPUT_DIR = 'src/models'
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, 'final_qrf_model.joblib')

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

# NOTE: QRF parameters are tuned differently.
# min_samples_leaf needs to be larger to create a stable distribution in the leaf.
# Start with these and tune them.
# File: src/models/train_final_qrf_model.py

# Use the same robust hyperparameters we just found
BEST_QRF_PARAMS = {
    'n_estimators': 500,
    'max_features': 'sqrt',
    'min_samples_leaf': 40,
    'max_depth': 7,
    'calibration': True,
    'random_state': 42,
    'n_jobs': -1
}

def train_and_save_qrf_model():
    """Loads data, detrends, and trains a single Quantile Regression Forest."""
    print("--- Starting Final QRF Model Training Pipeline ---")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {DATA_PATH}.")
        return

    # --- 1. Causal Detrending (Identical to before) ---
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
    print(f"\n--- Training QUANTILE REGRESSION FOREST Model ---")

    # Use the RandomForestQuantileRegressor class
    model = RandomForestQuantileRegressor(**BEST_QRF_PARAMS)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✅ QRF model saved to {MODEL_SAVE_PATH}")
    print("\n--- QRF Model Trained and Saved Successfully ---")


if __name__ == "__main__":
    train_and_save_qrf_model()