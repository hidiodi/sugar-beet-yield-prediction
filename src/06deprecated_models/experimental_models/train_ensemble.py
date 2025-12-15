# File: src/models/train_ensemble.py
# Description: Trains and saves a Homogeneous Deep Ensemble of XGBoost quantile models.

import pandas as pd
from xgboost import XGBRegressor
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_OUTPUT_DIR = 'src/models/ensemble' # Save to a dedicated subfolder
N_ENSEMBLE_MODELS = 10 # Number of independent models in the ensemble

# Use the same robust hyperparameters and features
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
BEST_PARAMS = {
    'n_estimators': 914, 'learning_rate': 0.026114, 'max_depth': 5,
    'subsample': 0.922850, 'colsample_bytree': 0.811573, 'gamma': 1.830853,
    'min_child_weight': 2, 'random_state': 42, 'n_jobs': -1
}
QUANTILES = {'lower': 0.025, 'median': 0.5, 'upper': 0.975}

def train_ensemble():
    """Loads data, detrends, and trains N independent quantile models on bootstrap samples."""
    print("--- Starting Deep Ensemble Model Training Pipeline ---")

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {DATA_PATH}.")
        return

    print("\n--- Applying Causal Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    for i in range(N_ENSEMBLE_MODELS):
        print(f"\n--- Training Ensemble Member {i+1}/{N_ENSEMBLE_MODELS} ---")

        # Create a bootstrap sample (sampling with replacement)
        # This is the key to making the models independent.
        bootstrap_df = df.sample(frac=1.0, replace=True, random_state=i) # Use index for reproducibility
        X_train_boot = bootstrap_df[FEATURE_COLS]
        y_train_boot = bootstrap_df['kreisYield_detrended']
        print(f" -> Training on a bootstrap sample of {len(X_train_boot)} data points.")

        for name, alpha in QUANTILES.items():
            print(f"  -> Training {name.upper()} model (Quantile: {alpha})...")

            model = XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=alpha,
                **BEST_PARAMS
            )
            model.fit(X_train_boot, y_train_boot)

            model_path = os.path.join(MODEL_OUTPUT_DIR, f'ensemble_model_{i}_{name}.joblib')
            joblib.dump(model, model_path)
            print(f"  ✅ {name.upper()} model saved to {model_path}")

    print("\n--- All Ensemble Models Trained and Saved Successfully ---")

if __name__ == "__main__":
    train_ensemble()