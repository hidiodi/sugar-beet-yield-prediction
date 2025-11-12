# system_1_biophysical.py
from sklearn.decomposition import PCA

# --- VSM 1 Expert Engine Parameters ---
# Defines the feature set representing the biophysical environment and forecast.
# This combines static soil/geo features, antecedent conditions, and weather forecasts.
VSM1_INPUT_FEATURES = [
    # --- Geographic & Soil (Static) ---
    'lat', 'lon', 'avg_elevation', 'avg_slope', 'avg_clay_0_30cm',
    'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm',

    # --- Antecedent Conditions (Available in March) ---
    'antecedent_frost_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'winter_cropland_ndvi_mean', 'winter_cropland_snow_cover_days',
    'winter_cropland_LST_mean',

    # --- Weather Forecasts (Anomalies & Probabilities) ---
    'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast',
    'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast',
    'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',

    # --- Engineered Biophysical Features & Interactions ---
    'is_summer_forecast_dry',
    'is_extreme_heat_forecast',
    'is_extreme_drought_forecast',
    'drought_x_heat',
    'hot_dry_interaction',
    'lat_x_summer_temp',
    'lon_x_spring_precip',
    'sandy_soil_x_drought',
    'spring_temp_x_spring_precip',

    # --- Polynomial Features for Non-Linear Weather Effects ---
    'antecedent_gdd_sum_anomaly_sq', 'antecedent_gdd_sum_anomaly_cubed',
    'summer_temp_anomaly_forecast_sq', 'summer_temp_anomaly_forecast_cubed',
    'summer_precip_anomaly_forecast_sq', 'summer_precip_anomaly_forecast_cubed',
    'spring_temp_prob_warm_forecast_sq', 'summer_temp_prob_warm_forecast_sq',
    'spring_precip_prob_wet_forecast_sq', 'summer_precip_prob_wet_forecast_sq',
]

# Defines the unsupervised model to be used
VSM1_EXPERT_ENGINE = PCA

# Defines the parameters for the unsupervised model
VSM1_EXPERT_ENGINE_PARAMS = {
    'n_components': 5,  # Increased components for a richer system
    'random_state': 42
}

# --- Artifact Naming ---
VSM1_SCALER_NAME = "scaler_vsm1_biophysical.joblib"
VSM1_ENGINE_NAME = "engine_vsm1_biophysical.joblib"
