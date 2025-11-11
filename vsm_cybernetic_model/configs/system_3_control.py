# system_3_control.py
from sklearn.decomposition import PCA

# --- VSM 3 Expert Engine Parameters ---
# Defines the feature set for the Economic Battery
VSM3_INPUT_FEATURES = [
    'avg_farm_size_ha',
    'land_tenure_ratio',
    'avg_land_price_eur_ha',
    'total_SO_NUTS3',
    'cost_pressure_index',
    'family_labor_ratio'
]

# Defines the unsupervised model to be used
VSM3_EXPERT_ENGINE = PCA

# Defines the parameters for the unsupervised model
VSM3_EXPERT_ENGINE_PARAMS = {
    'n_components': 1,
    'random_state': 42
}

# --- Artifact Naming ---
VSM3_SCALER_NAME = "scaler_vsm3_control.joblib"
VSM3_ENGINE_NAME = "engine_vsm3_control.joblib"
