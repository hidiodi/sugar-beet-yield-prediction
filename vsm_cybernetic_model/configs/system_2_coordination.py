# system_2_coordination.py
from sklearn.decomposition import PCA

# --- VSM 2 Expert Engine Parameters ---
# Defines the feature set that represents farmer coordination activities
VSM2_INPUT_FEATURES = [
    'avg_sowing_date_doy',
    'avg_harvest_date_doy',
    'crop_competition_ratio',
    'sugar_beet_specialization'
]

# Defines the unsupervised model to be used
VSM2_EXPERT_ENGINE = PCA

# Defines the parameters for the unsupervised model
VSM2_EXPERT_ENGINE_PARAMS = {
    'n_components': 1,
    'random_state': 42
}

# --- Artifact Naming ---
VSM2_SCALER_NAME = "scaler_vsm2_coordination.joblib"
VSM2_ENGINE_NAME = "engine_vsm2_coordination.joblib"
