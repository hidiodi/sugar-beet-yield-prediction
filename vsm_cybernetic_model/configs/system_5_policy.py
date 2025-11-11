# system_5_policy.py
from sklearn.decomposition import PCA

# --- VSM 5 Expert Engine Parameters ---
# Defines the feature set for the policy and regulatory environment
VSM5_INPUT_FEATURES = [
    'percent_UAA_in_NVZ',
    'CAP_Euros_per_Hectare_UAA'
]

# Defines the unsupervised model to be used
VSM5_EXPERT_ENGINE = PCA

# Defines the parameters for the unsupervised model
VSM5_EXPERT_ENGINE_PARAMS = {
    'n_components': 1,
    'random_state': 42
}

# --- Artifact Naming ---
VSM5_SCALER_NAME = "scaler_vsm5_policy.joblib"
VSM5_ENGINE_NAME = "engine_vsm5_policy.joblib"
