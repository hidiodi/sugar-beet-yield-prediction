# system_2_coordination.py
from sklearn.decomposition import PCA

# --- VSM 2 Expert Engine Parameters ---
# Defines the feature set representing farmer coordination and management.
# With the current data, this is limited to the area of sugar beet cultivation.
VSM2_INPUT_FEATURES = [
    'zuckerrben'  # Area cultivated with sugar beets in ha
]

# Defines the unsupervised model to be used
VSM2_EXPERT_ENGINE = PCA

# Defines the parameters for the unsupervised model
# With only one feature, PCA is acting as a simple scaler.
# This can be expanded when new management features are added.
VSM2_EXPERT_ENGINE_PARAMS = {
    'n_components': 1,
    'random_state': 42
}

# --- Artifact Naming ---
VSM2_SCALER_NAME = "scaler_vsm2_coordination.joblib"
VSM2_ENGINE_NAME = "engine_vsm2_coordination.joblib"
