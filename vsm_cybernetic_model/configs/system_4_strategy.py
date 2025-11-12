# system_4_strategy.py
from sklearn.decomposition import PCA

# --- VSM 4 Expert Engine Parameters ---
# Defines the feature set for market strategy and response to market signals.
# In the current dataset, this is primarily proxied by the national yield average.
VSM4_INPUT_FEATURES = [
    'national_avg_yield_lag1' # Proxy for national supply signal
]

# Defines the unsupervised model to be used
VSM4_EXPERT_ENGINE = PCA

# Defines the parameters for the unsupervised model
VSM4_EXPERT_ENGINE_PARAMS = {
    'n_components': 1,
    'random_state': 42
}

# --- Artifact Naming ---
VSM4_SCALER_NAME = "scaler_vsm4_strategy.joblib"
VSM4_ENGINE_NAME = "engine_vsm4_strategy.joblib"
