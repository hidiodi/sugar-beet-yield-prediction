# system_5_policy.py
from sklearn.decomposition import PCA

# --- VSM 5 Expert Engine Parameters ---
# Defines the feature set for the policy and regulatory environment.
#
# !! DATA GAP !!
# As identified in the analysis, there are no direct proxies for policy
# (e.g., CAP subsidies) or regulations (e.g., NVZs) in the current dataset.
# This list is intentionally left empty. The pipeline will gracefully bypass it.
# See the main report's "Future Work" section for a plan to fill this gap.
VSM5_INPUT_FEATURES = []

# Defines the unsupervised model to be used
VSM5_EXPERT_ENGINE = PCA

# Defines the parameters for the unsupervised model
VSM5_EXPERT_ENGINE_PARAMS = {
    'n_components': 0, # Set to 0 to reflect the empty feature list
    'random_state': 42
}

# --- Artifact Naming ---
VSM5_SCALER_NAME = "scaler_vsm5_policy.joblib"
VSM5_ENGINE_NAME = "engine_vsm5_policy.joblib"
