# system_3_control.py
from sklearn.decomposition import PCA

# --- VSM 3 Expert Engine Parameters ---
# Defines the feature set for the Economic Battery: the farm's economic
# health, cost structure, and ability to afford inputs.
VSM3_INPUT_FEATURES = [
    # Direct Input Costs (from original data)
    'dngemittel',                # Fertilizer costs
    'energie_und_schmierstoffe', # Energy costs
    'pflanzenschutzmittel',        # Pesticide costs
    'saat_und_pflanzgut',        # Seed costs

    # Engineered Economic Features (from pre-processing)
    'producer_price_index_lag1',
    'profit_margin_proxy_lag1',
    'cost_of_inputs_lag1',
    'profit_margin_momentum',
    'cost_of_inputs_momentum'
]

# Defines the unsupervised model to be used
VSM3_EXPERT_ENGINE = PCA

# Defines the parameters for the unsupervised model
VSM3_EXPERT_ENGINE_PARAMS = {
    'n_components': 2, # Increased to capture cost vs. profit dimensions
    'random_state': 42
}

# --- Artifact Naming ---
VSM3_SCALER_NAME = "scaler_vsm3_control.joblib"
VSM3_ENGINE_NAME = "engine_vsm3_control.joblib"
