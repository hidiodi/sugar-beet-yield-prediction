# system_3_control.py
from sklearn.decomposition import PCA

# --- VSM 3 Expert Engine Parameters ---
# Defines the feature set for the Economic Battery: the farm's economic
# health, cost structure, and ability to afford inputs.
VSM3_INPUT_FEATURES = [
    # --- Lagged Economic State ---
    'producer_price_index_lag1',
    'cost_of_inputs_lag1',
    'profit_margin_proxy_lag1',

    # --- Economic Momentum ---
    'profit_margin_momentum',
    'cost_of_inputs_momentum',

    # --- Engineered Interactions: How Economics Modulate Weather Response ---
    'gdd_x_fertilizer_price',
    'summer_heat_x_profit_margin',
    'summer_precip_x_input_costs',
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
