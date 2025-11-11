# system_1_biophysical.py
import numpy as np
from sklearn.decomposition import PCA

# --- RPP Simulation Parameters ---
# Placeholder for WOFOST configurations. In a real scenario, this would be more complex.
WOFOST_CROP_FILE = "sugarbeet.cab"
WOFOST_SITE_FILE = "ec5.site"

# --- VSM 1 Expert Engine Parameters ---
# Defines the feature set that will be used to train the biophysical expert engine
# This combines static features (e.g., soil) and dynamic RPP outputs
VSM1_INPUT_FEATURES = [
    'Soil_Water_Battery',
    'RPP_mean_yield',
    'RPP_biomass_volatility',
    'RPP_cumulative_stress',
]

# Defines the unsupervised model to be used
VSM1_EXPERT_ENGINE = PCA

# Defines the parameters for the unsupervised model
VSM1_EXPERT_ENGINE_PARAMS = {
    'n_components': 2,
    'random_state': 42
}

# --- Artifact Naming ---
# Standardized naming for the saved model files
VSM1_SCALER_NAME = "scaler_vsm1_biophysical.joblib"
VSM1_ENGINE_NAME = "engine_vsm1_biophysical.joblib"
