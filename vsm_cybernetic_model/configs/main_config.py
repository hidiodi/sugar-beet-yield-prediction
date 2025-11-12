# main_config.py
from pathlib import Path

# --- Base Directories ---
# A single point of change for the entire module's file structure
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data"  # Assuming 'data' is outside the module

# --- Raw Data Paths (Inputs to Preparation) ---
# To be populated by the user
RAW_DATA_DIR = DATA_DIR / "01_raw"

# --- Intermediate Data Paths (Outputs of Preparation / Inputs to Models) ---
INTERMEDIATE_DATA_DIR = DATA_DIR / "02_intermediate"
INTERMEDIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
FOUNDATIONAL_FEATURES_HUMAN = INTERMEDIATE_DATA_DIR / "vsm_human_system_features.csv"
FOUNDATIONAL_FEATURES_RPP = INTERMEDIATE_DATA_DIR / "vsm_rpp_outputs.csv"
FOUNDATIONAL_FEATURES_STATIC_BIO = INTERMEDIATE_DATA_DIR / "vsm_static_biophysical_features.csv"


# --- Primary Data Paths (Outputs of Expert Engines) ---
PRIMARY_DATA_DIR = DATA_DIR / "03_primary"
PRIMARY_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Model Input Paths (Final Assembled Matrix) ---
MODEL_INPUT_DIR = DATA_DIR / "05_model_input"
MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_FEATURES_PATH = MODEL_INPUT_DIR / "vsm_final_features.csv"


# --- Model Artifacts Paths ---
MODEL_ARTIFACTS_DIR = BASE_DIR / "models"
STAGE_1_EXPERTS_DIR = MODEL_ARTIFACTS_DIR / "stage_1_experts"
STAGE_1_EXPERTS_DIR.mkdir(parents=True, exist_ok=True)

STAGE_2_REGULATOR_DIR = MODEL_ARTIFACTS_DIR / "stage_2_regulator"
STAGE_2_REGULATOR_DIR.mkdir(parents=True, exist_ok=True)
REGULATOR_MODEL_PATH = STAGE_2_REGULATOR_DIR / "xgb_regulator.joblib"

# --- Verification Outputs ---
VERIFICATION_DIR = BASE_DIR.parent / "reports" / "verification"
VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
