# 01_train_economic_battery_engine.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from vsm_cybernetic_model.configs import main_config as cfg
from vsm_cybernetic_model.configs import system_3_control as sys_cfg

def train_economic_battery_engine():
    """
    Trains the VSM System 3 (Control / Economic Battery) expert engine.

    This function loads the foundational features related to the economic
    structure and pressures of farms, scales them, trains an unsupervised
    model (e.g., PCA), and saves the fitted scaler and model artifacts.
    """
    print("--- Training VSM System 3 (Economic Battery) Expert Engine ---")

    # 1. Load Data
    try:
        df = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN)
        print(f"Loaded foundational features from '{cfg.FOUNDATIONAL_FEATURES_HUMAN}'")
    except FileNotFoundError:
        print(f"Error: Foundational features file not found at '{cfg.FOUNDATIONAL_FEATURES_HUMAN}'.")
        print("Please run the preparation scripts first.")
        return

    # 2. Select and Preprocess Features
    X = df[sys_cfg.VSM3_INPUT_FEATURES].dropna()
    print(f"Selected {len(X.columns)} features for VSM 3 engine.")

    if X.empty:
        print("Error: No data available for training after dropping NaNs.")
        return

    # 3. Scale Data
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train Unsupervised Model
    print(f"Training {sys_cfg.VSM3_EXPERT_ENGINE.__name__} model...")
    engine = sys_cfg.VSM3_EXPERT_ENGINE(**sys_cfg.VSM3_EXPERT_ENGINE_PARAMS)
    engine.fit(X_scaled)

    # 5. Save Artifacts
    scaler_path = cfg.STAGE_1_EXPERTS_DIR / sys_cfg.VSM3_SCALER_NAME
    engine_path = cfg.STAGE_1_EXPERTS_DIR / sys_cfg.VSM3_ENGINE_NAME

    print(f"Saving scaler to '{scaler_path}'")
    joblib.dump(scaler, scaler_path)

    print(f"Saving trained engine to '{engine_path}'")
    joblib.dump(engine, engine_path)

    print("--- VSM System 3 Engine training complete ---")

if __name__ == "__main__":
    train_economic_battery_engine()
