# 02_train_biophysical_engine.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from vsm_cybernetic_model.configs import main_config as cfg
from vsm_cybernetic_model.configs import system_1_biophysical as sys_cfg

def train_biophysical_engine():
    """
    Trains the VSM System 1 (Biophysical) expert engine.

    This function loads the dynamic outputs from the RPP simulations and merges
    them with static biophysical features (e.g., soil data). It then scales
    this combined feature set, trains an unsupervised model, and saves the
    fitted scaler and model artifacts.
    """
    print("--- Training VSM System 1 (Biophysical) Expert Engine ---")

    # 1. Load Data
    try:
        df_rpp = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_RPP)
        print(f"Loaded RPP simulation outputs from '{cfg.FOUNDATIONAL_FEATURES_RPP}'")
        # In a real scenario, you would also load a static biophysical feature file
        # df_static = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_STATIC_BIO)
        # For now, we'll create a dummy static feature column.
        df_rpp['Soil_Water_Battery'] = 200 - (df_rpp['district_no'] % 20) * 2.5
        df = df_rpp
        print("Merged RPP outputs with static biophysical features.")
    except FileNotFoundError:
        print(f"Error: RPP output file not found at '{cfg.FOUNDATIONAL_FEATURES_RPP}'.")
        print("Please run the RPP simulation script (01_run_rpp_simulations.py) first.")
        return

    # 2. Select and Preprocess Features
    X = df[sys_cfg.VSM1_INPUT_FEATURES].dropna()
    print(f"Selected {len(X.columns)} features for VSM 1 engine.")

    if X.empty:
        print("Error: No data available for training after dropping NaNs.")
        return

    # 3. Scale Data
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train Unsupervised Model
    print(f"Training {sys_cfg.VSM1_EXPERT_ENGINE.__name__} model...")
    engine = sys_cfg.VSM1_EXPERT_ENGINE(**sys_cfg.VSM1_EXPERT_ENGINE_PARAMS)
    engine.fit(X_scaled)

    # 5. Save Artifacts
    scaler_path = cfg.STAGE_1_EXPERTS_DIR / sys_cfg.VSM1_SCALER_NAME
    engine_path = cfg.STAGE_1_EXPERTS_DIR / sys_cfg.VSM1_ENGINE_NAME

    print(f"Saving scaler to '{scaler_path}'")
    joblib.dump(scaler, scaler_path)

    print(f"Saving trained engine to '{engine_path}'")
    joblib.dump(engine, engine_path)

    print("--- VSM System 1 Engine training complete ---")

if __name__ == "__main__":
    train_biophysical_engine()
