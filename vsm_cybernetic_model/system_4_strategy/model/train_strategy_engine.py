# 01_train_strategy_engine.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from vsm_cybernetic_model.configs import main_config as cfg
from vsm_cybernetic_model.configs import system_4_strategy as sys_cfg

def train_strategy_engine():
    """
    Trains the VSM System 4 (Strategy / Market) expert engine.

    This function loads foundational features related to market access and
    price signals, scales them, trains an unsupervised model, and saves
    the fitted scaler and model artifacts.
    """
    print("--- Training VSM System 4 (Strategy) Expert Engine ---")

    # 1. Load Data
    try:
        df = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN)
        print(f"Loaded foundational features from '{cfg.FOUNDATIONAL_FEATURES_HUMAN}'")
    except FileNotFoundError:
        print(f"Error: Foundational features file not found at '{cfg.FOUNDATIONAL_FEATURES_HUMAN}'.")
        print("Please run the preparation scripts first.")
        return

    # 2. Select and Preprocess Features
    X = df[sys_cfg.VSM4_INPUT_FEATURES].dropna()
    print(f"Selected {len(X.columns)} features for VSM 4 engine.")

    if X.empty:
        print("Error: No data available for training after dropping NaNs.")
        return

    # 3. Scale Data
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train Unsupervised Model
    print(f"Training {sys_cfg.VSM4_EXPERT_ENGINE.__name__} model...")
    engine = sys_cfg.VSM4_EXPERT_ENGINE(**sys_cfg.VSM4_EXPERT_ENGINE_PARAMS)
    engine.fit(X_scaled)

    # 5. Save Artifacts
    scaler_path = cfg.STAGE_1_EXPERTS_DIR / sys_cfg.VSM4_SCALER_NAME
    engine_path = cfg.STAGE_1_EXPERTS_DIR / sys_cfg.VSM4_ENGINE_NAME

    print(f"Saving scaler to '{scaler_path}'")
    joblib.dump(scaler, scaler_path)

    print(f"Saving trained engine to '{engine_path}'")
    joblib.dump(engine, engine_path)

    print("--- VSM System 4 Engine training complete ---")

if __name__ == "__main__":
    train_strategy_engine()
