import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from vsm_cybernetic_model.configs import main_config as cfg
from vsm_cybernetic_model.configs import system_2_coordination as sys_cfg

def analyze_coordination_engine():
    print("--- Analyzing VSM System 2: Coordination Expert Engine ---")
    try:
        engine = joblib.load(cfg.STAGE_1_EXPERTS_DIR / sys_cfg.VSM2_ENGINE_NAME)
    except FileNotFoundError:
        print("Error: VSM 2 engine not found. Train the engine first.")
        return
    loadings = engine.components_
    feature_names = sys_cfg.VSM2_INPUT_FEATURES
    df_loadings = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])], index=feature_names)
    print("\nComponent Loadings:\n", df_loadings)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_loadings, annot=True, cmap='viridis')
    plt.title('VSM 2 Engine: Feature Loadings')
    output_path = cfg.VERIFICATION_DIR / "vsm2_component_loadings.png"
    plt.savefig(output_path)
    print(f"Saved component loadings heatmap to '{output_path}'")
    plt.close()
    print("--- VSM 2 Engine analysis complete ---")

if __name__ == "__main__":
    analyze_coordination_engine()
