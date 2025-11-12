import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from vsm_cybernetic_model.configs import main_config as cfg
from vsm_cybernetic_model.configs import system_1_biophysical as sys_cfg

def analyze_biophysical_engine():
    print("--- Analyzing VSM System 1: Biophysical Expert Engine ---")
    try:
        engine = joblib.load(cfg.STAGE_1_EXPERTS_DIR / sys_cfg.VSM1_ENGINE_NAME)
    except FileNotFoundError:
        print("Error: VSM 1 engine not found. Train the engine first.")
        return
    loadings = engine.components_
    feature_names = sys_cfg.VSM1_INPUT_FEATURES
    df_loadings = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])], index=feature_names)
    print("\nComponent Loadings:\n", df_loadings)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_loadings, annot=True, cmap='coolwarm')
    plt.title('VSM 1 Engine: Feature Loadings on Principal Components')
    output_path = cfg.VERIFICATION_DIR / "vsm1_component_loadings.png"
    plt.savefig(output_path)
    print(f"Saved component loadings heatmap to '{output_path}'")
    plt.close()
    print("--- VSM 1 Engine analysis complete ---")

if __name__ == "__main__":
    analyze_biophysical_engine()
