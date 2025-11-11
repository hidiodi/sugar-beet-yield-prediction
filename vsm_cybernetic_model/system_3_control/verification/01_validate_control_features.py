import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vsm_cybernetic_model.configs import main_config as cfg

def validate_control_features():
    print("--- Validating VSM System 3: Foundational Control Features ---")
    try:
        df = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN)
    except FileNotFoundError:
        print("Error: Foundational features file not found.")
        return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='avg_land_price_eur_ha', y='cost_pressure_index')
    plt.title('Land Price vs. Cost Pressure Index')
    output_path = cfg.VERIFICATION_DIR / "vsm3_land_price_vs_cost_pressure.png"
    plt.savefig(output_path)
    print(f"Saved land price plot to '{output_path}'")
    plt.close()
    print("--- VSM 3 Feature validation complete ---")

if __name__ == "__main__":
    validate_control_features()
