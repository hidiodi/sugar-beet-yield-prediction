import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vsm_cybernetic_model.configs import main_config as cfg

def validate_policy_features():
    print("--- Validating VSM System 5: Foundational Policy Features ---")
    try:
        df = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN)
    except FileNotFoundError:
        print("Error: Foundational features file not found.")
        return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='percent_UAA_in_NVZ', y='CAP_Euros_per_Hectare_UAA')
    plt.title('NVZ Percentage vs. CAP Subsidy per Hectare')
    output_path = cfg.VERIFICATION_DIR / "vsm5_nvz_vs_cap.png"
    plt.savefig(output_path)
    print(f"Saved NVZ vs. CAP plot to '{output_path}'")
    plt.close()
    print("--- VSM 5 Feature validation complete ---")

if __name__ == "__main__":
    validate_policy_features()
