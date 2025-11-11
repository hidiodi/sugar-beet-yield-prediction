import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vsm_cybernetic_model.configs import main_config as cfg

def validate_coordination_features():
    print("--- Validating VSM System 2: Foundational Coordination Features ---")
    try:
        df = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN)
    except FileNotFoundError:
        print("Error: Foundational features file not found.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(df['avg_sowing_date_doy'], bins=30, kde=True)
    plt.title('Distribution of Average Sowing Date')
    output_path = cfg.VERIFICATION_DIR / "vsm2_sowing_date_distribution.png"
    plt.savefig(output_path)
    print(f"Saved sowing date distribution to '{output_path}'")
    plt.close()
    print("--- VSM 2 Feature validation complete ---")

if __name__ == "__main__":
    validate_coordination_features()
