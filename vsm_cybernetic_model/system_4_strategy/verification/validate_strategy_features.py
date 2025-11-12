import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vsm_cybernetic_model.configs import main_config as cfg

def validate_strategy_features():
    print("--- Validating VSM System 4: Foundational Strategy Features ---")
    try:
        df = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN)
    except FileNotFoundError:
        print("Error: Foundational features file not found.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(df['national_avg_yield_lag1'], bins=30, kde=True)
    plt.title('VSM 4: Distribution of Lagged National Average Yield')
    output_path = cfg.VERIFICATION_DIR / "vsm4_nat_yield_distribution.png"
    plt.savefig(output_path)
    print(f"Saved national yield distribution plot to '{output_path}'")
    plt.close()
    print("--- VSM 4 Feature validation complete ---")

if __name__ == "__main__":
    validate_strategy_features()
