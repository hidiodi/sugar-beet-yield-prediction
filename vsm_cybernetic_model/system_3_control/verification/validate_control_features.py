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
    sns.scatterplot(data=df, x='producer_price_index_lag1', y='cost_of_inputs_lag1')
    plt.title('VSM 3: Producer Price vs. Cost of Inputs')
    output_path = cfg.VERIFICATION_DIR / "vsm3_price_vs_costs.png"
    plt.savefig(output_path)
    print(f"Saved price vs. costs plot to '{output_path}'")
    plt.close()
    print("--- VSM 3 Feature validation complete ---")

if __name__ == "__main__":
    validate_control_features()
