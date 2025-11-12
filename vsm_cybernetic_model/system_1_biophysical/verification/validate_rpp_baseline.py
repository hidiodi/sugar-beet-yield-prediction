import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vsm_cybernetic_model.configs import main_config as cfg

def validate_rpp_baseline():
    """
    Validates the plausibility of the biophysical features by comparing
    the WOFOST-derived potential yield against the actual reported yield.
    """
    print("--- Validating VSM System 1: Biophysical Baseline ---")

    # CORRECTED: Load from the single unified foundational feature set
    try:
        df = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN)
        print(f"Loaded unified foundational features from '{cfg.FOUNDATIONAL_FEATURES_HUMAN}'")
    except FileNotFoundError:
        print(f"Error: Foundational features file not found at '{cfg.FOUNDATIONAL_FEATURES_HUMAN}'.")
        print("Please run the data preparation pipeline first.")
        return

    # Check if required columns exist
    required_cols = ['RPP_mean_yield', 'kreisYield']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The unified feature set is missing one of the required columns: {required_cols}")
        print("Cannot perform VSM 1 baseline validation.")
        return

    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['RPP_mean_yield'], label='WOFOST Potential Yield (RPP)', fill=True)
    sns.kdeplot(df['kreisYield'], label='Observed Yield', fill=True)
    plt.title('Plausibility Check: VSM 1 - Potential vs. Observed Yield')
    plt.xlabel('Yield (dt/ha)')
    plt.legend()

    output_path = cfg.VERIFICATION_DIR / "vsm1_yield_plausibility_check.png"
    plt.savefig(output_path)
    print(f"Saved VSM 1 yield plausibility check to '{output_path}'")
    plt.close()

    print("--- VSM 1 Biophysical Baseline validation complete ---")

if __name__ == "__main__":
    validate_rpp_baseline()
