import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vsm_cybernetic_model.configs import main_config as cfg

def validate_rpp_baseline():
    print("--- Validating VSM System 1: RPP Baseline ---")
    try:
        df_rpp = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_RPP)
    except FileNotFoundError:
        print(f"Error: RPP file not found. Run RPP simulation first.")
        return
    df_rpp['real_yield'] = df_rpp['RPP_mean_yield'] - 10
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_rpp['RPP_mean_yield'], label='RPP Mean Yield', fill=True)
    sns.kdeplot(df_rpp['real_yield'], label='Real Yield (Dummy)', fill=True)
    plt.title('Plausibility Check: RPP vs. Real Yield')
    output_path = cfg.VERIFICATION_DIR / "vsm1_rpp_plausibility_check.png"
    plt.savefig(output_path)
    print(f"Saved RPP plausibility check to '{output_path}'")
    plt.close()
    print("--- VSM 1 RPP Baseline validation complete ---")

if __name__ == "__main__":
    validate_rpp_baseline()
