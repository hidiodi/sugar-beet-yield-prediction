import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vsm_cybernetic_model.configs import main_config as cfg

def validate_biophysical_features():
    print("--- Validating VSM System 1: Foundational Biophysical Features ---")
    try:
        df = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN)
    except FileNotFoundError:
        print("Error: Foundational features file not found.")
        return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='summer_temp_anomaly_forecast', y='summer_precip_anomaly_forecast')
    plt.title('VSM 1: Summer Temp vs. Precip Forecast Anomaly')
    output_path = cfg.VERIFICATION_DIR / "vsm1_temp_vs_precip.png"
    plt.savefig(output_path)
    print(f"Saved temp vs. precip plot to '{output_path}'")
    plt.close()
    print("--- VSM 1 Feature validation complete ---")

if __name__ == "__main__":
    validate_biophysical_features()
