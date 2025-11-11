# 01_run_post_hoc_shap_analysis.py
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from vsm_cybernetic_model.configs import main_config as cfg

def run_shap_analysis():
    """
    Performs a post-hoc SHAP analysis on the final trained regulator model.

    This script loads the final feature matrix and the trained XGBoost model,
    then calculates and plots the SHAP summary plot. This plot is the ultimate
    validation of the entire VSM-CPS system, as it reveals which high-level
    VSM indices (our engineered features) are the most important drivers of the
    final prediction.
    """
    print("--- Running Post-Hoc SHAP Analysis on Final Regulator Model ---")

    # 1. Load Model and Data
    try:
        model = joblib.load(cfg.REGULATOR_MODEL_PATH)
        print(f"Loaded regulator model from '{cfg.REGULATOR_MODEL_PATH}'")
    except FileNotFoundError:
        print(f"Error: Regulator model not found at '{cfg.REGULATOR_MODEL_PATH}'.")
        print("Please run the model training pipeline first.")
        return

    try:
        df_features = pd.read_csv(cfg.FINAL_FEATURES_PATH).dropna()
        print(f"Loaded final features from '{cfg.FINAL_FEATURES_PATH}'")
    except FileNotFoundError:
        print(f"Error: Final features file not found at '{cfg.FINAL_FEATURES_PATH}'.")
        print("Please run the transformation pipeline first.")
        return

    feature_cols = [col for col in df_features.columns if 'PC' in col]
    X = df_features[feature_cols]

    if X.empty:
        print("Error: No data available for SHAP analysis.")
        return

    # 2. Calculate SHAP Values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 3. Generate and Save SHAP Summary Plot
    print("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance for Final Regulator Model")

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    output_path = cfg.VERIFICATION_DIR / "final_model_shap_summary.png"
    plt.savefig(output_path)
    print(f"Saved SHAP summary plot to '{output_path}'")
    plt.close()

    print("--- Post-hoc SHAP analysis complete ---")


if __name__ == "__main__":
    run_shap_analysis()
