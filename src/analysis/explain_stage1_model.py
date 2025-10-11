# File: src/models/analyze_final_model.py
# Description: MODIFIED to explain the Stage 1 (Pre-Season) forecast model using SHAP and PDPs.

import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
import numpy as np
import os
import warnings
from sklearn.inspection import PartialDependenceDisplay

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- MODIFIED: Define the final feature list for the Stage 1 Pre-Season Model ---
FEATURE_COLS = [
    'district_no',
    'avg_elevation',
    'avg_soil_pawc',
    'winter_temp_anomaly',
    'winter_precip_anomaly',
    'national_avg_yield_lag1',
    'producer_price_index_lag1'
]
TARGET_COL = 'kreisYield'

# --- MODIFIED: Adjust paths to the new pre-season model and data ---
MODEL_PATH = os.path.join('src/models', 'stage1_preseason_xgb_model.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
REPORT_DIR = os.path.join('reports', 'stage1_preseason_explanation')


def load_model_and_data():
    """
    Loads the trained Stage 1 XGBoost model and the full dataset used for training.
    """
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(DATA_PATH)
        X_data = df[FEATURE_COLS]
        print(f"✅ Successfully loaded {len(X_data)} samples from {DATA_PATH}.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Cannot proceed.")
        return None, None

    # --- 2. Load Model ---
    try:
        xgb_model = joblib.load(MODEL_PATH)
        print(f"✅ Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
        return None, None

    return xgb_model, X_data


def plot_shap_summary(xgb_model: XGBRegressor, X_data: pd.DataFrame):
    """Generates the global SHAP summary plot."""
    print("\nGenerating SHAP Summary Plot (Global Feature Importance)...")

    explainer = shap.TreeExplainer(xgb_model)
    # Use a representative sample for faster SHAP calculation
    X_sample = X_data.sample(min(500, len(X_data)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary: Stage 1 Pre-Season Model")
    plt.xlabel("SHAP Value (Impact on Yield Prediction)")

    save_path = os.path.join(REPORT_DIR, 'shap_global_summary.png')
    os.makedirs(REPORT_DIR, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ SHAP Summary Plot saved to {save_path}")
    plt.close()


def plot_feature_dependence(xgb_model: XGBRegressor, X_data: pd.DataFrame, features: list):
    """Generates Partial Dependence Plots (PDPs) for the functional relationship."""
    print("\nGenerating Partial Dependence Plots (Feature Relationships)...")
    try:
        n_features = len(features)
        n_cols = min(n_features, 4)
        n_rows = int(np.ceil(n_features / n_cols))
        fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(5 * n_cols, 4 * n_rows))
        ax = ax.flatten() if n_features > 1 else [ax]

        PartialDependenceDisplay.from_estimator(
            xgb_model, X_data, features, kind='average', ax=ax
        )
        fig.suptitle("Partial Dependence: How Features Influence Yield Prediction", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(REPORT_DIR, 'pdp_feature_relationships.png')
        plt.savefig(save_path)
        print(f"✅ PDP Plots saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating PDPs: {e}")


def plot_shap_interaction(xgb_model: XGBRegressor, X_data: pd.DataFrame, main_feature: str, interaction_feature: str):
    """Generates a SHAP dependence plot to show interaction effects."""
    print(f"\nGenerating SHAP Interaction Plot: {main_feature} vs. {interaction_feature}...")

    explainer = shap.TreeExplainer(xgb_model)
    # Use the full dataset for a clear interaction plot
    shap_values = explainer.shap_values(X_data)

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        main_feature, shap_values, X_data,
        interaction_index=interaction_feature, show=False
    )
    plt.title(f"SHAP Interaction: {main_feature} vs. {interaction_feature}")

    save_path = os.path.join(REPORT_DIR, f'shap_interaction_{main_feature}_{interaction_feature}.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ SHAP Interaction Plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    xgb_model, X_data = load_model_and_data()

    if xgb_model is not None and X_data is not None:
        print(f"\n--- Starting Explanation Analysis for Stage 1 Pre-Season Model ---")

        # 1. Global Insight: What features matter most overall?
        plot_shap_summary(xgb_model, X_data)

        # 2. Feature Relationships: How do the most important features drive predictions?
        # MODIFIED: Plotting the most important features from our new model
        top_features_for_pdp = [
            'national_avg_yield_lag1',
            'producer_price_index_lag1',
            'winter_temp_anomaly',
            'avg_soil_pawc'
        ]
        plot_feature_dependence(xgb_model, X_data, top_features_for_pdp)

        # 3. Feature Interactions: Do features work together?
        # MODIFIED: Highlighting a plausible interaction between winter weather and soil type
        plot_shap_interaction(xgb_model, X_data,
                              main_feature="winter_temp_anomaly",
                              interaction_feature="avg_soil_pawc")

        print("\n✅ Analysis Complete. Outputs saved to directory: " + REPORT_DIR)
    else:
        print("\nAnalysis failed due to missing model or data.")