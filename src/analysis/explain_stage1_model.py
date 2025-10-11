import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split  # <--- ADDED IMPORT
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os
import warnings
from sklearn.inspection import \
    PartialDependenceDisplay  # Moving inside the function is better, but put at top for completeness

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the final feature list for consistency
FEATURE_COLS = [
    'district_no',
    'kreisField_ha',
    'producer_price_index',
    'precip_total_peak_growth',
    'heat_stress_days_peak_growth',
    'avg_elevation',
    'winter_temp_anomaly',
    'winter_precip_anomaly',
    'yield_density'
]
TARGET_COL = 'kreisYield'
# --- NOTE: Adjust these paths to your project structure ---
MODEL_PATH = os.path.join('src/models', 'stage1_xgb_model.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'final_imputed_dataset.csv')
REPORT_DIR = os.path.join('reports', 'stage1_explanation')


def load_model_and_data():
    """
    Loads the trained XGBoost model and the test dataset.
    Uses random_state=42 to ensure the train/test split matches the training script.
    """

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(DATA_PATH)
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]
        # Split data using the same parameters as training (80/20, random_state=42)
        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✅ Successfully loaded data and split {len(X_test)} samples for testing.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Cannot proceed with explanation.")
        return None, None

    # --- 2. Load Model ---
    try:
        # Load the model saved by the training script
        xgb_model = joblib.load(MODEL_PATH)
        print(f"✅ Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Attempting to train a quick dummy model.")
        # Fallback: If model is not saved, train a quick one for execution.
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)

    return xgb_model, X_test


def plot_shap_summary(xgb_model: XGBRegressor, X_test: pd.DataFrame):
    """Generates the global SHAP summary plot."""
    print("Generating SHAP Summary Plot (Global Feature Importance)...")

    explainer = shap.TreeExplainer(xgb_model)
    # Use a fraction of the test data for faster SHAP calculation if the dataset were larger
    X_test_sample = X_test.head(200)
    shap_values = explainer.shap_values(X_test_sample)

    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("SHAP Global Feature Importance and Effect on KreisYield")

    save_path = os.path.join(REPORT_DIR, 'shap_global_summary.png')
    os.makedirs(REPORT_DIR, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ SHAP Summary Plot saved to {save_path}")
    plt.close()


def plot_feature_dependence(xgb_model: XGBRegressor, X_test: pd.DataFrame, features: list):
    """Generates Partial Dependence Plots (PDPs) for the functional relationship."""

    print("\nGenerating Partial Dependence Plots (Feature Relationships)...")

    try:
        # Use a maximum of 4 plots per row for readability
        n_features = len(features)
        n_cols = min(n_features, 4)
        n_rows = int(np.ceil(n_features / n_cols))

        fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(4 * n_cols, 4 * n_rows))

        # Flatten the axis array if necessary
        if n_rows > 1 or n_cols > 1:
            ax = ax.flatten()
        else:
            # Make sure ax is iterable even for a single plot
            ax = [ax]

        PartialDependenceDisplay.from_estimator(
            xgb_model,
            X_test,
            features,
            kind='average',
            ax=ax,
            feature_names=X_test.columns.tolist()
        )
        fig.suptitle("Partial Dependence Plots (PDP) on KreisYield")
        plt.tight_layout()
        save_path = os.path.join(REPORT_DIR, 'pdp_feature_relationships.png')
        plt.savefig(save_path)
        print(f"✅ PDP Plots saved to {save_path}")
        plt.close()

    except Exception as e:
        print(f"Error generating PDPs: {e}")


def plot_shap_interaction(xgb_model: XGBRegressor, X_test: pd.DataFrame, main_feature: str, interaction_feature: str):
    """Generates a SHAP dependence plot to show interaction effects."""
    print(f"\nGenerating SHAP Interaction Plot: {main_feature} vs. {interaction_feature}...")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # SHAP Dependence Plot with interaction coloring
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        main_feature,
        shap_values,
        X_test,
        interaction_index=interaction_feature,
        show=False,
        title=f"Yield Dependency on {main_feature} (Colored by {interaction_feature})"
    )

    save_path = os.path.join(REPORT_DIR, f'shap_interaction_{main_feature}_{interaction_feature}.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ SHAP Interaction Plot saved to {save_path}")
    plt.close()


def explain_local_prediction(xgb_model: XGBRegressor, X_test: pd.DataFrame, index: int = 0):
    """Prints the local SHAP Force Plot for a single data point."""
    print(f"\nGenerating Local Explanation (Force Plot) for Observation Index {index}...")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    print(f"Prediction for Observation {index}: {xgb_model.predict(X_test.iloc[index].to_frame().T)[0]:.2f} dt/ha")

    # The force plot requires a JS environment (notebook/HTML), so we return the object
    # The user must execute this in a suitable environment to see the interactive chart.
    print("Local SHAP Force Plot object created. Run in a Jupyter/Colab notebook to view the interactive chart.")
    return shap.force_plot(explainer.expected_value,
                           shap_values[index, :],
                           X_test.iloc[index, :])


if __name__ == "__main__":

    # 1. Load Model and Data
    xgb_model, X_test = load_model_and_data()

    if xgb_model is not None and X_test is not None:
        print(f"\nStarting explanation analysis on {len(X_test)} test samples.")

        # --- 1. Global Model Insight: SHAP Summary ---
        plot_shap_summary(xgb_model, X_test)

        # --- 2. Feature Relationships: PDPs ---
        # Plotting the top non-density features
        top_agronomic_features = [
            'precip_total_peak_growth',
            'heat_stress_days_peak_growth',
            'district_no',
            'kreisField_ha'
        ]
        plot_feature_dependence(xgb_model, X_test, top_agronomic_features)

        # --- 3. Feature Interactions: SHAP Dependence Plot ---
        # Highlighting the critical weather interaction
        plot_shap_interaction(xgb_model, X_test,
                              main_feature="precip_total_peak_growth",
                              interaction_feature="heat_stress_days_peak_growth")

        # --- 4. Local Model Insight: SHAP Force Plot ---
        # We will explain the first observation.
        local_explanation = explain_local_prediction(xgb_model, X_test, index=0)

    else:
        print("Analysis failed due to missing model or data. Please check file paths.")