# File: src/models/analyze_final_model.py
# Description: FINALIZED & IMPROVED analysis script. Fixes the PDP crash and
#              correctly analyzes the final, complex model.

import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
import numpy as np
import os
import warnings
from sklearn.inspection import PartialDependenceDisplay

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion_final.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
REPORT_DIR = os.path.join('reports', 'final_model_explanation')


def load_model_and_split_data(feature_list):
    """
    Loads the model and data, and recreates the exact train/validation/test splits.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        missing_cols = [col for col in feature_list if col not in df.columns]
        if missing_cols:
            print(f"Error: The data file is missing required feature columns: {missing_cols}")
            return [None] * 5
        print(f"✅ Successfully loaded {len(df)} samples from {DATA_PATH}.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}.")
        return [None] * 5

    try:
        xgb_model = joblib.load(MODEL_PATH)
        print(f"✅ Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}.")
        return [None] * 5

    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    ).fillna(method='ffill').fillna(method='bfill')

    validation_start_year = 2009
    test_start_year = 2019

    validation_df = df[(df['year'] >= validation_start_year) & (df['year'] < test_start_year)].copy()
    test_df = df[df['year'] >= test_start_year].copy()

    X_validation = validation_df[feature_list]
    X_test = test_df[feature_list]

    return xgb_model, X_validation, X_test, validation_df, test_df


def plot_shap_summary(xgb_model: XGBRegressor, X_data: pd.DataFrame, data_split_name: str):
    """Generates the global SHAP summary plot."""
    print(f"\nGenerating SHAP Summary Plot on {data_split_name} data...")
    explainer = shap.TreeExplainer(xgb_model)
    X_sample = X_data.sample(min(2000, len(X_data)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_sample, show=False, plot_type='bar')
    plt.title(f"SHAP Global Feature Importance ({data_split_name} Set)")
    plt.xlabel("mean(|SHAP value|) (Average impact on model output magnitude)")
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, f'shap_global_summary_{data_split_name}.png')
    os.makedirs(REPORT_DIR, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ SHAP Summary Plot saved to {save_path}")
    plt.close()


def plot_feature_dependence(xgb_model: XGBRegressor, full_df: pd.DataFrame, features: list):
    """Generates PDPs with a re-trended, real-unit y-axis."""
    print("\nGenerating Partial Dependence Plots (Feature Relationships)...")
    try:
        avg_trend = full_df['yield_trend'].mean()

        n_features = len(features)
        n_cols = min(n_features, 3)
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(6 * n_cols, 5 * n_rows), sharey=True)
        axes = axes.flatten() if n_features > 1 else [axes]

        feature_data = full_df[xgb_model.feature_names_in_]

        display = PartialDependenceDisplay.from_estimator(
            xgb_model, feature_data, features,
            kind='average', ax=axes
        )

        # ============================ THE FIX ============================
        # ### CRITICAL FIX: Manually re-trend the y-axis labels AFTER plotting ###
        # This works around the API limitation of the `response_method` parameter.
        for ax in display.axes_.flatten():
            detrended_ticks = ax.get_yticks()
            retrended_labels = [f'{tick + avg_trend:.1f}' for tick in detrended_ticks]
            ax.set_yticks(detrended_ticks)  # Keep original positions
            ax.set_yticklabels(retrended_labels)  # Apply new labels

        display.figure_.get_axes()[0].set_ylabel("Predicted Yield (dt/ha)")
        # ===============================================================

        fig.suptitle("Partial Dependence: How Features Influence Yield Prediction", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(REPORT_DIR, 'pdp_feature_relationships.png')
        plt.savefig(save_path)
        print(f"✅ PDP Plots saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating PDPs: {e}")


if __name__ == "__main__":
    try:
        temp_model = joblib.load(MODEL_PATH)
        FEATURE_COLS = temp_model.feature_names_in_
        print("✅ Automatically loaded feature list from the trained model.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model to get feature list. Error: {e}")
        exit()

    xgb_model, X_validation, X_test, validation_df, test_df = load_model_and_split_data(FEATURE_COLS)

    if xgb_model is not None and test_df is not None:
        print(f"\n--- Starting Explanation Analysis for Final Model ---")

        plot_shap_summary(xgb_model, X_test, "test_set")

        importance_df = pd.DataFrame({
            'feature': xgb_model.feature_names_in_,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        top_features_for_pdp = importance_df['feature'].head(6).tolist()
        print(f"\nTop 6 most important features for PDP analysis: {top_features_for_pdp}")

        plot_feature_dependence(xgb_model, validation_df, top_features_for_pdp)

        print("\n✅ Analysis Complete. Outputs saved to directory: " + REPORT_DIR)
    else:
        print("\nAnalysis failed due to missing model or data.")