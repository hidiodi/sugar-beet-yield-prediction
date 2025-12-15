# FILE: src/03_analysis/shap_analysis_standalone.py
# DESCRIPTION:
# Performs SHAP (SHapley Additive exPlanations) analysis on the STANDALONE XGBoost model.
# Aligned with 'train_standalone_xgb_model.py' (Predicts Residual from Rolling Trend).

import pandas as pd
import os
import joblib
import warnings
import numpy as np
import shap
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src import config

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Use the Standalone config
XGB_CONFIG = config.STANDALONE_XGB_CONFIG
SHAP_CONFIG = config.SHAP_ANALYSIS_CONFIG

# Update output directory specifically for Standalone analysis
OUTPUT_DIR = SHAP_CONFIG['SHAP_OUTPUT_DIR'] / "standalone_xgb"


def load_and_prep_data(model_feature_names):
    """
    Loads and prepares the training data, ensuring the final DataFrame
    has columns in the exact order specified by model_feature_names.
    Matches logic in train_standalone_xgb_model.py.
    """
    print(f"Loading data from {XGB_CONFIG['DATA_PATH']}...")
    df = pd.read_csv(XGB_CONFIG['DATA_PATH'])

    # --- REPLICATE TARGET ENGINEERING FOR FILTERING ---
    # We need to calculate the baseline/target to drop NaNs exactly like the training script.
    baseline_col = 'yield_rolling_trend'
    target_col = 'trend_residual'

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # 5-Year Rolling Average (Lagged to prevent leak)
    df[baseline_col] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=3).mean().shift(1)
    )

    # Calculate Target (Residual)
    df[target_col] = df['kreisYield'] - df[baseline_col]

    # Drop rows where target or baseline are missing (Matches training logic)
    df.dropna(subset=[target_col, baseline_col], inplace=True)

    # --- FINAL STEP: Select features using the model's exact list ---
    try:
        # This guarantees the columns and their order match the model
        # The Standalone training script relies on the CSV having these columns pre-calculated.
        missing_cols = [col for col in model_feature_names if col not in df.columns]
        if missing_cols:
            raise KeyError(f"The following features expected by the model are missing from the CSV: {missing_cols}")

        X_train = df[model_feature_names]

        # Ensure no NaNs in features (Standard XGBoost training prep)
        X_train.dropna(inplace=True)

    except KeyError as e:
        print(f"--- FATAL ERROR ---")
        print(f"Column mismatch: {e}")
        raise e

    print(f"Data loaded and prepared. Found {len(X_train)} samples.")
    return X_train


def run_analysis():
    """Main function to run the SHAP analysis loop."""
    print("--- Starting SHAP Analysis for Standalone XGBoost Models ---")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"SHAP plots will be saved to: {OUTPUT_DIR}")

    for name, alpha in XGB_CONFIG['QUANTILES'].items():
        print(f"\n--- Analyzing {name.upper()} Model (Quantile: {alpha}) ---")

        model_path = XGB_CONFIG[f'{name.upper()}_MODEL_PATH']
        if not os.path.exists(model_path):
            print(f"⚠️  Model file not found, skipping: {model_path}")
            continue

        print(f"Loading model: {model_path}")
        model = joblib.load(model_path)

        # --- Base Score Patch & Feature Name Extraction ---
        try:
            booster = model.get_booster()

            # --- CRITICAL: Get feature names FROM THE MODEL ---
            model_feature_names = booster.feature_names

            # Patch base_score if needed (fix for JSON serialization issues in older XGB versions)
            config_str = booster.save_config()
            config_json = json.loads(config_str)
            base_score_val = config_json.get('learner', {}).get('learner_model_param', {}).get('base_score')
            if base_score_val and isinstance(base_score_val, str):
                parsed_score = float(base_score_val.strip('[]'))
                config_json['learner']['learner_model_param']['base_score'] = parsed_score
                booster.load_config(json.dumps(config_json))
                model.base_score = parsed_score
        except Exception as e:
            print(f"Warning: Could not parse/patch model metadata. Analysis might fail. Error: {e}")
            # Fallback for simple list mismatch
            if hasattr(model, 'feature_names_in_'):
                model_feature_names = list(model.feature_names_in_)
        # --- End of Patch ---

        # --- Load data using the Model's feature list ---
        X_train = load_and_prep_data(model_feature_names)

        # Subsample if dataset is too large for SHAP
        shap_sample_size = SHAP_CONFIG['SHAP_SAMPLE_SIZE']
        if len(X_train) > shap_sample_size:
            print(f"Dataset is large ({len(X_train)} samples). Subsampling to {shap_sample_size} for SHAP analysis.")
            X_train_shap = shap.sample(X_train, shap_sample_size, random_state=42)
        else:
            X_train_shap = X_train

        print("Initializing SHAP Explainer...")
        # TreeExplainer is optimized for XGBoost
        explainer = shap.Explainer(model.predict, X_train_shap)

        print(f"Calculating SHAP values for {len(X_train_shap)} samples...")
        shap_values = explainer(X_train_shap)
        print("SHAP values calculated.")

        # 1. Global Feature Importance (Bar Plot)
        print("Generating Global Feature Importance (bar) plot...")
        try:
            plt.figure(figsize=(10, 16))
            plt.title(f'SHAP Global Feature Importance - {name.upper()} (Standalone)', fontsize=16, pad=20)
            shap.summary_plot(shap_values, X_train_shap, plot_type='bar', show=False, max_display=30)
            save_path = os.path.join(OUTPUT_DIR, f'shap_global_importance_bar_{name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved bar plot to {save_path}")
        except Exception as e:
            print(f"❌ Failed to generate bar plot for {name}: {e}")
            plt.close()

        # 2. Feature Distribution (Beeswarm Plot)
        print("Generating Feature Distribution (beeswarm) plot...")
        try:
            plt.figure(figsize=(10, 16))
            plt.title(f'SHAP Feature Distribution - {name.upper()} (Standalone)', fontsize=16, pad=20)
            shap.summary_plot(shap_values, X_train_shap, plot_type='dot', show=False, max_display=30)
            save_path = os.path.join(OUTPUT_DIR, f'shap_distribution_beeswarm_{name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved beeswarm plot to {save_path}")
        except Exception as e:
            print(f"❌ Failed to generate beeswarm plot for {name}: {e}")
            plt.close()

        print(f"--- Completed analysis for {name.upper()} Model ---")

    print("\n--- SHAP Analysis Complete. All plots saved. ---")


if __name__ == "__main__":
    run_analysis()