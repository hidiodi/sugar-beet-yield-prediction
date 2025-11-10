# File: src/models/run_shap_analysis.py
# DESCRIPTION:
# This script loads the three final quantile models (lower, median, upper)
# and performs a SHAP (SHapley Additive exPlanations) analysis on each.
# Refactored to use central configuration from src.config

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

# Use the config dictionaries from the central config file
XGB_CONFIG = config.XGBOOST_TRAINING_CONFIG
SHAP_CONFIG = config.SHAP_ANALYSIS_CONFIG

def load_and_prep_data():
    """Loads and prepares the training data exactly as in the training script."""
    print(f"Loading data from {XGB_CONFIG['DATA_PATH']}...")
    df = pd.read_csv(XGB_CONFIG['DATA_PATH'])

    df.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df['forecast_residual'] = df['kreisYield'] - df['stage1_forecast']
    df.dropna(subset=['stage1_forecast', 'forecast_residual'], inplace=True)

    X_train = df[XGB_CONFIG['FEATURE_COLS']]
    print(f"Data loaded. Found {len(X_train)} samples (NaNs in features are kept).")
    return X_train


def run_analysis():
    """Main function to run the SHAP analysis loop."""
    print("--- Starting SHAP Analysis for Final Quantile Models ---")

    shap_output_dir = SHAP_CONFIG['SHAP_OUTPUT_DIR']
    os.makedirs(shap_output_dir, exist_ok=True)
    print(f"SHAP plots will be saved to: {shap_output_dir}")

    X_train = load_and_prep_data()

    shap_sample_size = SHAP_CONFIG['SHAP_SAMPLE_SIZE']
    if len(X_train) > shap_sample_size:
        print(f"Dataset is large ({len(X_train)} samples). Subsampling to {shap_sample_size} for SHAP analysis.")
        X_train_shap = shap.sample(X_train, shap_sample_size, random_state=42)
    else:
        X_train_shap = X_train

    for name, alpha in XGB_CONFIG['QUANTILES'].items():
        print(f"\n--- Analyzing {name.upper()} Model (Quantile: {alpha}) ---")

        model_path = XGB_CONFIG[f'{name.upper()}_MODEL_PATH']
        if not os.path.exists(model_path):
            print(f"⚠️  Model file not found, skipping: {model_path}")
            continue

        print(f"Loading model: {model_path}")
        model = joblib.load(model_path)

        try:
            booster = model.get_booster()
            config_str = booster.save_config()
            config_json = json.loads(config_str)
            base_score_val = config_json.get('learner', {}).get('learner_model_param', {}).get('base_score')
            if base_score_val and isinstance(base_score_val, str):
                parsed_score = float(base_score_val.strip('[]'))
                print(f"Warning: Model's internal 'base_score' was a string '{base_score_val}'.")
                print(f"Patching booster config with float: {parsed_score}")
                config_json['learner']['learner_model_param']['base_score'] = parsed_score
                booster.load_config(json.dumps(config_json))
                model.base_score = parsed_score
        except Exception as e:
            print(f"Error: Could not parse or patch model.base_score. SHAP analysis might fail. Error: {e}")
            if isinstance(model.base_score, str):
                try:
                    model.base_score = float(model.base_score.strip('[]'))
                except:
                    pass

        print("Initializing SHAP Explainer...")
        explainer = shap.Explainer(model.predict, X_train_shap)
        print(f"Calculating SHAP values for {len(X_train_shap)} samples... (This may take a moment)")
        shap_values = explainer(X_train_shap)
        print("SHAP values calculated.")

        print("Generating Global Feature Importance (bar) plot...")
        try:
            plt.figure(figsize=(10, 16))
            plt.title(f'SHAP Global Feature Importance - {name.upper()} Model', fontsize=16, pad=20)
            shap.summary_plot(shap_values, X_train_shap, plot_type='bar', show=False, max_display=30)
            save_path = os.path.join(shap_output_dir, f'shap_global_importance_bar_{name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved bar plot to {save_path}")
        except Exception as e:
            print(f"❌ Failed to generate bar plot for {name}: {e}")
            plt.close()

        print("Generating Feature Distribution (beeswarm) plot...")
        try:
            plt.figure(figsize=(10, 16))
            plt.title(f'SHAP Feature Distribution - {name.upper()} Model', fontsize=16, pad=20)
            shap.summary_plot(shap_values, X_train_shap, plot_type='dot', show=False, max_display=30)
            save_path = os.path.join(shap_output_dir, f'shap_distribution_beeswarm_{name}.png')
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
