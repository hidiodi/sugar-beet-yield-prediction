# File: src/03_analysis/automated_error_analysis.py
# Description: Performs an automated, data-driven analysis to discover the specific
# conditions and feature interactions that lead to the largest model errors.
# Version 2: Improved diagnostic model to find more general scenarios.

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ks_2samp

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load all necessary configuration from the central config object
MODEL_CONFIG = config.MODEL_COMPARISON_CONFIG
XGB_CONFIG = config.XGBOOST_TRAINING_CONFIG


# --- Main Analysis Function ---

def automated_error_analysis():
    """
    Main orchestrator for the automated error analysis pipeline.
    """
    logging.info("--- Starting Automated Model Error Analysis ---")

    # --- 1. Load and Prepare Data ---
    try:
        predictions_file = MODEL_CONFIG['HYBRID_XGB_PREDICTIONS_FILE']
        df_preds = pd.read_csv(predictions_file)
        logging.info(f"✓ Loaded predictions ({len(df_preds)} rows).")

        features_file = XGB_CONFIG['DATA_PATH']
        df_features = pd.read_csv(features_file)
        logging.info(f"✓ Loaded input features ({len(df_features)} rows).")

    except Exception as e:
        logging.error(f"❌ FATAL: Could not load data. Details: {e}")
        sys.exit(1)

    merge_keys = ['year', 'district_no']
    df_analysis = pd.merge(df_preds, df_features, on=merge_keys, how='left')
    df_analysis.drop_duplicates(subset=merge_keys, keep='first', inplace=True)
    df_analysis.reset_index(drop=True, inplace=True)
    logging.info(f"✓ Data merged and cleaned ({len(df_analysis)} unique rows).")

    error_threshold = df_analysis['abs_error'].quantile(0.90)
    df_analysis['is_high_error'] = (df_analysis['abs_error'] >= error_threshold).astype(int)

    feature_cols = [col for col in XGB_CONFIG['FEATURE_COLS'] if col in df_analysis.columns]
    X = df_analysis[feature_cols].copy()
    y = df_analysis['is_high_error']

    # FIX for FutureWarning: Avoid inplace on a copy
    for col in X.select_dtypes(include=np.number).columns:
        X[col] = X[col].fillna(X[col].median())

    # --- 2. Statistical Feature Analysis (Kolmogorov-Smirnov Test) ---
    logging.info("Performing statistical tests to find discriminating features...")
    df_good_preds = df_analysis[df_analysis['is_high_error'] == 0]
    df_bad_preds = df_analysis[df_analysis['is_high_error'] == 1]

    ks_results = [{'feature': col, 'ks_statistic': ks_2samp(df_good_preds[col].dropna(), df_bad_preds[col].dropna())[0],
                   'p_value': ks_2samp(df_good_preds[col].dropna(), df_bad_preds[col].dropna())[1]} for col in
                  X.select_dtypes(include=np.number).columns]
    df_ks = pd.DataFrame(ks_results).sort_values('ks_statistic', ascending=False)

    print("\n" + "=" * 80)
    print("      AUTOMATED ANALYSIS REPORT: MODEL WEAKNESSES")
    print("=" * 80)
    print("\n--- SECTION 1: TOP 5 MOST DISCRIMINATING INDIVIDUAL FEATURES ---")
    print(df_ks[df_ks['p_value'] < 0.05].head(5).to_string(index=False, float_format="%.4f"))

    # --- 3. Train a "Failure Diagnosis" Model ---
    logging.info("Training a diagnostic model to find high-error scenarios...")
    # FIX: Use a simpler tree to find more general and robust rules
    error_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
    error_model.fit(X, y)

    # --- 4. Extract and Report High-Error Scenarios ---
    feature_importances = pd.DataFrame(
        {'feature': feature_cols, 'importance': error_model.feature_importances_}).sort_values('importance',
                                                                                               ascending=False)
    print("\n\n--- SECTION 2: TOP FEATURES DRIVING ERROR INTERACTIONS ---")
    print(feature_importances[feature_importances['importance'] > 0].to_string(index=False, float_format="%.4f"))

    print("\n\n--- SECTION 3: AUTOMATICALLY DISCOVERED HIGH-ERROR SCENARIOS ---")

    children_left, children_right, feature, threshold, value = error_model.tree_.children_left, error_model.tree_.children_right, error_model.tree_.feature, error_model.tree_.threshold, error_model.tree_.value

    def find_high_error_rules(node_index=0, path=[]):
        if children_left[node_index] == children_right[node_index]:
            error_prob = value[node_index][0][1] / value[node_index][0].sum()
            num_samples = int(value[node_index][0].sum())
            if error_prob > 0.20 and num_samples > 40:  # Only show rules for significant groups
                print(
                    f"SCENARIO FOUND: High probability of large error ({error_prob:.1%}) in {num_samples} cases where:")
                for rule in path: print(f"  - {rule}")
                print("-" * 30)
            return

        feature_name = feature_cols[feature[node_index]]
        threshold_val = threshold[node_index]
        find_high_error_rules(children_left[node_index], path + [f"{feature_name} <= {threshold_val:.2f}"])
        find_high_error_rules(children_right[node_index], path + [f"{feature_name} > {threshold_val:.2f}"])

    find_high_error_rules()
    print("\n" + "=" * 80)
    logging.info("--- Automated analysis finished successfully. ---")


if __name__ == '__main__':
    automated_error_analysis()