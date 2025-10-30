# File: src/models/diagnose_hybrid_model.py
# Description: A PROFESSIONAL diagnostic script with automated subgroup discovery to
#              find the model's biggest blind spots.
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_BACKTEST_FILE = 'reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv'
FEATURES_FILE = 'data/05_model_input/stage1_preseason_features.csv'
GEOJSON_PATH = 'data/01_raw/districts_official.geojson'
OUTPUT_DIR = Path('reports/figures/hybrid_model_diagnostics_advanced')


# --- Main Diagnostic Functions ---

def load_and_merge_data(backtest_path, features_path):
    # This function is correct and remains unchanged.
    df_backtest = pd.read_csv(backtest_path)
    df_features = pd.read_csv(features_path)
    df_backtest['error'] = df_backtest['predicted_yield_median'] - df_backtest['kreisYield']
    df_backtest['abs_error'] = df_backtest['error'].abs()
    df_merged = pd.merge(df_backtest, df_features, on=['district_no', 'year'])
    return df_merged


def analyze_error_interactions(df: pd.DataFrame, output_dir: Path):
    # This function provides our deep dive on the "wetness" problem and remains valuable.
    # It is unchanged.
    precip_bins = [-np.inf, -0.1, 0.1, 0.2, np.inf]
    precip_labels = ["Drought (< -0.1)", "Normal (-0.1 to 0.1)", "Wet (0.1 to 0.2)", "Very Wet (> 0.2)"]
    df['precip_category'] = pd.cut(df['summer_precip_anomaly_forecast'], bins=precip_bins, labels=precip_labels)
    clay_bins = [0, 15, 25, np.inf]
    clay_labels = ["Low Clay (<15%)", "Medium Clay (15-25%)", "High Clay (>25%)"]
    df['clay_category'] = pd.cut(df['avg_clay_0_30cm'], bins=clay_bins, labels=clay_labels)
    interaction_mae = df.groupby(['precip_category', 'clay_category'], observed=False)['abs_error'].mean().unstack()
    interaction_count = df.groupby(['precip_category', 'clay_category'], observed=False).size().unstack()
    print("\n--- Deep Dive: MAE by Precipitation vs. Clay Content ---\n")
    print(interaction_mae.to_string(float_format="%.2f"))
    print("\n--- Data Count for Deep Dive ---\n")
    print(interaction_count.to_string(float_format="%d"))


# --- NEW AND MOST POWERFUL FUNCTION: AUTOMATED SUBGROUP DISCOVERY ---

def find_worst_performing_subgroups(df: pd.DataFrame, overall_mae: float):
    """
    Automatically searches for subgroups with disproportionately high error rates.
    """
    logging.info("--- Automated Discovery: Identifying Worst-Performing Subgroups ---")

    # Define categorical features and continuous features to bin for the search
    categorical_features = ['state_encoded']
    continuous_features_to_bin = {
        'summer_temp_anomaly_forecast': [-np.inf, 0, 0.5, 1.0, np.inf],
        'summer_precip_anomaly_forecast': [-np.inf, -0.1, 0.1, np.inf],
        'avg_sand_0_30cm': [0, 20, 50, np.inf],
        'wofost_forecast_yield_fresh_dt': [0, 500, 700, np.inf]
    }

    # Create binned versions of continuous features
    for col, bins in continuous_features_to_bin.items():
        bin_labels = [f'{col}_bin_{i}' for i in range(len(bins) - 1)]
        df[f'{col}_binned'] = pd.cut(df[col], bins=bins, labels=bin_labels, right=False)
        categorical_features.append(f'{col}_binned')

    results = []
    # Analyze single-feature subgroups
    for feature in categorical_features:
        grouped = df.groupby(feature, observed=False)
        for name, group in grouped:
            if len(group) > 50:  # Minimum sample size to be considered a valid subgroup
                mae = group['abs_error'].mean()
                if mae > overall_mae * 1.25:  # Report if MAE is 25% worse than average
                    results.append({'Subgroup': f"{feature} = {name}", 'MAE': mae, 'Count': len(group)})

    # Analyze two-feature interactions (example: state and weather)
    feature1 = 'state_encoded'
    feature2 = 'summer_precip_anomaly_forecast_binned'
    grouped = df.groupby([feature1, feature2], observed=False)
    for name, group in grouped:
        if len(group) > 30:  # Stricter minimum for interactions
            mae = group['abs_error'].mean()
            if mae > overall_mae * 1.4:  # Report if MAE is 40% worse than average
                results.append(
                    {'Subgroup': f"{feature1}={name[0]} & {feature2}={name[1]}", 'MAE': mae, 'Count': len(group)})

    if not results:
        print("\nNo significant underperforming subgroups found based on current criteria.")
        return

    # Sort results by the highest error and print the top N
    top_results = sorted(results, key=lambda x: x['MAE'], reverse=True)

    print("\n" + "=" * 80)
    print("      TOP UNDERPERFORMING SUBGROUPS (Model Blind Spots)")
    print(f"      (Overall MAE for reference: {overall_mae:.2f} dt/ha)")
    print("=" * 80)
    for res in top_results[:10]:  # Print top 10 findings
        print(f"  - MAE: {res['MAE']:.2f} | Count: {res['Count']:<4} | Subgroup: {res['Subgroup']}")
    print("=" * 80 + "\n")


# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_analysis = load_and_merge_data(MODEL_BACKTEST_FILE, FEATURES_FILE)

    if df_analysis is not None:
        # Calculate overall MAE once
        overall_mae = df_analysis['abs_error'].mean()

        print("\n" + "=" * 55)
        print("      Overall Model Performance Summary")
        print("=" * 55)
        print(f"  Overall MAE:              {overall_mae:.2f} dt/ha")
        print("=" * 55 + "\n")

        # --- Run the NEW, automated discovery diagnostic first ---
        find_worst_performing_subgroups(df_analysis, overall_mae)

        # --- Then, run the deep dive on our known "wetness" hypothesis ---
        analyze_error_interactions(df_analysis, OUTPUT_DIR)

        logging.info("✓ Definitive diagnostic analysis complete.")