# File: compare_models.py
# Description: A script to directly compare the performance of the WOFOST simulation
#              against the existing Quantile XGBoost model backtest.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Input files
WOFOST_OUTPUT_FILE = 'data/06_model_output/multi_year_final/final_comparison_1981-2024.csv'
XGBOOST_BACKTEST_FILE = 'reports/figures/district_level_diagnostics/quantile_model_diagnostics/full_backtest_predictions.csv'
FEATURES_FILE = 'data/05_model_input/stage1_preseason_features.csv'

# Output directory for the plot
OUTPUT_DIR = Path('reports/figures')

# Constants
DMC_SUGARBEET = 0.25
Z_SCORE_90_PERCENT_INTERVAL = 1.645  # Z-score for a 90% confidence interval (matches 0.05 and 0.95 quantiles)


def create_comparative_analysis(wofost_path, xgboost_path, features_path, output_dir):
    """
    Compares WOFOST and XGBoost model performance side-by-side.
    """
    logging.info("Starting WOFOST vs. XGBoost comparative analysis...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load and Merge Data ---
    try:
        df_wofost = pd.read_csv(wofost_path)
        df_xgb = pd.read_csv(xgboost_path)
        df_features = pd.read_csv(features_path)
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required file was not found. Error: {e}")
        return

    # --- 2. Data Preparation and Unification ---
    logging.info("Preparing and merging datasets...")
    # Prep WOFOST
    df_wofost['wofost_forecast_fresh_dt'] = df_wofost['forecast_yield_dry_kgha'] / (DMC_SUGARBEET * 100)
    # Create a WOFOST 90% prediction interval from its standard deviation
    wofost_uncertainty_dt = df_wofost['forecast_uncertainty_std'] / (DMC_SUGARBEET * 100)
    df_wofost['wofost_lower'] = df_wofost['wofost_forecast_fresh_dt'] - (
                Z_SCORE_90_PERCENT_INTERVAL * wofost_uncertainty_dt)
    df_wofost['wofost_upper'] = df_wofost['wofost_forecast_fresh_dt'] + (
                Z_SCORE_90_PERCENT_INTERVAL * wofost_uncertainty_dt)

    # Merge
    df_merged = pd.merge(
        df_xgb[['district_no', 'year', 'kreisYield', 'predicted_yield_lower', 'predicted_yield_median',
                'predicted_yield_upper']],
        df_wofost[['district_no', 'year', 'wofost_forecast_fresh_dt', 'wofost_lower', 'wofost_upper']],
        on=['district_no', 'year']
    )

    # --- 3. Calculate Metrics for BOTH Models ---
    logging.info("Calculating comparative metrics for both models...")
    # XGBoost metrics
    df_merged['xgb_abs_error'] = (df_merged['predicted_yield_median'] - df_merged['kreisYield']).abs()
    df_merged['xgb_is_in_interval'] = (df_merged['kreisYield'] >= df_merged['predicted_yield_lower']) & (
                df_merged['kreisYield'] <= df_merged['predicted_yield_upper'])
    df_merged['xgb_interval_width'] = df_merged['predicted_yield_upper'] - df_merged['predicted_yield_lower']

    # WOFOST metrics
    df_merged['wofost_abs_error'] = (df_merged['wofost_forecast_fresh_dt'] - df_merged['kreisYield']).abs()
    df_merged['wofost_is_in_interval'] = (df_merged['kreisYield'] >= df_merged['wofost_lower']) & (
                df_merged['kreisYield'] <= df_merged['wofost_upper'])
    df_merged['wofost_interval_width'] = df_merged['wofost_upper'] - df_merged['wofost_lower']

    # --- 4. Aggregate by Year ---
    logging.info("Aggregating results by year...")
    df_yearly = df_merged.groupby('year').agg(
        xgb_mae=('xgb_abs_error', 'mean'),
        wofost_mae=('wofost_abs_error', 'mean'),
        xgb_coverage=('xgb_is_in_interval', 'mean'),
        wofost_coverage=('wofost_is_in_interval', 'mean'),
        xgb_width=('xgb_interval_width', 'mean'),
        wofost_width=('wofost_interval_width', 'mean')
    ).reset_index()

    # Add weather context
    df_context = df_features.groupby('year')[
        ['summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast']].mean().reset_index()
    df_yearly = pd.merge(df_yearly, df_context, on='year')

    print("\n" + "=" * 110)
    print("      Yearly Head-to-Head Performance: WOFOST vs. XGBoost")
    print("=" * 110)
    print(df_yearly.to_string(index=False, float_format="%.2f"))
    print("=" * 110 + "\n")

    # --- 5. Visualize the Showdown ---
    logging.info("Generating comparative dashboard plot...")
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    start_year, end_year = df_yearly['year'].min(), df_yearly['year'].max()
    fig.suptitle(f'Model Showdown: WOFOST vs. XGBoost ({start_year}-{end_year})', fontsize=18, y=0.99)

    # Panel 1: Accuracy (MAE)
    axes[0].plot(df_yearly['year'], df_yearly['wofost_mae'], marker='o', color='purple', label='WOFOST MAE')
    axes[0].plot(df_yearly['year'], df_yearly['xgb_mae'], marker='s', color='darkgreen', label='XGBoost MAE')
    axes[0].set_title('Accuracy: Mean Absolute Error (Lower is Better)', fontsize=14)
    axes[0].set_ylabel('MAE (dt/ha)')
    axes[0].legend()

    # Panel 2: Interval Coverage (Reliability)
    axes[1].plot(df_yearly['year'], df_yearly['wofost_coverage'] * 100, marker='o', color='purple',
                 label='WOFOST Coverage')
    axes[1].plot(df_yearly['year'], df_yearly['xgb_coverage'] * 100, marker='s', color='darkgreen',
                 label='XGBoost Coverage')
    axes[1].axhline(90, color='black', linestyle='--', lw=1.5, label='Target (90%)')
    axes[1].set_title('Reliability: Prediction Interval Coverage (Closer to 90% is Better)', fontsize=14)
    axes[1].set_ylabel('Coverage (%)')
    axes[1].set_ylim(0, 105)
    axes[1].legend()

    # Panel 3: Interval Width (Confidence)
    axes[2].plot(df_yearly['year'], df_yearly['wofost_width'], marker='o', color='purple', label='WOFOST Width')
    axes[2].plot(df_yearly['year'], df_yearly['xgb_width'], marker='s', color='darkgreen', label='XGBoost Width')
    axes[2].set_title('Confidence: Average Prediction Interval Width (Lower is More Confident)', fontsize=14)
    axes[2].set_ylabel('Interval Width (dt/ha)')
    axes[2].legend()

    # Panel 4: Weather Context
    ax4 = axes[3]
    ax4_twin = ax4.twinx()
    colors = ['#86BBD8' if x > 0 else '#F26419' for x in df_yearly['summer_precip_anomaly_forecast']]
    ax4.bar(df_yearly['year'], df_yearly['summer_precip_anomaly_forecast'], color=colors, alpha=0.7)
    ax4_twin.plot(df_yearly['year'], df_yearly['summer_temp_anomaly_forecast'], marker='^', color='#F6AE2D')
    ax4.set_title('Context: Average Summer Weather Anomalies', fontsize=14)
    ax4.set_ylabel('Precip Anomaly (%)', color='#33658A')
    ax4_twin.set_ylabel('Temp Anomaly (°C)', color='#F6AE2D')

    plt.xticks(df_yearly['year'], rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_filename = f'model_showdown_dashboard_{start_year}-{end_year}.png'
    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=300)
    logging.info(f"✓ Comparative analysis plot saved to: {output_path}")

    plt.show()


if __name__ == '__main__':
    create_comparative_analysis(WOFOST_OUTPUT_FILE, XGBOOST_BACKTEST_FILE, FEATURES_FILE, OUTPUT_DIR)