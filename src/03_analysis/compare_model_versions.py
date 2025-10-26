# File: src/models/diagnose_hybrid_model.py
# Description: A diagnostic script to analyze the errors of the final hybrid model
#              and identify systematic weaknesses for future improvement.

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Inputs ---
MODEL_BACKTEST_FILE = 'reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv'
FEATURES_FILE = 'data/05_model_input/stage1_preseason_features.csv'
GEOJSON_PATH = 'data/01_raw/districts_official.geojson'

# --- Outputs ---
OUTPUT_DIR = Path('reports/figures/hybrid_model_diagnostics')

# --- Thresholds for Analysis ---
DROUGHT_THRESHOLD = -0.1  # Precip anomaly below -10%
WET_THRESHOLD = 0.1  # Precip anomaly above +10%
HOT_THRESHOLD = 0.5  # Temp anomaly above +0.5 C


def load_and_merge_data(backtest_path, features_path):
    """Loads backtest and feature data and merges them for analysis."""
    try:
        df_backtest = pd.read_csv(backtest_path)
        df_features = pd.read_csv(features_path)
    except FileNotFoundError as e:
        logging.error(f"❌ FATAL: A required file was not found. Error: {e}")
        return None

    # Calculate error on the median prediction
    df_backtest['error'] = df_backtest['predicted_yield_median'] - df_backtest['kreisYield']
    df_backtest['abs_error'] = df_backtest['error'].abs()

    # Merge with features to get weather context
    df_merged = pd.merge(
        df_backtest,
        df_features[['district_no', 'year', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast']],
        on=['district_no', 'year']
    )
    return df_merged


def plot_error_vs_weather(df: pd.DataFrame, output_dir: Path):
    """Plots prediction error against key summer weather anomalies."""
    logging.info("Generating error vs. weather anomaly plots...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Hybrid Model Error Diagnosis vs. Weather Conditions', fontsize=18)

    # Error vs. Temperature Anomaly
    sns.scatterplot(ax=axes[0], data=df, x='summer_temp_anomaly_forecast', y='error', alpha=0.2)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Error vs. Summer Temperature Anomaly', fontsize=14)
    axes[0].set_xlabel('Temperature Anomaly (°C)')
    axes[0].set_ylabel('Prediction Error (Pred - Actual)')
    axes[0].grid(True, linestyle=':')

    # Error vs. Precipitation Anomaly
    sns.scatterplot(ax=axes[1], data=df, x='summer_precip_anomaly_forecast', y='error', alpha=0.2)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title('Error vs. Summer Precipitation Anomaly', fontsize=14)
    axes[1].set_xlabel('Precipitation Anomaly (%)')
    axes[1].set_ylabel('Prediction Error (Pred - Actual)')
    axes[1].grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = output_dir / 'error_vs_weather_anomalies.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"✓ Saved error vs. weather plot to {output_path}")


# --- REPLACE THE OLD FUNCTION WITH THIS CORRECTED VERSION ---
def plot_geographic_error_map(df: pd.DataFrame, geojson_path: str, output_dir: Path):
    """Maps the average prediction error by district."""
    logging.info("Generating geographic error map...")
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.rename(columns={'id': 'district_no'})
    # Ensure the GeoDataFrame key is a zero-padded string
    gdf['district_no'] = gdf['district_no'].astype(str).str.zfill(5)

    district_errors = df.groupby('district_no')['error'].mean().reset_index()

    # --- FIX: Ensure the district_errors key is ALSO a zero-padded string ---
    # This is the line that fixes the bug.
    district_errors['district_no'] = district_errors['district_no'].astype(str).str.zfill(5)
    # --- END FIX ---

    merged_gdf = gdf.merge(district_errors, on='district_no', how='left')

    # Determine symmetric color scale limit
    max_abs_error = merged_gdf['error'].abs().max()

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(
        column='error', cmap='RdBu_r', linewidth=0.5, ax=ax, edgecolor='0.8',
        legend=True,
        legend_kwds={'label': "Average Prediction Error (dt/ha)\n(Red=Over-prediction, Blue=Under-prediction)",
                     'orientation': "horizontal"},
        missing_kwds={'color': 'lightgrey'}, vmin=-max_abs_error, vmax=max_abs_error
    )
    ax.set_title('Geographic Bias: Average Prediction Error by District', fontsize=16)
    ax.set_axis_off()
    output_path = output_dir / 'geographic_error_map.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"✓ Saved geographic error map to {output_path}")

def summarize_performance_by_condition(df: pd.DataFrame):
    """Calculates and prints the model's MAE under different weather conditions."""
    logging.info("Calculating performance under specific weather conditions...")

    # Define conditions
    df['is_drought'] = df['summer_precip_anomaly_forecast'] < DROUGHT_THRESHOLD
    df['is_wet'] = df['summer_precip_anomaly_forecast'] > WET_THRESHOLD
    df['is_hot'] = df['summer_temp_anomaly_forecast'] > HOT_THRESHOLD

    # Calculate MAE for each condition
    mae_overall = df['abs_error'].mean()
    mae_drought = df[df['is_drought']]['abs_error'].mean()
    mae_wet = df[df['is_wet']]['abs_error'].mean()
    mae_hot = df[df['is_hot']]['abs_error'].mean()
    mae_normal = df[~df['is_drought'] & ~df['is_wet'] & ~df['is_hot']]['abs_error'].mean()

    print("\n" + "=" * 55)
    print("      Hybrid Model MAE by Weather Condition")
    print("=" * 55)
    print(f"  Overall MAE:              {mae_overall:.2f} dt/ha")
    print("-" * 55)
    print(f"  MAE in 'Hot' Summers:       {mae_hot:.2f} dt/ha")
    print(f"  MAE in 'Drought' Summers:   {mae_drought:.2f} dt/ha")
    print(f"  MAE in 'Wet' Summers:       {mae_wet:.2f} dt/ha")
    print(f"  MAE in 'Normal' Summers:    {mae_normal:.2f} dt/ha")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_analysis = load_and_merge_data(MODEL_BACKTEST_FILE, FEATURES_FILE)

    if df_analysis is not None:
        summarize_performance_by_condition(df_analysis)
        plot_error_vs_weather(df_analysis, OUTPUT_DIR)
        plot_geographic_error_map(df_analysis, GEOJSON_PATH, OUTPUT_DIR)
        logging.info("✓ Diagnostic analysis complete.")