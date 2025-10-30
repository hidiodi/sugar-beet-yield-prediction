# File: src/models/compare_all_models_final.py
# Description: The ultimate "Proof of Work" script. It compares the final hybrid model with
#              and without spatial features against the time-series baselines for the 2000-2024 period.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from sklearn.metrics import r2_score

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Input Files ---
# --- THIS IS THE KEY UPDATE ---
# Pointing to the new, correct, 2000-2024 backtest file
SPATIAL_MODEL_BACKTEST_FILE = 'reports/figures/district_level_diagnostics/final_hybrid_champion/full_backtest_predictions_hybrid.csv'

# Keep the old ones for comparison
NON_SPATIAL_MODEL_BACKTEST_FILE = 'reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv'
HONEST_FORECAST_FILE = 'data/09_model_output_walkforward_final/final_honest_forecasts.csv'
FEATURES_FILE = 'data/05_model_input/stage1_preseason_features.csv'

# --- Output Directory ---
OUTPUT_DIR = Path('reports/figures/final_proof_of_work_ultimate')


def create_ultimate_dashboard():
    logging.info("--- Starting Ultimate Proof of Work Analysis ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load All Data Sources
    try:
        df_spatial = pd.read_csv(SPATIAL_MODEL_BACKTEST_FILE)
        df_non_spatial = pd.read_csv(NON_SPATIAL_MODEL_BACKTEST_FILE)
        df_forecast = pd.read_csv(HONEST_FORECAST_FILE)
        df_features = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError as e:
        logging.error(f"❌ FATAL: A required file was not found. Please run all backtests first. Error: {e}");
        return

    # 2. Prepare Forecasts for All Four Models
    df_spatial = df_spatial[['district_no', 'year', 'kreisYield', 'predicted_yield_median']].rename(
        columns={'predicted_yield_median': 'spatial_pred'})
    df_non_spatial = df_non_spatial[['district_no', 'year', 'predicted_yield_median']].rename(
        columns={'predicted_yield_median': 'non_spatial_pred'})
    df_ts_final = df_forecast[['district_no', 'year', 'final_corrected_forecast']].rename(
        columns={'final_corrected_forecast': 'ts_final_pred'})
    df_ts_base = df_forecast[['district_no', 'year', 'base_trend_forecast']].rename(
        columns={'base_trend_forecast': 'ts_base_pred'})

    # 3. Merge into a Single DataFrame (inner merge ensures years align)
    df_merged = pd.merge(df_spatial, df_non_spatial, on=['district_no', 'year'], how='inner')
    df_merged = pd.merge(df_merged, df_ts_final, on=['district_no', 'year'], how='inner')
    df_merged = pd.merge(df_merged, df_ts_base, on=['district_no', 'year'], how='inner')
    df_merged.dropna(inplace=True)

    # 4. Calculate Overall Performance Metrics
    models = {
        "Spatial Hybrid Model (Final Champion)": "spatial_pred",
        "Non-Spatial Hybrid Model": "non_spatial_pred",
        "Time-Series Model (Trend + ARIMA)": "ts_final_pred",
        "Base Trend Model (GAM Only)": "ts_base_pred"
    }
    results = [{'Model': name, 'MAE': (df_merged[pred_col] - df_merged['kreisYield']).abs().mean(),
                'R-squared': r2_score(df_merged['kreisYield'], df_merged[pred_col])} for name, pred_col in
               models.items()]
    df_summary = pd.DataFrame(results).sort_values('MAE').reset_index(drop=True)

    print("\n" + "=" * 80);
    print("      ULTIMATE PROOF OF WORK: FINAL MODEL PERFORMANCE SCORECARD");
    print("=" * 80)
    print(df_summary.to_string(index=False, float_format="%.4f"));
    print("=" * 80 + "\n")

    # 5. Generate the Final Dashboard Plot
    for name, pred_col in models.items(): df_merged[f'{name}_mae'] = (
                df_merged[pred_col] - df_merged['kreisYield']).abs()
    yearly_mae = df_merged.groupby('year').agg({f'{name}_mae': 'mean' for name in models}).reset_index()
    df_context = df_features.groupby('year')[
        ['summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast']].mean().reset_index()
    yearly_mae = pd.merge(yearly_mae, df_context, on='year', how='left')

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(18, 16), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    start_year, end_year = yearly_mae['year'].min(), yearly_mae['year'].max()
    fig.suptitle(f'Ultimate Proof of Work: Model Hierarchy Showdown ({start_year}-{end_year})', fontsize=22, y=0.98)

    ax1 = axes[0]
    ax1.plot(yearly_mae['year'], yearly_mae['Base Trend Model (GAM Only)_mae'], marker='x', color='blue', linestyle=':',
             label='Baseline: Trend Only')
    ax1.plot(yearly_mae['year'], yearly_mae['Time-Series Model (Trend + ARIMA)_mae'], marker='^', color='purple',
             linestyle='--', label='Lvl 2: Time-Series Correction')
    ax1.plot(yearly_mae['year'], yearly_mae['Non-Spatial Hybrid Model_mae'], marker='o', color='darkorange',
             linestyle='-', label='Lvl 3: Hybrid (TS + XGBoost)')
    ax1.plot(yearly_mae['year'], yearly_mae['Spatial Hybrid Model (Final Champion)_mae'], marker='*', markersize=12,
             color='darkgreen', linewidth=3, label='Lvl 4: Spatial Hybrid (Final Champion)')
    ax1.set_title('Accuracy Showdown: Mean Absolute Error by Year (Lower is Better)', fontsize=18)
    ax1.set_ylabel('Mean Absolute Error (MAE) in dt/ha', fontsize=14);
    ax1.legend(fontsize=14)

    ax2 = axes[1];
    ax2_twin = ax2.twinx()
    colors = ['#86BBD8' if x > 0 else '#F26419' for x in yearly_mae['summer_precip_anomaly_forecast']]
    ax2.bar(yearly_mae['year'], yearly_mae['summer_precip_anomaly_forecast'], color=colors, alpha=0.7)
    ax2_twin.plot(yearly_mae['year'], yearly_mae['summer_temp_anomaly_forecast'], marker='^', color='#F6AE2D')
    ax2.set_title('Context: Average Summer Weather Anomalies', fontsize=18);
    ax2.set_ylabel('Precipitation Anomaly (%)', color='#33658A', fontsize=14)
    ax2_twin.set_ylabel('Temperature Anomaly (°C)', color='#F6AE2D', fontsize=14);
    ax2.set_xlabel('Year', fontsize=16)

    plt.xticks(yearly_mae['year'], rotation=45);
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = OUTPUT_DIR / f'ultimate_proof_of_work_dashboard_{start_year}-{end_year}.png'
    plt.savefig(output_path, dpi=300)
    logging.info(f"✓ Ultimate Proof of Work dashboard saved to: {output_path}");
    plt.show()


if __name__ == '__main__':
    create_ultimate_dashboard()