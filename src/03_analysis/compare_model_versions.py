# File: src/models/compare_model_versions.py
# Description: The definitive "Proof of Work" script. It compares the final hybrid model
#              (Residual Fitting) against its own strong time-series baseline to provide a
#              final, comprehensive performance verdict.
#
# REVISED VERSION v2: Updated to use the new, honestly-validated time-series models
# as the primary baselines for a fair and powerful comparison.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from sklearn.metrics import r2_score

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Input Files ---
# Point to the backtest results of your final, residual-fitting model
NEW_MODEL_BACKTEST_FILE = 'reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv'
# The honest walk-forward forecast file now serves as our primary baseline
HONEST_FORECAST_FILE = 'data/09_model_output_walkforward_final/final_honest_forecasts.csv'
# The features file is used for weather context
FEATURES_FILE = 'data/05_model_input/stage1_preseason_features.csv'

# --- Output Directory ---
OUTPUT_DIR = Path('reports/figures/final_proof_of_work')


def create_final_proof_of_work_dashboard():
    """
    Generates a comprehensive dashboard comparing the final models to provide a
    verdict on the project's success.
    """
    logging.info("--- Starting Final Proof of Work Analysis ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load All Data Sources ---
    try:
        df_new = pd.read_csv(NEW_MODEL_BACKTEST_FILE)
        df_forecast = pd.read_csv(HONEST_FORECAST_FILE)
        df_features = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError as e:
        logging.error(f"❌ FATAL: A required file was not found. Please check paths. Error: {e}")
        return

    # --- 2. Prepare Forecasts for the Three Key Models ---
    logging.info("Preparing forecasts from all models for comparison...")

    # Model 1: Final Hybrid Model (Our Champion: Time-Series + XGBoost on Residuals)
    df_new = df_new[['district_no', 'year', 'kreisYield', 'predicted_yield_median']].rename(
        columns={'predicted_yield_median': 'final_hybrid_pred'})

    # Model 2: Final Time-Series Model (Trend + ARIMA Correction)
    df_ts_final = df_forecast[['district_no', 'year', 'final_corrected_forecast']].rename(
        columns={'final_corrected_forecast': 'ts_final_pred'})

    # Model 3: Base Trend Model (GAM Trend Only)
    df_ts_base = df_forecast[['district_no', 'year', 'base_trend_forecast']].rename(
        columns={'base_trend_forecast': 'ts_base_pred'})

    # --- 3. Merge into a Single DataFrame for a True Apples-to-Apples Comparison ---
    df_merged = pd.merge(df_new, df_ts_final, on=['district_no', 'year'], how='inner')
    df_merged = pd.merge(df_merged, df_ts_base, on=['district_no', 'year'], how='inner')
    df_merged.dropna(inplace=True)

    # --- 4. Calculate Overall Performance Metrics ---
    models = {
        "Final Hybrid Model (TS + XGB Residuals)": "final_hybrid_pred",
        "Time-Series Model (Trend + ARIMA)": "ts_final_pred",
        "Base Trend Model (GAM Only)": "ts_base_pred"
    }

    results = []
    for name, pred_col in models.items():
        mae = (df_merged[pred_col] - df_merged['kreisYield']).abs().mean()
        r2 = r2_score(df_merged['kreisYield'], df_merged[pred_col])
        results.append({'Model': name, 'MAE': mae, 'R-squared': r2})

    df_summary = pd.DataFrame(results).sort_values('MAE').reset_index(drop=True)

    print("\n" + "=" * 80)
    print("      FINAL PROOF OF WORK: OVERALL MODEL PERFORMANCE SCORECARD")
    print("=" * 80)
    print(df_summary.to_string(index=False, float_format="%.4f"))
    print("=" * 80 + "\n")

    # --- 5. Calculate Yearly MAE for Visualization ---
    for name, pred_col in models.items():
        df_merged[f'{name}_mae'] = (df_merged[pred_col] - df_merged['kreisYield']).abs()

    yearly_mae = df_merged.groupby('year').agg({f'{name}_mae': 'mean' for name in models}).reset_index()

    # Add weather context
    df_context = df_features.groupby('year')[
        ['summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast']].mean().reset_index()
    yearly_mae = pd.merge(yearly_mae, df_context, on='year', how='left')
    yearly_mae = yearly_mae[yearly_mae['year'] >= yearly_mae['year'].min()].copy()

    # --- 6. Generate the Final Dashboard Plot ---
    logging.info("Generating final proof of work dashboard...")
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(18, 16), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    start_year, end_year = yearly_mae['year'].min(), yearly_mae['year'].max()
    fig.suptitle(f'Final Proof of Work: Model Performance Showdown ({start_year}-{end_year})', fontsize=22, y=0.98)

    # Panel 1: Accuracy Showdown
    ax1 = axes[0]
    ax1.plot(yearly_mae['year'], yearly_mae['Base Trend Model (GAM Only)_mae'], marker='x', color='blue', linestyle=':',
             label='Base Trend (GAM Only)')
    ax1.plot(yearly_mae['year'], yearly_mae['Time-Series Model (Trend + ARIMA)_mae'], marker='^', color='purple', alpha=0.7, linestyle='--',
             label='Time-Series (GAM + ARIMA)')
    ax1.plot(yearly_mae['year'], yearly_mae['Final Hybrid Model (TS + XGB Residuals)_mae'], marker='s', markersize=8,
             color='darkgreen', linewidth=3, label='Final Hybrid Model (Champion)')

    ax1.set_title('Accuracy Showdown: Mean Absolute Error by Year (Lower is Better)', fontsize=18)
    ax1.set_ylabel('Mean Absolute Error (MAE) in dt/ha', fontsize=14)
    ax1.legend(fontsize=14)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.7)

    # Panel 2: Weather Context
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    colors = ['#86BBD8' if x > 0 else '#F26419' for x in yearly_mae['summer_precip_anomaly_forecast']]
    ax2.bar(yearly_mae['year'], yearly_mae['summer_precip_anomaly_forecast'], color=colors, alpha=0.7)
    ax2_twin.plot(yearly_mae['year'], yearly_mae['summer_temp_anomaly_forecast'], marker='^', color='#F6AE2D')
    ax2.set_title('Context: Average Summer Weather Anomalies', fontsize=18)
    ax2.set_ylabel('Precipitation Anomaly (%)', color='#33658A', fontsize=14)
    ax2_twin.set_ylabel('Temperature Anomaly (°C)', color='#F6AE2D', fontsize=14)
    ax2.set_xlabel('Year', fontsize=16)

    plt.xticks(yearly_mae['year'], rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_filename = f'final_proof_of_work_dashboard_{start_year}-{end_year}.png'
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300)
    logging.info(f"✓ Final Proof of Work dashboard saved successfully to: {output_path}")
    plt.show()


if __name__ == '__main__':
    create_final_proof_of_work_dashboard()