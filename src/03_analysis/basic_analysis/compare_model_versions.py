# File: src/models/compare_model_versions.py
# Description: This script provides a comprehensive performance comparison between different
#              modeling approaches, including the Adaptive CQR model, a hybrid XGBoost model,
#              and various time-series baselines.
#
# REVISED VERSION v10: The definitive version. Implements the proper Interval Score to
#                      holistically evaluate prediction interval quality. The nominal
#                      coverage is now correctly configured to 95%. The dashboard is
#                      upgraded to three panels to visualize accuracy, sharpness, and score.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import sys
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Quantile Configuration ---
# Define the ACTUAL column names for the lower and upper prediction bounds.
LOWER_BOUND_COL_NAME = 'predicted_yield_lower'
UPPER_BOUND_COL_NAME = 'predicted_yield_upper'

# **CRITICAL**: Set the nominal coverage percentage that your interval represents.
# For a 95% interval (e.g., from 2.5 to 97.5 quantiles, or 5 to 95), set this to 95.0.
NOMINAL_COVERAGE_PERCENT = 95.0
ALPHA = 1 - (NOMINAL_COVERAGE_PERCENT / 100.0)  # Alpha is the desired miss-rate (e.g., 0.05 for 95%)

# --- Input Files ---
ADAPTIVE_CQR_PREDICTIONS_FILE = 'reports/figures/district_level_diagnostics/adaptive_cqr_champion/full_backtest_predictions.csv'
HYBRID_XGB_PREDICTIONS_FILE = 'reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv'
TIMESERIES_FORECAST_FILE = 'data/05_model_input/wofost_walkforward/final_honest_forecasts.csv'

# --- Output Directory ---
OUTPUT_DIR = Path('reports/figures/district_level_diagnostics/adaptive_cqr_champion')


def validate_dataframe_columns(df, required_cols, filename):
    """Checks if a dataframe contains all required columns and logs errors if not."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ FATAL: Missing required columns in file: {filename}")
        logging.error(f"   Missing columns: {missing_cols}")
        logging.error(f"   Available columns are: {df.columns.tolist()}")
        sys.exit(1)
    return True

def calculate_interval_score(y_true, lower, upper, alpha):
    """Calculates the Winkler Interval Score. Lower is better."""
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return width + penalty_lower + penalty_upper


def create_model_comparison_dashboard():
    """
    Generates a comprehensive dashboard comparing models on point accuracy (MAE)
    and interval quality (Coverage, Width, and Interval Score).
    """
    logging.info("--- Starting Comprehensive Model Comparison Analysis ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load All Data Sources ---
    try:
        logging.info(f"Loading Adaptive CQR predictions from: {ADAPTIVE_CQR_PREDICTIONS_FILE}")
        df_adaptive_cqr = pd.read_csv(ADAPTIVE_CQR_PREDICTIONS_FILE)
        logging.info(f"Columns found in Adaptive CQR file: {df_adaptive_cqr.columns.tolist()}")
        logging.info(f"Loading Hybrid XGB predictions from: {HYBRID_XGB_PREDICTIONS_FILE}")
        df_hybrid_xgb = pd.read_csv(HYBRID_XGB_PREDICTIONS_FILE)
        logging.info(f"Columns found in Hybrid XGB file: {df_hybrid_xgb.columns.tolist()}")
        logging.info(f"Loading time-series baselines from: {TIMESERIES_FORECAST_FILE}")
        df_timeseries = pd.read_csv(TIMESERIES_FORECAST_FILE)
    except FileNotFoundError as e:
        logging.error(f"❌ FATAL: Input file not found. Details: {e}")
        sys.exit(1)

    # --- 2. Define required columns and VALIDATE DataFrames ---
    cqr_required_cols = ['district_no', 'year', 'kreisYield', 'predicted_yield_median', LOWER_BOUND_COL_NAME, UPPER_BOUND_COL_NAME]
    validate_dataframe_columns(df_adaptive_cqr, cqr_required_cols, ADAPTIVE_CQR_PREDICTIONS_FILE)
    xgb_required_cols = ['district_no', 'year', 'predicted_yield_median', LOWER_BOUND_COL_NAME, UPPER_BOUND_COL_NAME]
    validate_dataframe_columns(df_hybrid_xgb, xgb_required_cols, HYBRID_XGB_PREDICTIONS_FILE)
    logging.info("✓ All data sources loaded and validated successfully.")

    # --- 3. Prepare Forecasts ---
    logging.info("Preparing forecasts from all models for comparison...")
    df_adaptive_cqr = df_adaptive_cqr[cqr_required_cols].rename(columns={
        'predicted_yield_median': 'adaptive_cqr_pred',
        LOWER_BOUND_COL_NAME: 'adaptive_cqr_lower',
        UPPER_BOUND_COL_NAME: 'adaptive_cqr_upper'})
    df_hybrid_xgb = df_hybrid_xgb[xgb_required_cols].rename(columns={
        'predicted_yield_median': 'hybrid_xgb_pred',
        LOWER_BOUND_COL_NAME: 'hybrid_xgb_lower',
        UPPER_BOUND_COL_NAME: 'hybrid_xgb_upper'})
    df_ts_final = df_timeseries[['district_no', 'year', 'final_corrected_forecast']].rename(columns={'final_corrected_forecast': 'ts_final_pred'})
    df_ts_base = df_timeseries[['district_no', 'year', 'base_trend_forecast']].rename(columns={'base_trend_forecast': 'ts_base_pred'})

    # --- 4. Merge DataFrames ---
    df_merged = pd.merge(df_adaptive_cqr, df_hybrid_xgb, on=['district_no', 'year'], how='inner')
    df_merged = pd.merge(df_merged, df_ts_final, on=['district_no', 'year'], how='inner')
    df_merged = pd.merge(df_merged, df_ts_base, on=['district_no', 'year'], how='inner')
    df_merged.dropna(inplace=True)
    df_merged.sort_values(by=['district_no', 'year'], inplace=True)

    # --- 5. Calculate Leak-Proof Linear Trend Baseline ---
    # ... (Code remains the same) ...
    logging.info("Calculating leak-proof basic linear trend...")
    df_merged['basic_linear_trend_pred'] = np.nan
    predictions = []
    for district, group in tqdm(df_merged.groupby('district_no'), desc="Processing Districts for Trend Baseline"):
        district_preds = []
        for year_to_predict in group['year']:
            train_df = group[group['year'] < year_to_predict]
            if len(train_df) >= 2:
                lr = LinearRegression()
                lr.fit(train_df[['year']], train_df['kreisYield'])
                prediction = lr.predict(pd.DataFrame({'year': [year_to_predict]}))
                district_preds.append(prediction[0])
            else:
                district_preds.append(np.nan)
        predictions.extend(district_preds)
    df_merged['basic_linear_trend_pred'] = predictions
    df_merged.dropna(subset=['basic_linear_trend_pred'], inplace=True)

    # --- 6. Calculate Overall Performance Metrics ---
    # Point forecast metrics
    models_point = {
        "Adaptive CQR": "adaptive_cqr_pred", "Hybrid (TS+XGB)": "hybrid_xgb_pred",
        "TS (GAM+ARIMA)": "ts_final_pred", "TS (GAM Only)": "ts_base_pred",
        "Linear Trend": "basic_linear_trend_pred"}
    point_results = []
    for name, pred_col in models_point.items():
        mae = (df_merged[pred_col] - df_merged['kreisYield']).abs().mean()
        r2 = r2_score(df_merged['kreisYield'], df_merged[pred_col])
        point_results.append({'Model': name, 'MAE': mae, 'R-squared': r2})
    df_summary = pd.DataFrame(point_results).sort_values('MAE').reset_index(drop=True)

    print("\n" + "=" * 80)
    print("      MODEL COMPARISON: POINT FORECAST ACCURACY (MEDIAN)")
    print("=" * 80)
    print(df_summary.to_string(index=False, float_format="%.4f"))
    print("=" * 80)

    # --- 7. Quantile Quality Analysis (Coverage, Width, and Interval Score) ---
    logging.info("Performing comprehensive quantile quality analysis...")
    quantile_results = []
    quantile_models = {"Adaptive CQR": ("adaptive_cqr_lower", "adaptive_cqr_upper"),
                       "Hybrid (TS+XGB)": ("hybrid_xgb_lower", "hybrid_xgb_upper")}
    for name, (lower_col, upper_col) in quantile_models.items():
        df_merged[f'{name}_width'] = df_merged[upper_col] - df_merged[lower_col]
        df_merged[f'{name}_score'] = calculate_interval_score(df_merged['kreisYield'], df_merged[lower_col], df_merged[upper_col], ALPHA)
        avg_width = df_merged[f'{name}_width'].mean()
        avg_score = df_merged[f'{name}_score'].mean()
        is_covered = (df_merged['kreisYield'] >= df_merged[lower_col]) & (df_merged['kreisYield'] <= df_merged[upper_col])
        coverage_percent = is_covered.mean() * 100
        quantile_results.append({
            'Model': name, 'Interval Score (Lower is Better)': avg_score,
            'Coverage (%)': coverage_percent, 'Nominal Coverage (%)': NOMINAL_COVERAGE_PERCENT,
            'Avg. Interval Width': avg_width})
    df_quantile_summary = pd.DataFrame(quantile_results).sort_values('Interval Score (Lower is Better)').reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f"      MODEL COMPARISON: PREDICTION INTERVAL QUALITY ({int(NOMINAL_COVERAGE_PERCENT)}% Interval)")
    print("=" * 80)
    print(df_quantile_summary.to_string(index=False, float_format="%.4f"))
    print("=" * 80 + "\n")

    # --- 8. Prepare Data for Plotting ---
    for name, pred_col in models_point.items():
        df_merged[f'{name}_mae'] = (df_merged[pred_col] - df_merged['kreisYield']).abs()
    agg_dict = {f'{name}_mae': 'mean' for name in models_point}
    agg_dict.update({f'{name}_width': 'mean' for name in quantile_models})
    agg_dict.update({f'{name}_score': 'mean' for name in quantile_models})
    yearly_stats = df_merged.groupby('year').agg(agg_dict).reset_index()

    # --- 9. Generate the Final 3-Panel Dashboard Plot ---
    logging.info("Generating final 3-panel model comparison dashboard...")
    sns.set_style("whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 24), sharex=True)
    start_year, end_year = yearly_stats['year'].min(), yearly_stats['year'].max()
    fig.suptitle(f'Comprehensive Model Performance Comparison ({start_year}-{end_year})', fontsize=28, y=0.98)

    # Panel 1: Accuracy Comparison (MAE)
    ax1.plot(yearly_stats['year'], yearly_stats['Linear Trend_mae'], marker='o', color='grey', linestyle=':', label='Linear Trend')
    ax1.plot(yearly_stats['year'], yearly_stats['TS (GAM Only)_mae'], marker='x', color='cornflowerblue', linestyle='--', label='TS (GAM Only)')
    ax1.plot(yearly_stats['year'], yearly_stats['TS (GAM+ARIMA)_mae'], marker='^', color='slateblue', linestyle='--', label='TS (GAM+ARIMA)')
    ax1.plot(yearly_stats['year'], yearly_stats['Hybrid (TS+XGB)_mae'], marker='s', color='darkorange', linewidth=2.5, label='Hybrid (TS+XGB)')
    ax1.plot(yearly_stats['year'], yearly_stats['Adaptive CQR_mae'], marker='*', markersize=10, color='darkgreen', linewidth=3.5, label='Adaptive CQR')
    ax1.set_title('Panel 1: Point Accuracy - Mean Absolute Error by Year (Lower is Better)', fontsize=20, pad=10)
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=16)
    ax1.legend(fontsize=16, title="Model", title_fontsize=16)

    # Panel 2: Interval Sharpness (Width)
    cqr_cov = df_quantile_summary.loc[df_quantile_summary['Model'] == 'Adaptive CQR', 'Coverage (%)'].iloc[0]
    hybrid_cov = df_quantile_summary.loc[df_quantile_summary['Model'] == 'Hybrid (TS+XGB)', 'Coverage (%)'].iloc[0]
    ax2.plot(yearly_stats['year'], yearly_stats['Hybrid (TS+XGB)_width'], marker='s', color='darkorange', linewidth=2.5, label=f'Hybrid (TS+XGB) — Overall Coverage: {hybrid_cov:.1f}%')
    ax2.plot(yearly_stats['year'], yearly_stats['Adaptive CQR_width'], marker='*', markersize=10, color='darkgreen', linewidth=3.5, label=f'Adaptive CQR — Overall Coverage: {cqr_cov:.1f}%')
    ax2.set_title('Panel 2: Interval Sharpness - Average Width by Year (Lower is Better)', fontsize=20, pad=10)
    ax2.set_ylabel('Average Interval Width', fontsize=16)
    ax2.legend(fontsize=16, title="Model & Achieved Coverage", title_fontsize=16)

    # Panel 3: Overall Interval Quality (Score)
    ax3.plot(yearly_stats['year'], yearly_stats['Hybrid (TS+XGB)_score'], marker='s', color='darkorange', linewidth=2.5, label='Hybrid (TS+XGB)')
    ax3.plot(yearly_stats['year'], yearly_stats['Adaptive CQR_score'], marker='*', markersize=10, color='darkgreen', linewidth=3.5, label='Adaptive CQR')
    ax3.set_title(f'Panel 3: Overall Interval Quality - {int(NOMINAL_COVERAGE_PERCENT)}% Interval Score by Year (Lower is Better)', fontsize=20, pad=10)
    ax3.set_ylabel('Mean Interval Score', fontsize=16)
    ax3.set_xlabel('Year', fontsize=16)
    ax3.legend(fontsize=16, title="Model", title_fontsize=16)

    for ax in [ax1, ax2, ax3]:
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(yearly_stats['year'], rotation=45, ha="right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_filename = f'model_comparison_dashboard_v3_final_{start_year}-{end_year}.png'
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300)
    logging.info(f"✓ Final 3-panel dashboard saved successfully to: {output_path}")
    plt.show()

if __name__ == '__main__':
    create_model_comparison_dashboard()