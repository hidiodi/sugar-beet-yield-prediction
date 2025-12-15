# File: src/models/backtest_adaptive_cqr_model.py
# Description: Backtesting for Adaptive Conformalized Quantile Regression (CQR).
#              Refactored to use the "Risk-Based" Residual Strategy (v5.0).

import pandas as pd
import geopandas as gpd
import joblib
from xgboost import XGBRegressor
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
XGB_CONFIG = config.XGBOOST_TRAINING_CONFIG
# CQR specific paths but reusing the main config structure where possible
LOWER_MODEL_PATH = config.BASE_DIR / 'src/models/final_quantile_model_lower.joblib'
MEDIAN_MODEL_PATH = config.BASE_DIR / 'src/models/final_quantile_model_median.joblib'
UPPER_MODEL_PATH = config.BASE_DIR / 'src/models/final_quantile_model_upper.joblib'

# We use the exact same data and features as the main model
DATA_PATH = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
GEOJSON_PATH = config.DATA_DIR / '01_raw/districts_official.geojson'
REPORT_DIR = config.BASE_DIR / 'reports/figures/district_level_diagnostics/adaptive_cqr_champion'

BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2024
CALIBRATION_WINDOW_SIZE = 3  # Years of recent data to use for conformity score calibration
TARGET_COVERAGE = 0.95
LOW_DATA_THRESHOLD = 10
MIN_DATAPOINTS_FOR_PLOT = 10


def run_adaptive_cqr_backtest(df: pd.DataFrame, feature_cols: list, model_lower_clone: XGBRegressor,
                              model_median_clone: XGBRegressor, model_upper_clone: XGBRegressor):
    """
    Performs a rolling forecast using CQR on the RESIDUALS.
    """
    print(f"\n--- Starting Adaptive CQR Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")

    # 1. Global Baseline Calculation (Standardized v5.0 logic)
    # We predict residuals from the 5-year rolling trend (shifted).
    baseline_col = 'yield_rolling_trend'
    target_col = 'trend_residual'

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # Anchor: 5-Year Rolling Average (Lagged)
    df[baseline_col] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=3).mean().shift(1)
    )

    # Target: Residual
    df[target_col] = df['kreisYield'] - df[baseline_col]

    # Clean for backtest availability
    # We need baseline, target, and the 'stat_trend_forecast' anchor
    valid_mask = df[baseline_col].notna() & df[target_col].notna() & df['stat_trend_forecast'].notna()

    print(f" -> Backtesting with {len(feature_cols)} features.")

    # Ensure we have enough history for the first calibration window
    start_year = df[valid_mask]['year'].min() + CALIBRATION_WINDOW_SIZE
    if BACKTEST_START_YEAR < start_year:
        print(f"Adjusting start year to {start_year} to allow for calibration window.")

    all_predictions = []

    for year_to_predict in tqdm(range(max(BACKTEST_START_YEAR, start_year), BACKTEST_END_YEAR + 1),
                                desc="Backtesting Years"):

        # --- Splitting Strategy ---
        # Train: History up to (Year - Window)
        # Calibration: (Year - Window) to (Year - 1)
        # Test: Year

        calib_start_year = year_to_predict - CALIBRATION_WINDOW_SIZE

        train_df = df[(df['year'] < calib_start_year) & valid_mask].copy()
        calib_df = df[(df['year'] >= calib_start_year) & (df['year'] < year_to_predict) & valid_mask].copy()
        test_df = df[(df['year'] == year_to_predict) & df[baseline_col].notna()].copy()

        if test_df.empty or train_df.empty or calib_df.empty:
            continue

        # --- Model Training (On Residuals) ---
        X_train, y_train = train_df[feature_cols], train_df[target_col]

        # Clone and fit fresh models for this fold
        model_lower = clone(model_lower_clone).fit(X_train, y_train)
        model_median = clone(model_median_clone).fit(X_train, y_train)
        model_upper = clone(model_upper_clone).fit(X_train, y_train)

        # --- Adaptive Calibration (CQR) ---
        # We calculate how badly the models missed the residuals in the calibration set
        X_calib = calib_df[feature_cols]
        y_calib_resid = calib_df[target_col]

        lower_pred_calib = model_lower.predict(X_calib)
        upper_pred_calib = model_upper.predict(X_calib)

        # CQR Score: max(lower - y, y - upper)
        # Positive score = Actual was outside the interval
        conformity_scores = np.maximum(
            lower_pred_calib - y_calib_resid,
            y_calib_resid - upper_pred_calib
        )

        # Calculate quantile adjustment (q_adj)
        # We want to cover (1 - alpha) of the calibration data
        n = len(calib_df)
        alpha = 1 - TARGET_COVERAGE
        # Finite sample correction
        quantile_to_find = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
        q_adj_fold = np.quantile(conformity_scores, quantile_to_find, method='higher')

        # --- Prediction (With Adjustment) ---
        X_test = test_df[feature_cols]
        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name', baseline_col]].copy()

        # Predict Residuals
        raw_lower_resid = model_lower.predict(X_test)
        raw_upper_resid = model_upper.predict(X_test)
        raw_median_resid = model_median.predict(X_test)

        # Reconstruct: Baseline + Predicted Residual +/- CQR Adjustment
        fold_results['predicted_yield_median'] = fold_results[baseline_col] + raw_median_resid
        fold_results['predicted_yield_lower'] = fold_results[baseline_col] + raw_lower_resid - q_adj_fold
        fold_results['predicted_yield_upper'] = fold_results[baseline_col] + raw_upper_resid + q_adj_fold

        # Clip to physical reality
        for col in ['predicted_yield_lower', 'predicted_yield_median', 'predicted_yield_upper']:
            fold_results[col] = fold_results[col].clip(lower=0)

        all_predictions.append(fold_results)

    if not all_predictions:
        return pd.DataFrame()

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield_median'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\nAdaptive CQR backtest complete.")
    return results_df


def analyze_interval_performance(results_df: pd.DataFrame):
    """Analyzes the performance of the adaptive conformalized prediction interval."""
    print(f"\n--- Analyzing Adaptive Prediction Interval Performance (Target: {TARGET_COVERAGE:.0%}) ---")
    results_df['is_covered'] = (results_df['kreisYield'] >= results_df['predicted_yield_lower']) & \
                               (results_df['kreisYield'] <= results_df['predicted_yield_upper'])

    coverage = results_df['is_covered'].mean()
    print(f"Adaptive Prediction Interval Coverage (PICP): {coverage:.2%}")

    results_df['interval_width'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']
    avg_width = results_df['interval_width'].mean()
    print(f"Mean Prediction Interval Width (MPIW): {avg_width:.2f} dt/ha")


def calculate_district_metrics(results_df: pd.DataFrame, report_dir: str):
    """Calculates R², MAE, and data count for each district."""
    print("-> Calculating district-level metrics...")

    def r2_safe(g): return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99

    performance = results_df.groupby('district_no').apply(lambda g: pd.Series(
        {'r2': r2_safe(g), 'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
         'name': g['name'].iloc[0] if 'name' in g.columns else 'Unknown',
         'data_point_count': len(g)})).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < LOW_DATA_THRESHOLD
    performance.to_csv(os.path.join(report_dir, 'district_level_metrics.csv'), index=False)
    return performance


def plot_national_average_timeline(backtest_results: pd.DataFrame, report_dir: str):
    """Generates a national average plot with the 95% adaptive CQR interval."""
    print("-> Generating National Average Prediction Timeline...")
    yearly_avg = backtest_results.groupby('year').agg(
        avg_actual_yield=('kreisYield', 'mean'),
        avg_pred_median=('predicted_yield_median', 'mean'),
        avg_pred_lower=('predicted_yield_lower', 'mean'),
        avg_pred_upper=('predicted_yield_upper', 'mean')
    ).reset_index()
    plt.figure(figsize=(14, 8))
    plt.plot(yearly_avg['year'], yearly_avg['avg_actual_yield'], label='Actual Yield', color='navy', marker='o',
             zorder=3)
    plt.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='Median Prediction', color='purple',
             linestyle='--', zorder=4)
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='purple',
                     alpha=0.2, label=f'{int(TARGET_COVERAGE * 100)}% Adaptive CQR Interval', zorder=2)
    plt.title("National Average Yield vs. Predicted Interval (Adaptive CQR Model)", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Yield (dt/ha)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(report_dir, '01_adaptive_cqr_national_average_timeline.png'), bbox_inches='tight')
    plt.close()


def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame,
                                       report_dir: str):
    """Generates timeline plots for the 3 best and 3 worst reliable districts."""
    print("-> Generating Best vs. Worst District Timelines...")
    reliable_perf = district_performance[district_performance['data_point_count'] >= MIN_DATAPOINTS_FOR_PLOT]
    if len(reliable_perf) < 6: return
    best_districts = reliable_perf.nlargest(3, 'r2')
    worst_districts = reliable_perf.nsmallest(3, 'r2')
    districts_to_plot = pd.concat([best_districts, worst_districts])
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, (_, district_info) in enumerate(districts_to_plot.iterrows()):
        ax = axes.flatten()[i]
        data = backtest_results[backtest_results['district_no'] == district_info['district_no']]
        ax.plot(data['year'], data['kreisYield'], label='Actual', color='navy', marker='o')
        ax.plot(data['year'], data['predicted_yield_median'], label='Pred', color='purple', linestyle='--')
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'], color='purple',
                        alpha=0.2)
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})")
        ax.legend()
        ax.grid(True, linestyle=':')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(report_dir, '02_adaptive_cqr_best_worst_districts.png'), bbox_inches='tight')
    plt.close()


def plot_performance_map(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame, report_dir: str):
    print("-> Generating R-squared Map...")
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    plt.title('Model Performance (R²) by District - Adaptive CQR Model', fontsize=16)
    plt.axis('off')
    plt.savefig(os.path.join(report_dir, '03_adaptive_cqr_r_squared_map.png'), bbox_inches='tight')
    plt.close()


def plot_r2_vs_data_count(district_performance: pd.DataFrame, report_dir: str):
    print("-> Generating R² vs. Data Count plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.axhline(0, color='grey', linestyle='--')
    plt.title("Data Availability vs Model Performance")
    plt.savefig(os.path.join(report_dir, '04_adaptive_cqr_r2_vs_data_count.png'), bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting Adaptive CQR Model Evaluation Pipeline ---")

    try:
        # Load Base Models (To be cloned)
        model_lower = joblib.load(LOWER_MODEL_PATH)
        model_median = joblib.load(MEDIAN_MODEL_PATH)
        model_upper = joblib.load(UPPER_MODEL_PATH)

        df = pd.read_csv(DATA_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
        print("✅ Models, data, and geo-data loaded.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return

    feature_cols = [col for col in XGB_CONFIG['FEATURE_COLS'] if col in df.columns]

    backtest_results = run_adaptive_cqr_backtest(df, feature_cols, model_lower, model_median, model_upper)
    if backtest_results.empty:
        print("❌ Backtest failed.")
        return

    # --- Reporting ---
    output_predictions_path = os.path.join(REPORT_DIR, 'full_backtest_predictions.csv')
    backtest_results.to_csv(output_predictions_path, index=False)
    print(f"-> Saved results to: {output_predictions_path}")

    analyze_interval_performance(backtest_results)
    district_performance = calculate_district_metrics(backtest_results, REPORT_DIR)

    plot_national_average_timeline(backtest_results, REPORT_DIR)
    plot_best_worst_district_timelines(district_performance, backtest_results, REPORT_DIR)
    plot_performance_map(district_performance, gdf_districts, REPORT_DIR)
    plot_r2_vs_data_count(district_performance, REPORT_DIR)

    print("\n--- Overall Performance Summary ---")
    r2_total = r2_score(backtest_results['kreisYield'], backtest_results['predicted_yield_median'])
    mae_total = backtest_results['abs_error'].mean()
    print(f"  R-squared (R²): {r2_total:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_total:.2f} dt/ha")
    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main()