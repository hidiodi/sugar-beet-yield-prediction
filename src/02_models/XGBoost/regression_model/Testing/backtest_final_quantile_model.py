# File: src/models/backtest_final_quantile_model.py
# FINAL VERSION: Implements Simple CQR with a sequential split to provide a
#                robust, final evaluation of the base models.
# Refactored to use central configuration from src.config

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

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Use the config dictionaries from the central config file
XGB_CONFIG = config.XGBOOST_TRAINING_CONFIG
BACKTEST_CONFIG = config.BACKTESTING_CONFIG

def run_backtest_with_cqr(df: pd.DataFrame, feature_cols: list, model_lower_clone: XGBRegressor,
                          model_median_clone: XGBRegressor, model_upper_clone: XGBRegressor):
    print(f"\n--- Starting Backtest with Simple CQR from {BACKTEST_CONFIG['BACKTEST_START_YEAR']} to {BACKTEST_CONFIG['BACKTEST_END_YEAR']} ---")
    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1), desc="Backtesting Years"):
        train_df_full = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()
        if test_df.empty or len(train_df_full) < 20: continue

        calib_size = int(len(train_df_full) * BACKTEST_CONFIG['CALIBRATION_SET_SIZE'])
        if calib_size < 10: continue
        train_df = train_df_full.iloc[:-calib_size]
        calib_df = train_df_full.iloc[-calib_size:]

        X_train, y_train_residual = train_df[feature_cols], train_df['forecast_residual']
        X_calib, y_calib_residual = calib_df[feature_cols], calib_df['forecast_residual']
        X_test = test_df[feature_cols]

        model_lower, model_median, model_upper = clone(model_lower_clone), clone(model_median_clone), clone(model_upper_clone)
        model_lower.fit(X_train, y_train_residual)
        model_median.fit(X_train, y_train_residual)
        model_upper.fit(X_train, y_train_residual)

        calib_pred_lower = model_lower.predict(X_calib)
        calib_pred_upper = model_upper.predict(X_calib)
        conformity_scores = np.maximum(y_calib_residual - calib_pred_upper, calib_pred_lower - y_calib_residual)
        q_hat = np.quantile(conformity_scores, BACKTEST_CONFIG['NOMINAL_COVERAGE'], interpolation='higher')

        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name', 'stage1_forecast']].copy()
        predicted_residual_lower = model_lower.predict(X_test)
        predicted_residual_median = model_median.predict(X_test)
        predicted_residual_upper = model_upper.predict(X_test)

        fold_results['predicted_yield_lower'] = test_df['stage1_forecast'] + predicted_residual_lower - q_hat
        fold_results['predicted_yield_median'] = test_df['stage1_forecast'] + predicted_residual_median
        fold_results['predicted_yield_upper'] = test_df['stage1_forecast'] + predicted_residual_upper + q_hat
        all_predictions.append(fold_results)

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield_median'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\nBacktest complete.")
    return results_df

def analyze_interval_performance(results_df: pd.DataFrame):
    print(f"\n--- Analyzing Prediction Interval Performance (Target: {BACKTEST_CONFIG['NOMINAL_COVERAGE']:.0%}) ---")
    results_df['is_covered'] = (results_df['kreisYield'] >= results_df['predicted_yield_lower']) & \
                               (results_df['kreisYield'] <= results_df['predicted_yield_upper'])
    coverage = results_df['is_covered'].mean()
    print(f"Prediction Interval Coverage (PICP): {coverage:.2%}")
    results_df['interval_width'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']
    avg_width = results_df['interval_width'].mean()
    print(f"Mean Prediction Interval Width (MPIW): {avg_width:.2f} dt/ha")

def calculate_district_metrics(results_df: pd.DataFrame, report_dir: str):
    print("-> Calculating district-level metrics...")
    def r2_safe(g): return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99
    performance = results_df.groupby('district_no').apply(lambda g: pd.Series(
        {'r2': r2_safe(g), 'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
         'name': g['name'].iloc[0], 'data_point_count': len(g)})).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < BACKTEST_CONFIG['LOW_DATA_THRESHOLD']
    performance.to_csv(os.path.join(report_dir, 'district_level_metrics.csv'), index=False)
    return performance

def plot_national_average_timeline(backtest_results: pd.DataFrame, report_dir: str):
    print("-> Generating National Average Prediction Timeline...")
    yearly_avg = backtest_results.groupby('year').agg(
        avg_actual_yield=('kreisYield', 'mean'), avg_pred_median=('predicted_yield_median', 'mean'),
        avg_pred_lower=('predicted_yield_lower', 'mean'), avg_pred_upper=('predicted_yield_upper', 'mean')
    ).reset_index()
    plt.figure(figsize=(14, 8)); plt.plot(yearly_avg['year'], yearly_avg['avg_actual_yield'], label='National Average Actual Yield', color='navy', marker='o', zorder=3)
    plt.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='Median Prediction', color='red', linestyle='--', zorder=4)
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='red', alpha=0.2, label='95% Prediction Interval', zorder=2)
    plt.title("National Average Yield vs. Predicted Interval (Final Quantile Model)", fontsize=16); plt.xlabel("Year"); plt.ylabel("Yield (dt/ha)")
    plt.legend(); plt.grid(True, linestyle=':'); plt.savefig(os.path.join(report_dir, '01_national_average_timeline.png'), bbox_inches='tight'); plt.close()

def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame, report_dir: str):
    print("-> Generating Best vs. Worst District Timelines...")
    reliable_perf = district_performance[district_performance['data_point_count'] >= BACKTEST_CONFIG['MIN_DATAPOINTS_FOR_PLOT']]
    if len(reliable_perf) < 6: print(f"   Warning: Not enough reliable districts (found {len(reliable_perf)}) to plot. Skipping."); return
    best_districts, worst_districts = reliable_perf.nlargest(3, 'r2'), reliable_perf.nsmallest(3, 'r2')
    districts_to_plot = pd.concat([best_districts, worst_districts])
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, (_, district_info) in enumerate(districts_to_plot.iterrows()):
        ax = axes.flatten()[i]; data = backtest_results[backtest_results['district_no'] == district_info['district_no']]
        ax.plot(data['year'], data['kreisYield'], label='Actual Yield', color='navy', marker='o', markersize=4, zorder=3)
        ax.plot(data['year'], data['predicted_yield_median'], label='Median Prediction', color='red', linestyle='--', zorder=4)
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'], color='red', alpha=0.2, label='95% Interval')
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})"); ax.legend(); ax.grid(True, linestyle=':')
    plt.suptitle(f"Prediction Timelines for Best & Worst Districts (Final Quantile Model)", fontsize=18); plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(report_dir, '02_best_worst_districts.png'), bbox_inches='tight'); plt.close()

def plot_performance_map(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame, report_dir: str):
    print("-> Generating R-squared Map..."); merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12)); merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8', legend=True, legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"}, missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'] == True]
    if not low_data_gdf.empty: low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black', label=f'Low Data (< {BACKTEST_CONFIG["LOW_DATA_THRESHOLD"]} years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability'); ax.set_title('Model Performance (R²) by District - Final Quantile Model', fontsize=16); ax.set_axis_off(); plt.savefig(os.path.join(report_dir, '03_r_squared_map.png'), bbox_inches='tight'); plt.close()

def plot_r2_vs_data_count(district_performance: pd.DataFrame, report_dir: str):
    print("-> Generating R² vs. Data Count plot..."); plt.figure(figsize=(10, 6)); sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data', palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1); plt.axhline(0, color='grey', linestyle='--'); plt.title("Data Availability vs Model Performance - Final Quantile Model")
    plt.xlabel("Number of Years in Backtest per District"); plt.ylabel("R-squared (R²)"); plt.legend(title='Is Low Data?'); plt.savefig(os.path.join(report_dir, '04_r2_vs_data_count.png'), bbox_inches='tight'); plt.close()

def main():
    report_dir = BACKTEST_CONFIG['REPORT_DIR']
    os.makedirs(report_dir, exist_ok=True)
    print("--- Starting Final Quantile Champion Model Evaluation Pipeline ---")
    try:
        model_lower, model_median, model_upper = joblib.load(XGB_CONFIG['LOWER_MODEL_PATH']), joblib.load(
            XGB_CONFIG['MEDIAN_MODEL_PATH']), joblib.load(XGB_CONFIG['UPPER_MODEL_PATH'])
        df = pd.read_csv(XGB_CONFIG['DATA_PATH'])
        gdf_districts = gpd.read_file(BACKTEST_CONFIG['GEOJSON_PATH'])
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
        print("Models, data, and geo-data loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}"); return

    print("\n--- Preparing target variable (Forecast Residuals) ---")
    df.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df['forecast_residual'] = df['kreisYield'] - df['stage1_forecast']
    df.dropna(subset=['stage1_forecast', 'forecast_residual'], inplace=True)
    print(" -> Target variable defined.")

    feature_cols = XGB_CONFIG['FEATURE_COLS']

    backtest_results = run_backtest_with_cqr(df, feature_cols, model_lower, model_median, model_upper)
    if backtest_results.empty:
        print("❌ Backtest did not produce results. Terminating."); return

    backtest_csv_path = os.path.join(report_dir, 'full_backtest_predictions.csv')
    backtest_results.to_csv(backtest_csv_path, index=False)
    print(f"\n✅ Full backtest results saved to {backtest_csv_path}")

    analyze_interval_performance(backtest_results)
    district_performance = calculate_district_metrics(backtest_results, str(report_dir))
    plot_national_average_timeline(backtest_results, str(report_dir))
    plot_best_worst_district_timelines(district_performance, backtest_results, str(report_dir))
    plot_performance_map(district_performance, gdf_districts, str(report_dir))
    plot_r2_vs_data_count(district_performance, str(report_dir))

    print("\n--- Overall Performance Summary (All Districts, Median Prediction) ---")
    r2_total = r2_score(backtest_results['kreisYield'], backtest_results['predicted_yield_median'])
    mae_total = backtest_results['abs_error'].mean()
    print(f"  R-squared (R²): {r2_total:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_total:.2f} dt/ha")

    reliable_districts = district_performance[district_performance['data_point_count'] >= BACKTEST_CONFIG['LOW_DATA_THRESHOLD']]
    if not reliable_districts.empty:
        reliable_backtest_results = backtest_results[backtest_results['district_no'].isin(reliable_districts['district_no'])]
        r2_reliable = r2_score(reliable_backtest_results['kreisYield'], reliable_backtest_results['predicted_yield_median'])
        mae_reliable = reliable_backtest_results['abs_error'].mean()
        print(f"\n--- Performance Summary for Reliable Districts (>={BACKTEST_CONFIG['LOW_DATA_THRESHOLD']} data points) ---")
        print(f"  Number of Reliable Districts: {len(reliable_districts)}")
        print(f"  R-squared (R²): {r2_reliable:.4f}")
        print(f"  Mean Absolute Error (MAE): {mae_reliable:.2f} dt/ha")

    print("\n--- Evaluation Complete ---")
    print(f"All reports and plots saved in: {report_dir}")

if __name__ == "__main__":
    main()
