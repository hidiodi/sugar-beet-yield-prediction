# File: src/models/backtest_final_ngboost_model.py
# Description: The definitive backtesting script for the NGBoost model,
#              refactored to use the central config and a robust, leak-proof feature set.

import pandas as pd
import geopandas as gpd
import joblib
from ngboost import NGBRegressor
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

# --- START OF REFACTOR ---

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3] # Adjust path depth as needed
sys.path.insert(0, str(project_root))

from src import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Use the config dictionaries from the central config file
XGB_CONFIG = config.XGBOOST_TRAINING_CONFIG # NGBoost shares data/features with XGBoost
BACKTEST_CONFIG = config.BACKTESTING_CONFIG # NGBoost uses the same backtest settings

# Define paths using the central config
MODEL_PATH = XGB_CONFIG['MODEL_OUTPUT_DIR'] / 'final_ngboost_model.joblib'
REPORT_DIR = Path('reports/figures/district_level_diagnostics/final_ngboost_champion')

# --- END OF REFACTOR ---


def run_backtest_with_ngboost(df: pd.DataFrame, full_feature_list: list, model_clone: NGBRegressor):
    """Performs a rolling forecast for the NGBoost model with proper validation."""
    print(f"\n--- Starting Backtest with NGBoost from {BACKTEST_CONFIG['BACKTEST_START_YEAR']} to {BACKTEST_CONFIG['BACKTEST_END_YEAR']} ---")
    all_predictions = []

    # --- START OF FIX: Create a clean list of predictors ---
    wofost_col = 'wofost_forecast_yield_fresh_dt'
    cols_to_exclude = [wofost_col, 'stage1_forecast', 'kreisYield']
    actual_training_features = [col for col in full_feature_list if col in df.columns and col not in cols_to_exclude]
    # --- END OF FIX ---

    for year_to_predict in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1), desc="Backtesting Years"):
        full_train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()
        if test_df.empty or len(full_train_df) < 20: continue

        train_end_idx = int(len(full_train_df) * 0.85)
        train_df = full_train_df[:train_end_idx]
        val_df = full_train_df[train_end_idx:]
        if len(train_df) < 10 or len(val_df) < 10: continue

        X_train, y_train = train_df[actual_training_features], train_df['forecast_residual']
        X_val, y_val = val_df[actual_training_features], val_df['forecast_residual']
        X_test = test_df[actual_training_features]

        ngb_model = clone(model_clone)
        ngb_model.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=20)

        pred_dist = ngb_model.pred_dist(X_test)

        lower_bounds = pred_dist.ppf(0.5 - BACKTEST_CONFIG['NOMINAL_COVERAGE'] / 2)
        upper_bounds = pred_dist.ppf(0.5 + BACKTEST_CONFIG['NOMINAL_COVERAGE'] / 2)
        median_preds = pred_dist.ppf(0.5)

        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name', wofost_col]].copy()
        fold_results['predicted_yield_lower'] = test_df[wofost_col] + lower_bounds
        fold_results['predicted_yield_median'] = test_df[wofost_col] + median_preds
        fold_results['predicted_yield_upper'] = test_df[wofost_col] + upper_bounds
        all_predictions.append(fold_results)

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield_median'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\nBacktest complete.")
    return results_df


# ... All plotting and analysis functions are unchanged ...
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
    def r2_safe(g):
        return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99
    performance = results_df.groupby('district_no').apply(lambda g: pd.Series(
        {'r2': r2_safe(g), 'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
         'name': g['name'].iloc[0], 'data_point_count': len(g)})).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < BACKTEST_CONFIG['LOW_DATA_THRESHOLD']
    performance.to_csv(os.path.join(report_dir, 'district_level_metrics.csv'), index=False)
    return performance

def plot_national_average_timeline(backtest_results: pd.DataFrame, report_dir: str):
    print("-> Generating National Average Prediction Timeline...")
    yearly_avg = backtest_results.groupby('year').agg(
        avg_actual_yield=('kreisYield', 'mean'),
        avg_pred_median=('predicted_yield_median', 'mean'),
        avg_pred_lower=('predicted_yield_lower', 'mean'),
        avg_pred_upper=('predicted_yield_upper', 'mean')
    ).reset_index()
    plt.figure(figsize=(14, 8))
    plt.plot(yearly_avg['year'], yearly_avg['avg_actual_yield'], label='National Average Actual Yield', color='navy', marker='o', zorder=3)
    plt.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='Median Prediction', color='red', linestyle='--', zorder=4)
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='red', alpha=0.2, label='95% Prediction Interval', zorder=2)
    plt.title("National Average Yield vs. Predicted Interval (NGBoost Model)", fontsize=16)
    plt.xlabel("Year"); plt.ylabel("Yield (dt/ha)"); plt.legend(); plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(report_dir, '01_national_average_timeline.png'), bbox_inches='tight'); plt.close()

def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame, report_dir: str):
    print("-> Generating Best vs. Worst District Timelines...")
    reliable_perf = district_performance[district_performance['data_point_count'] >= BACKTEST_CONFIG['MIN_DATAPOINTS_FOR_PLOT']]
    if len(reliable_perf) < 6:
        print(f"   Warning: Not enough reliable districts (found {len(reliable_perf)}) to plot. Skipping."); return
    best_districts = reliable_perf.nlargest(3, 'r2'); worst_districts = reliable_perf.nsmallest(3, 'r2')
    districts_to_plot = pd.concat([best_districts, worst_districts])
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, (_, district_info) in enumerate(districts_to_plot.iterrows()):
        ax = axes.flatten()[i]
        data = backtest_results[backtest_results['district_no'] == district_info['district_no']]
        ax.plot(data['year'], data['kreisYield'], label='Actual Yield', color='navy', marker='o', markersize=4, zorder=3)
        ax.plot(data['year'], data['predicted_yield_median'], label='Median Prediction', color='red', linestyle='--', zorder=4)
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'], color='red', alpha=0.2, label='95% Interval')
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})")
        ax.legend(); ax.grid(True, linestyle=':')
    plt.suptitle(f"Prediction Timelines for Best & Worst Districts (NGBoost Model)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(report_dir, '02_best_worst_districts.png'), bbox_inches='tight'); plt.close()

def plot_performance_map(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame, report_dir: str):
    print("-> Generating R-squared Map...")
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8', legend=True,
                    legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'] == True]
    if not low_data_gdf.empty:
        low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black', label=f'Low Data (< {BACKTEST_CONFIG["LOW_DATA_THRESHOLD"]} years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability')
    ax.set_title('Model Performance (R²) by District - NGBoost Model', fontsize=16); ax.set_axis_off()
    plt.savefig(os.path.join(report_dir, '03_r_squared_map.png'), bbox_inches='tight'); plt.close()

def plot_r2_vs_data_count(district_performance: pd.DataFrame, report_dir: str):
    print("-> Generating R² vs. Data Count plot...")
    plt.figure(figsize=(10, 6)); sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data', palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1); plt.axhline(0, color='grey', linestyle='--'); plt.title("Data Availability vs Model Performance - NGBoost Model")
    plt.xlabel("Number of Years in Backtest per District"); plt.ylabel("R-squared (R²)"); plt.legend(title='Is Low Data?')
    plt.savefig(os.path.join(report_dir, '04_r2_vs_data_count.png'), bbox_inches='tight'); plt.close()


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting NGBoost Model Evaluation Pipeline ---")
    try:
        ngb_model = joblib.load(MODEL_PATH)
        df = pd.read_csv(XGB_CONFIG['DATA_PATH']) # Use config path
        gdf_districts = gpd.read_file(BACKTEST_CONFIG['GEOJSON_PATH']) # Use config path
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
        print("Model, data, and geo-data loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}"); return

    print("\n--- Preparing target variable (Forecast Residuals) ---")
    wofost_col = 'wofost_forecast_yield_fresh_dt'
    df['forecast_residual'] = df['kreisYield'] - df[wofost_col]
    df.dropna(subset=[wofost_col, 'forecast_residual', 'kreisYield'], inplace=True)
    print(" -> Target variable defined.")

    feature_cols_from_config = XGB_CONFIG['FEATURE_COLS'] # Use config feature list

    backtest_results = run_backtest_with_ngboost(df, feature_cols_from_config, ngb_model)
    if backtest_results.empty:
        print("❌ Backtest did not produce results. Terminating."); return

    backtest_csv_path = os.path.join(REPORT_DIR, 'full_backtest_predictions.csv')
    backtest_results.to_csv(backtest_csv_path, index=False)
    print(f"\n✅ Full backtest results saved to {backtest_csv_path}")

    analyze_interval_performance(backtest_results)
    district_performance = calculate_district_metrics(backtest_results, str(REPORT_DIR))
    plot_national_average_timeline(backtest_results, str(REPORT_DIR))
    plot_best_worst_district_timelines(district_performance, backtest_results, str(REPORT_DIR))
    plot_performance_map(district_performance, gdf_districts, str(REPORT_DIR))
    plot_r2_vs_data_count(district_performance, str(REPORT_DIR))

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
    print(f"All reports and plots saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()