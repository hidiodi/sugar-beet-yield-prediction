# File: src/02_models/XGBoost/regression_model/Testing/backtest_standalone_xgb_model.py
# Description: (FINAL, CORRECTED VERSION 2) Backtests the detrended standalone model.
#              Fixes the KeyError from the complex .transform() call by using a
#              simpler and more robust .apply() method for detrending.

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
from sklearn.linear_model import LinearRegression
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
XGB_CONFIG = config.STANDALONE_XGB_CONFIG
BACKTEST_CONFIG = config.STANDALONE_BACKTESTING_CONFIG


# ==============================================================================
# SELF-CONTAINED ANALYSIS AND PLOTTING FUNCTIONS
# ==============================================================================

def analyze_interval_performance(results_df: pd.DataFrame):
    """Analyzes and prints key interval performance metrics."""
    print(f"\n--- Analyzing Prediction Interval Performance (Target: {BACKTEST_CONFIG['NOMINAL_COVERAGE']:.0%}) ---")
    results_df['is_covered'] = (results_df['kreisYield'] >= results_df['predicted_yield_lower']) & \
                               (results_df['kreisYield'] <= results_df['predicted_yield_upper'])
    coverage = results_df['is_covered'].mean()
    print(f"Prediction Interval Coverage (PICP): {coverage:.2%}")
    results_df['interval_width'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']
    avg_width = results_df['interval_width'].mean()
    print(f"Mean Prediction Interval Width (MPIW): {avg_width:.2f} dt/ha")


def calculate_district_metrics(results_df: pd.DataFrame, report_dir: str):
    """Calculates and saves district-level performance metrics."""
    print("-> Calculating district-level metrics...")

    def r2_safe(g): return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99

    performance = results_df.groupby('district_no').apply(lambda g: pd.Series({
        'r2': r2_safe(g),
        'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
        'name': g['name'].iloc[0],
        'data_point_count': len(g)
    })).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < BACKTEST_CONFIG['LOW_DATA_THRESHOLD']
    performance.to_csv(os.path.join(report_dir, 'district_level_metrics.csv'), index=False)
    return performance


def plot_national_average_timeline(backtest_results: pd.DataFrame, report_dir: str):
    """Generates the national average yield vs. prediction plot."""
    print("-> Generating National Average Prediction Timeline...")
    yearly_avg = backtest_results.groupby('year').agg(
        avg_actual_yield=('kreisYield', 'mean'),
        avg_pred_median=('predicted_yield_median', 'mean'),
        avg_pred_lower=('predicted_yield_lower', 'mean'),
        avg_pred_upper=('predicted_yield_upper', 'mean')
    ).reset_index()
    plt.figure(figsize=(14, 8))
    plt.plot(yearly_avg['year'], yearly_avg['avg_actual_yield'], label='National Average Actual Yield', color='navy',
             marker='o', zorder=3)
    plt.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='Median Prediction', color='red', linestyle='--',
             zorder=4)
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='red',
                     alpha=0.2, label='95% Prediction Interval', zorder=2)
    plt.title("National Average Yield vs. Predicted Interval (Detrended Standalone XGB)", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Yield (dt/ha)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(report_dir, '01_national_average_timeline.png'), bbox_inches='tight')
    plt.close()


def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame,
                                       report_dir: str):
    # This function and the others below are identical to the previous version and remain correct.
    print("-> Generating Best vs. Worst District Timelines...")
    reliable_perf = district_performance[
        district_performance['data_point_count'] >= BACKTEST_CONFIG['MIN_DATAPOINTS_FOR_PLOT']]
    if len(reliable_perf) < 6:
        print(f"   Warning: Not enough reliable districts to plot. Skipping.")
        return
    best_districts, worst_districts = reliable_perf.nlargest(3, 'r2'), reliable_perf.nsmallest(3, 'r2')
    districts_to_plot = pd.concat([best_districts, worst_districts])
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, (_, district_info) in enumerate(districts_to_plot.iterrows()):
        ax = axes.flatten()[i]
        data = backtest_results[backtest_results['district_no'] == district_info['district_no']]
        ax.plot(data['year'], data['kreisYield'], label='Actual Yield', color='navy', marker='o', markersize=4,
                zorder=3)
        ax.plot(data['year'], data['predicted_yield_median'], label='Median Prediction', color='red', linestyle='--',
                zorder=4)
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'], color='red',
                        alpha=0.2, label='95% Interval')
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})")
        ax.legend()
        ax.grid(True, linestyle=':')
    plt.suptitle("Prediction Timelines for Best & Worst Districts (Detrended Standalone XGB)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(report_dir, '02_best_worst_districts.png'), bbox_inches='tight')
    plt.close()


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
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black',
                                 label=f'Low Data (< {BACKTEST_CONFIG["LOW_DATA_THRESHOLD"]} years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability')
    ax.set_title('Model Performance (R²) by District - Detrended Standalone XGB', fontsize=16)
    ax.set_axis_off()
    plt.savefig(os.path.join(report_dir, '03_r_squared_map.png'), bbox_inches='tight')
    plt.close()


def plot_r2_vs_data_count(district_performance: pd.DataFrame, report_dir: str):
    print("-> Generating R² vs. Data Count plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='grey', linestyle='--')
    plt.title("Data Availability vs Model Performance - Detrended Standalone XGB")
    plt.xlabel("Number of Years in Backtest per District")
    plt.ylabel("R-squared (R²)")
    plt.legend(title='Is Low Data?')
    plt.savefig(os.path.join(report_dir, '04_r2_vs_data_count.png'), bbox_inches='tight')
    plt.close()


# ==============================================================================
# CORE BACKTESTING LOGIC
# ==============================================================================

def run_backtest_detrended(df: pd.DataFrame, models: dict, feature_list: list) -> pd.DataFrame:
    """
    Runs a time-series backtest with on-the-fly detrending.
    """
    print(f"\n--- Starting Detrended Standalone Model Backtest ---")
    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1),
                                desc="Backtesting Years"):
        train_df_full = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()
        if test_df.empty or len(train_df_full) < 20:
            continue

        # --- SIMPLIFIED AND CORRECTED DETRENDING LOGIC ---

        # 1. Create a function to apply to each district group in the training data
        def get_detrended_series(group):
            if len(group) >= 5 and not group['kreisYield'].isnull().any():
                lr = LinearRegression().fit(group[['year']], group['kreisYield'])
                trend = lr.predict(group[['year']])
                return group['kreisYield'] - trend
            else:
                return pd.Series(np.nan, index=group.index)

        # 2. Apply this function to create the detrended target for the entire training set
        train_df_full['yield_detrended'] = train_df_full.groupby('district_no').apply(get_detrended_series).reset_index(
            level=0, drop=True)

        # 3. Calculate the trend for the test year separately for each district
        test_trends = {}
        for dist_id, group in train_df_full.groupby('district_no'):
            if len(group) >= 5 and not group['kreisYield'].isnull().any():
                lr = LinearRegression().fit(group[['year']], group['kreisYield'])
                test_trends[dist_id] = lr.predict(pd.DataFrame({'year': [year_to_predict]}))[0]
        test_df['yield_trend_backtest'] = test_df['district_no'].map(test_trends)

        # 4. Clean data for this fold
        train_df_full.dropna(subset=['yield_detrended'], inplace=True)
        test_df.dropna(subset=['yield_trend_backtest'], inplace=True)
        if test_df.empty or train_df_full.empty:
            continue

        X_train = train_df_full[feature_list]
        y_train = train_df_full['yield_detrended']
        X_test = test_df[feature_list]

        # Fit models, predict, and re-add the trend
        fitted_models = {name: clone(model).fit(X_train, y_train) for name, model in models.items()}
        preds = {name: model.predict(X_test) for name, model in fitted_models.items()}

        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name']].copy()
        fold_results['predicted_yield_median'] = preds['median'] + test_df['yield_trend_backtest']
        fold_results['predicted_yield_lower'] = preds['lower'] + test_df['yield_trend_backtest']
        fold_results['predicted_yield_upper'] = preds['upper'] + test_df['yield_trend_backtest']
        all_predictions.append(fold_results)

    if not all_predictions:
        return pd.DataFrame()

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield_median'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\nBacktest complete.")
    return results_df


def main():
    """Main execution function."""
    report_dir = Path(BACKTEST_CONFIG['REPORT_DIR'])
    report_dir.mkdir(parents=True, exist_ok=True)
    print("--- Starting Detrended Standalone Model Evaluation ---")

    try:
        models = {name: joblib.load(XGB_CONFIG[f'{name.upper()}_MODEL_PATH']) for name in ['lower', 'median', 'upper']}
        df = pd.read_csv(config.STANDALONE_XGB_CONFIG['DATA_PATH'])
        gdf = gpd.read_file(BACKTEST_CONFIG['GEOJSON_PATH'])

        gdf.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf['district_no'] = gdf['district_no'].astype(str).str.zfill(5)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf[['district_no', 'name']], on='district_no', how='left')
        print("Models and data loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}")
        return

    feature_list = [col for col in XGB_CONFIG['FEATURE_COLS'] if
                    col in df.columns and col not in ['kreisYield', 'yield_detrended', 'yield_trend']]

    results = run_backtest_detrended(df, models, feature_list)

    if results.empty:
        print("❌ Backtest did not produce results. Terminating.")
        return

    results_path = report_dir / 'full_backtest_predictions.csv'
    results.to_csv(results_path, index=False)
    print(f"\n✅ Full backtest results saved to {results_path}")

    analyze_interval_performance(results)
    perf = calculate_district_metrics(results, str(report_dir))

    plot_national_average_timeline(results, str(report_dir))
    plot_best_worst_district_timelines(perf, results, str(report_dir))
    plot_performance_map(perf, gdf, str(report_dir))
    plot_r2_vs_data_count(perf, str(report_dir))

    print("\n--- Overall Performance Summary ---")
    print(f"  R-squared (R²): {r2_score(results['kreisYield'], results['predicted_yield_median']):.4f}")
    print(f"  Mean Absolute Error (MAE): {results['abs_error'].mean():.2f} dt/ha")
    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main()