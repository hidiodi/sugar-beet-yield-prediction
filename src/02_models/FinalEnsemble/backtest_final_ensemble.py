# File: src/03_analysis/backtest_final_ensemble.py
# Description: This script constructs and evaluates the final champion model: the Hybrid Ensemble.
# Refactored to use central configuration from src.config

import pandas as pd
import geopandas as gpd
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import sys

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Use the ENSEMBLE_BACKTESTING_CONFIG dictionary from the central config file
CONFIG = config.ENSEMBLE_BACKTESTING_CONFIG


def analyze_interval_performance(results_df: pd.DataFrame):
    print(f"\n--- Analyzing Prediction Interval Performance (Target: {CONFIG['NOMINAL_COVERAGE']:.0%}) ---")
    results_df['is_covered'] = (results_df['kreisYield'] >= results_df['predicted_yield_lower']) & \
                               (results_df['kreisYield'] <= results_df['predicted_yield_upper'])
    coverage = results_df['is_covered'].mean()
    print(f"Prediction Interval Coverage (PICP): {coverage:.2%}")
    results_df['interval_width'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']
    avg_width = results_df['interval_width'].mean()
    print(f"Mean Prediction Interval Width (MPIW): {avg_width:.2f} dt/ha")


def calculate_district_metrics(results_df: pd.DataFrame, report_dir: Path):
    print("-> Calculating district-level metrics...")

    def r2_safe(g): return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99

    performance = results_df.groupby('district_no').apply(lambda g: pd.Series(
        {'r2': r2_safe(g), 'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
         'name': g['name'].iloc[0], 'data_point_count': len(g)})).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < CONFIG['LOW_DATA_THRESHOLD']
    performance.to_csv(report_dir / 'district_level_metrics.csv', index=False)
    return performance


def plot_national_average_timeline(backtest_results: pd.DataFrame, report_dir: Path):
    print("-> Generating National Average Prediction Timeline...")
    yearly_avg = backtest_results.groupby('year').agg(avg_actual_yield=('kreisYield', 'mean'),
                                                      avg_pred_median=('predicted_yield_median', 'mean'),
                                                      avg_pred_lower=('predicted_yield_lower', 'mean'),
                                                      avg_pred_upper=('predicted_yield_upper', 'mean')).reset_index()
    plt.figure(figsize=(14, 8));
    plt.plot(yearly_avg['year'], yearly_avg['avg_actual_yield'], label='National Average Actual Yield', color='navy',
             marker='o', zorder=3);
    plt.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='Median Prediction', color='purple',
             linestyle='--', zorder=4);
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='purple',
                     alpha=0.2, label='95% Prediction Interval', zorder=2)
    plt.title("National Average Yield vs. Predicted Interval (Final Ensemble Model)", fontsize=16);
    plt.xlabel("Year");
    plt.ylabel("Yield (dt/ha)");
    plt.legend();
    plt.grid(True, linestyle=':');
    plt.savefig(report_dir / '01_national_average_timeline.png', bbox_inches='tight');
    plt.close()


def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame,
                                       report_dir: Path):
    print("-> Generating Best vs. Worst District Timelines...");
    reliable_perf = district_performance[district_performance['data_point_count'] >= CONFIG['MIN_DATAPOINTS_FOR_PLOT']]
    if len(reliable_perf) < 6: print(f"   Warning: Not enough reliable districts to plot. Skipping."); return
    best_districts, worst_districts = reliable_perf.nlargest(3, 'r2'), reliable_perf.nsmallest(3, 'r2');
    districts_to_plot = pd.concat([best_districts, worst_districts])
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, (_, district_info) in enumerate(districts_to_plot.iterrows()):
        ax = axes.flatten()[i];
        data = backtest_results[backtest_results['district_no'] == district_info['district_no']];
        ax.plot(data['year'], data['kreisYield'], label='Actual Yield', color='navy', marker='o', markersize=4,
                zorder=3);
        ax.plot(data['year'], data['predicted_yield_median'], label='Median Prediction', color='purple', linestyle='--',
                zorder=4);
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'], color='purple',
                        alpha=0.2, label='95% Interval')
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})");
        ax.legend();
        ax.grid(True, linestyle=':')
    plt.suptitle(f"Prediction Timelines for Best & Worst Districts (Final Ensemble Model)", fontsize=18);
    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    plt.savefig(report_dir / '02_best_worst_districts.png', bbox_inches='tight');
    plt.close()


def plot_performance_map(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame, report_dir: Path):
    print("-> Generating R-squared Map...")

    district_performance['district_no'] = district_performance['district_no'].astype(str).str.zfill(5)

    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12));
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8', legend=True,
                    legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'] == True]
    if not low_data_gdf.empty: low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black',
                                 label=f'Low Data (< {CONFIG["LOW_DATA_THRESHOLD"]} years)');
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability');
    ax.set_title('Model Performance (R²) by District - Final Ensemble Model', fontsize=16);
    ax.set_axis_off();
    plt.savefig(report_dir / '03_r_squared_map.png', bbox_inches='tight');
    plt.close()


def plot_r2_vs_data_count(district_performance: pd.DataFrame, report_dir: Path):
    print("-> Generating R² vs. Data Count plot...");
    plt.figure(figsize=(10, 6));
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1);
    plt.axhline(0, color='grey', linestyle='--');
    plt.title("Data Availability vs Model Performance - Final Ensemble Model");
    plt.xlabel("Number of Years in Backtest per District");
    plt.ylabel("R-squared (R²)");
    plt.legend(title='Is Low Data?');
    plt.savefig(report_dir / '04_r2_vs_data_count.png', bbox_inches='tight');
    plt.close()


def main():
    report_dir = CONFIG['REPORT_DIR']
    report_dir.mkdir(parents=True, exist_ok=True)
    print("--- Starting Final Ensemble Champion Model Evaluation Pipeline ---")

    try:
        df_xgb = pd.read_csv(CONFIG['HYBRID_XGB_INPUT_FILE'])
        df_cqr = pd.read_csv(CONFIG['ADAPTIVE_CQR_INPUT_FILE'])
        gdf_districts = gpd.read_file(CONFIG['GEOJSON_PATH'])
        gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        print("Model component data and geo-data loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}");
        return

    df_xgb['district_no'] = df_xgb['district_no'].astype(str).str.zfill(5)
    df_cqr['district_no'] = df_cqr['district_no'].astype(str).str.zfill(5)

    print("\n--- Constructing the Hybrid Ensemble ---")
    df_merged = pd.merge(
        df_xgb[['district_no', 'year', 'kreisYield', 'name', 'predicted_yield_median']],
        df_cqr[['district_no', 'year', 'predicted_yield_median', 'predicted_yield_lower', 'predicted_yield_upper']],
        on=['district_no', 'year'],
        suffixes=('_xgb', '_cqr')
    )

    delta_upper = df_merged['predicted_yield_upper'] - df_merged['predicted_yield_median_cqr']
    delta_lower = df_merged['predicted_yield_median_cqr'] - df_merged['predicted_yield_lower']

    ensemble_df = df_merged[['district_no', 'year', 'kreisYield', 'name']].copy()
    ensemble_df['predicted_yield_median'] = df_merged['predicted_yield_median_xgb']
    ensemble_df['predicted_yield_lower'] = ensemble_df['predicted_yield_median'] - delta_lower
    ensemble_df['predicted_yield_upper'] = ensemble_df['predicted_yield_median'] + delta_upper

    ensemble_df['error'] = ensemble_df['predicted_yield_median'] - ensemble_df['kreisYield']
    ensemble_df['abs_error'] = ensemble_df['error'].abs()

    print("✓ Ensemble constructed.")

    backtest_csv_path = report_dir / 'full_backtest_predictions.csv'
    ensemble_df.to_csv(backtest_csv_path, index=False)
    print(f"\n✅ Full backtest results for Champion Ensemble saved to {backtest_csv_path}")

    analyze_interval_performance(ensemble_df)
    district_performance = calculate_district_metrics(ensemble_df, report_dir)

    plot_national_average_timeline(ensemble_df, report_dir)
    plot_best_worst_district_timelines(district_performance, ensemble_df, report_dir)
    plot_performance_map(district_performance, gdf_districts, report_dir)
    plot_r2_vs_data_count(district_performance, report_dir)

    print("\n--- Overall Performance Summary (All Districts, Median Prediction) ---")
    r2_total = r2_score(ensemble_df['kreisYield'], ensemble_df['predicted_yield_median'])
    mae_total = ensemble_df['abs_error'].mean()
    print(f"  R-squared (R²): {r2_total:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_total:.2f} dt/ha")

    reliable_districts = district_performance[district_performance['data_point_count'] >= CONFIG['LOW_DATA_THRESHOLD']]
    if not reliable_districts.empty:
        reliable_results = ensemble_df[ensemble_df['district_no'].isin(reliable_districts['district_no'])]
        r2_reliable = r2_score(reliable_results['kreisYield'], reliable_results['predicted_yield_median'])
        mae_reliable = reliable_results['abs_error'].mean()
        print(f"\n--- Performance Summary for Reliable Districts (>={CONFIG['LOW_DATA_THRESHOLD']} data points) ---")
        print(f"  Number of Reliable Districts: {len(reliable_districts)}")
        print(f"  R-squared (R²): {r2_reliable:.4f}")
        print(f"  Mean Absolute Error (MAE): {mae_reliable:.2f} dt/ha")

    print("\n--- Champion Ensemble Evaluation Complete ---")
    print(f"All reports and plots saved in: {report_dir}")


if __name__ == "__main__":
    main()
