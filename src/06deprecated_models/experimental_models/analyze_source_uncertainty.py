# File: src/visualization/analyze_source_uncertainty.py
# Description: Loads the results from the Source Uncertainty Deep Ensemble backtest,
#              calculates all key metrics, and generates a full suite of diagnostic
#              and uncertainty plots for a comprehensive final analysis.

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
RESULTS_CSV_PATH = 'reports/figures/district_level_diagnostics/source_uncertainty_ensemble/source_uncertainty_ensemble_results.csv'
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = 'reports/figures/district_level_diagnostics/source_uncertainty_ensemble'

# Constants for plotting
LOW_DATA_THRESHOLD = 10
MIN_DATAPOINTS_FOR_PLOT = 10


def print_summary_metrics(results_df: pd.DataFrame):
    """Recalculates and prints the headline performance metrics from the CSV file."""
    print("--- Recalculating Performance Summary for Source Uncertainty Ensemble ---")

    results_df['is_covered'] = (results_df['kreisYield'] >= results_df['predicted_yield_lower']) & \
                               (results_df['kreisYield'] <= results_df['predicted_yield_upper'])
    results_df['interval_width'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']

    picp = results_df['is_covered'].mean()
    mpiw = results_df['interval_width'].mean()
    r2 = r2_score(results_df['kreisYield'], results_df['predicted_yield_median'])
    mae = mean_absolute_error(results_df['kreisYield'], results_df['predicted_yield_median'])
    avg_epistemic = results_df['epistemic_uncertainty'].mean()

    print(f"Prediction Interval Coverage (PICP): {picp:.2%}")
    print(f"Mean Prediction Interval Width (MPIW): {mpiw:.2f} dt/ha")
    print(f"Overall R-squared (R²): {r2:.4f}")
    print(f"Overall Mean Absolute Error (MAE): {mae:.2f} dt/ha")
    print(f"\nAverage Epistemic Uncertainty (Model Ignorance): {avg_epistemic:.2f} dt/ha")


def calculate_district_metrics(results_df: pd.DataFrame):
    """Calculates R², MAE, and data count for each district."""
    print("\n-> Calculating district-level metrics...")

    def r2_safe(g): return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99

    performance = results_df.groupby('district_no').apply(lambda g: pd.Series({
        'r2': r2_safe(g),
        'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
        'name': g['name'].iloc[0] if 'name' in g.columns and not g['name'].empty else 'N/A',
        'data_point_count': len(g)
    })).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < LOW_DATA_THRESHOLD
    return performance


# --- Plotting Functions ---

def plot_uncertainty_map(results_df: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    """Generates the geographic map of epistemic uncertainty."""
    print("-> Generating Model Ignorance (Epistemic Uncertainty) Map...")
    district_uncertainty = results_df.groupby('district_no')['epistemic_uncertainty'].mean().reset_index()
    merged_gdf = gdf_districts.merge(district_uncertainty, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='epistemic_uncertainty', cmap='Reds', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True,
                    legend_kwds={'label': "Average Epistemic Uncertainty (dt/ha)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'})
    ax.set_title('Model Ignorance by District (Source Uncertainty Ensemble)', fontsize=16)
    ax.set_axis_off()
    plt.savefig(os.path.join(REPORT_DIR, '01_epistemic_uncertainty_map.png'), bbox_inches='tight')
    plt.close()


def plot_national_average_timeline(backtest_results: pd.DataFrame):
    """Generates the national average yield vs. prediction timeline plot."""
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
    plt.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='Median Prediction (Ensemble Mean)', color='red',
             linestyle='--', zorder=4)
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='green',
                     alpha=0.25, label='Ensemble Interval', zorder=2)
    plt.title("National Average Yield vs. Predicted Interval (Source Uncertainty Ensemble)", fontsize=16)
    plt.xlabel("Year");
    plt.ylabel("Yield (dt/ha)")
    plt.legend();
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(REPORT_DIR, '02_national_average_timeline.png'), bbox_inches='tight')
    plt.close()


def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame):
    """Generates timeline plots for the 3 best and 3 worst reliable districts."""
    print("-> Generating Best vs. Worst District Timelines...")
    reliable_perf = district_performance[district_performance['data_point_count'] >= MIN_DATAPOINTS_FOR_PLOT]
    if len(reliable_perf) < 6:
        print("   Warning: Not enough reliable districts to plot. Skipping.")
        return
    best_districts = reliable_perf.nlargest(3, 'r2')
    worst_districts = reliable_perf.nsmallest(3, 'r2')
    districts_to_plot = pd.concat([best_districts, worst_districts])
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, (_, district_info) in enumerate(districts_to_plot.iterrows()):
        ax = axes.flatten()[i]
        data = backtest_results[backtest_results['district_no'] == district_info['district_no']]
        ax.plot(data['year'], data['kreisYield'], label='Actual Yield', color='navy', marker='o', markersize=4,
                zorder=3)
        ax.plot(data['year'], data['predicted_yield_median'], label='Median Prediction', color='red', linestyle='--',
                zorder=4)
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'], color='green',
                        alpha=0.25, label='Ensemble Interval')
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})")
        ax.legend();
        ax.grid(True, linestyle=':')
    plt.suptitle("Prediction Timelines for Best & Worst Districts (Source Uncertainty Ensemble)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(REPORT_DIR, '03_best_worst_districts.png'), bbox_inches='tight')
    plt.close()


def plot_performance_map(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    """Generates the geographic map of R-squared performance."""
    print("-> Generating R-squared Map...")
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)

    merged_gdf['is_low_data'] = merged_gdf['is_low_data'].fillna(False)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data']]

    if not low_data_gdf.empty:
        low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)

    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black',
                                 label=f'Low Data (< {LOW_DATA_THRESHOLD} years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability')
    ax.set_title('Model Performance (R²) by District - Source Uncertainty Ensemble', fontsize=16)
    ax.set_axis_off()
    plt.savefig(os.path.join(REPORT_DIR, '04_r_squared_map.png'), bbox_inches='tight')
    plt.close()


def main():
    """Main function to load results and generate all diagnostic plots."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"--- Generating Full Analysis for Source Uncertainty Ensemble ---")
    print(f"Loading results from: {RESULTS_CSV_PATH}")

    try:
        results_df = pd.read_csv(RESULTS_CSV_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        results_df['district_no'] = results_df['district_no'].astype(str).str.zfill(5)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        # Merge names into results for plotting
        results_df = pd.merge(results_df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: Could not find a required file. Details: {e}")
        return

    # Recalculate and display the headline metrics
    print_summary_metrics(results_df)

    # Calculate the per-district metrics needed for the plots
    district_performance = calculate_district_metrics(results_df)

    # Generate all the standard and new plots
    plot_uncertainty_map(results_df, gdf_districts)
    plot_national_average_timeline(results_df)
    plot_best_worst_district_timelines(district_performance, results_df)
    plot_performance_map(district_performance, gdf_districts)

    print(f"\n--- Analysis Complete ---")
    print(f"All standard and new diagnostic plots have been saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()