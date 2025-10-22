import pandas as pd
import geopandas as gpd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# --- Configuration ---
sns.set_theme(style="whitegrid")
BASE_RESULTS_DIR = 'reports/hybrid_stable_calibration_results'
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
LOW_DATA_THRESHOLD = 10
MIN_DATAPOINTS_FOR_PLOT = 5
CONFIDENCE_LEVEL = 0.95


# --- Core Metric Functions ---

def calculate_overall_metrics(df: pd.DataFrame):
    """Calculates both point and interval metrics for the entire dataset."""
    r2 = r2_score(df['kreisYield'], df['predicted_yield_median'])
    mae = mean_absolute_error(df['kreisYield'], df['predicted_yield_median'])
    df['is_covered'] = (df['kreisYield'] >= df['predicted_yield_lower']) & (
                df['kreisYield'] <= df['predicted_yield_upper'])
    picp = df['is_covered'].mean()
    df['interval_width'] = df['predicted_yield_upper'] - df['predicted_yield_lower']
    mpiw = df['interval_width'].mean()
    return {'r2': r2, 'mae': mae, 'picp': picp, 'mpiw': mpiw}


def calculate_district_metrics(df: pd.DataFrame):
    """Calculates R², MAE, and data count for each district."""
    print("-> Calculating district-level performance metrics...")

    def r2_safe(g):
        return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99

    # This function now expects 'name' to be in the dataframe 'df'
    performance = df.groupby('district_no').apply(
        lambda g: pd.Series({
            'r2': r2_safe(g),
            'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
            'name': g['name'].iloc[0] if not g.empty else 'Unknown',  # This line caused the error
            'data_point_count': len(g)
        })
    ).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < LOW_DATA_THRESHOLD
    return performance


# --- Plotting Functions (No changes needed here) ---

def plot_performance_map(district_perf: pd.DataFrame, gdf: gpd.GeoDataFrame, output_dir: str, model_name: str):
    print("-> Generating R² performance map...")
    merged_gdf = gdf.merge(district_perf, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'] == True]
    if not low_data_gdf.empty:
        low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black',
                                 label=f'Low Data (< {LOW_DATA_THRESHOLD} years)')
    plt.legend(handles=[hatch_patch], loc='lower left')
    ax.set_title(f'Model Performance (R²) by District - {model_name}', fontsize=16)
    ax.set_axis_off()
    plt.savefig(os.path.join(output_dir, '01_r_squared_map.png'), bbox_inches='tight')
    plt.close()


def plot_r2_vs_data_count(district_perf: pd.DataFrame, output_dir: str, model_name: str):
    print("-> Generating R² vs. Data Count plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_perf, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='grey', linestyle='--')
    plt.title(f"Data Availability vs. Model Performance - {model_name}", fontsize=16)
    plt.xlabel("Number of Years in Backtest per District")
    plt.ylabel("R-squared (R²)")
    plt.savefig(os.path.join(output_dir, '02_r2_vs_data_count.png'), bbox_inches='tight')
    plt.close()


def plot_yearly_performance(df: pd.DataFrame, output_dir: str, model_name: str):
    print("-> Generating yearly performance plot...")
    yearly_perf = df.groupby('year').apply(
        lambda g: r2_score(g['kreisYield'], g['predicted_yield_median'])).reset_index(name='r2')
    plt.figure(figsize=(12, 7))
    plt.plot(yearly_perf['year'], yearly_perf['r2'], color='blue', marker='o')
    plt.axhline(0, color='grey', linestyle='--')
    plt.title(f'Model Performance Over Time - {model_name}', fontsize=16)
    plt.xlabel('Year');
    plt.ylabel('R-squared (R²)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, '03_performance_over_time.png'), bbox_inches='tight')
    plt.close()


def plot_best_worst_districts(dist_perf: pd.DataFrame, results_df: pd.DataFrame, output_dir: str, model_name: str):
    print("-> Generating best vs. worst district timelines...")
    reliable_perf = dist_perf[dist_perf['data_point_count'] >= MIN_DATAPOINTS_FOR_PLOT].copy()
    if len(reliable_perf) < 6:
        print(f"   Warning: Not enough reliable districts (found {len(reliable_perf)}) to plot best/worst. Skipping.")
        return
    best = reliable_perf.nlargest(3, 'r2')
    worst = reliable_perf.nsmallest(3, 'r2')
    districts_to_plot = pd.concat([best, worst])
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, (idx, district) in enumerate(districts_to_plot.iterrows()):
        ax = axes.flatten()[i]
        data = results_df[results_df['district_no'] == district['district_no']].sort_values('year')
        ax.plot(data['year'], data['kreisYield'], 'o-', label='Actual Yield', color='navy', zorder=3)
        ax.plot(data['year'], data['predicted_yield_median'], '--', label='Median Prediction', color='red', zorder=4)
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'],
                        color='red', alpha=0.2, label=f'{CONFIDENCE_LEVEL:.0%} Interval', zorder=2)
        title_prefix = "Best" if i < 3 else "Worst"
        ax.set_title(f"{title_prefix}: {district['name']}\n(R² = {district['r2']:.2f})")
        ax.legend()
    plt.suptitle(f"Prediction Timelines for Best & Worst Districts - {model_name}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, '04_best_worst_districts.png'), bbox_inches='tight')
    plt.close()


def plot_national_average(df: pd.DataFrame, output_dir: str, model_name: str):
    print("-> Generating national average timeline...")
    yearly_avg = df.groupby('year').agg({
        'kreisYield': 'mean',
        'predicted_yield_median': 'mean',
        'predicted_yield_lower': 'mean',
        'predicted_yield_upper': 'mean'
    }).reset_index()
    plt.figure(figsize=(14, 8))
    plt.plot(yearly_avg['year'], yearly_avg['kreisYield'], 'o-', label='National Average Actual Yield', color='navy',
             zorder=3)
    plt.plot(yearly_avg['year'], yearly_avg['predicted_yield_median'], '--', label='National Average Prediction',
             color='red', zorder=4)
    plt.fill_between(yearly_avg['year'], yearly_avg['predicted_yield_lower'], yearly_avg['predicted_yield_upper'],
                     color='red', alpha=0.2, label=f'{CONFIDENCE_LEVEL:.0%} Prediction Interval', zorder=2)
    plt.title(f"National Average Yield: Actual vs. Predicted - {model_name}", fontsize=16)
    plt.xlabel("Year");
    plt.ylabel("Yield (dt/ha)")
    plt.legend();
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(output_dir, '05_national_average_timeline.png'), bbox_inches='tight')
    plt.close()


# --- Main Orchestrator ---

def run_full_evaluation(results_path: str, gdf: gpd.GeoDataFrame):
    """Loads a single result file, runs the full evaluation pipeline, and returns summary metrics."""
    model_name = os.path.basename(os.path.dirname(results_path)).replace('_run', '').replace('hybrid_', '')
    print(f"\n--- Evaluating Model: {model_name.upper()} ---")

    output_dir = os.path.join(os.path.dirname(results_path), 'evaluation_report')
    os.makedirs(output_dir, exist_ok=True)

    # Load results data
    df = pd.read_csv(results_path)
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)

    # --- FIX IS HERE: Merge with GeoDataFrame to get district names EARLY ---
    # This adds the 'name' column required by `calculate_district_metrics`
    df = pd.merge(df, gdf[['district_no', 'name']], on='district_no', how='left')
    # --- END FIX ---

    # RENAME COLUMNS to match the plotting functions' expectations
    df.rename(columns={
        'predicted_yield': 'predicted_yield_median',
        'lower_bound': 'predicted_yield_lower',
        'upper_bound': 'predicted_yield_upper'
    }, inplace=True)

    # Calculate metrics and generate plots
    overall_metrics = calculate_overall_metrics(df)
    district_perf = calculate_district_metrics(df)  # This will now succeed

    plot_performance_map(district_perf, gdf, output_dir, model_name.upper())
    plot_r2_vs_data_count(district_perf, output_dir, model_name.upper())
    plot_yearly_performance(df, output_dir, model_name.upper())
    plot_best_worst_districts(district_perf, df, output_dir, model_name.upper())
    plot_national_average(df, output_dir, model_name.upper())

    print(f"-> Evaluation for {model_name.upper()} complete. Reports saved in: {output_dir}")
    overall_metrics['model'] = f"hybrid_{model_name}"
    return overall_metrics


def main():
    """Finds and evaluates all hybrid model results automatically."""
    print("--- Starting Automated Hybrid Model Evaluation ---")

    try:
        gdf_districts = gpd.read_file(GEOJSON_PATH)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load GeoJSON file at '{GEOJSON_PATH}'. Details: {e}")
        return

    search_pattern = os.path.join(BASE_RESULTS_DIR, 'hybrid_*_run', '*.csv')
    result_files = [f for f in glob.glob(search_pattern) if 'summary_metrics' not in f]

    if not result_files:
        print(f"❌ Error: No result files found matching pattern in '{BASE_RESULTS_DIR}'.")
        print("   Please run `backtest_hybrid_asymmetric.py` for both 'lasso' and 'gam'.")
        return

    all_metrics = [run_full_evaluation(file, gdf_districts) for file in result_files]

    # --- Final Comparison ---
    print("\n\n" + "=" * 50)
    print("      HYBRID MODEL PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"Target Interval Coverage (PICP): {CONFIDENCE_LEVEL:.0%}\n")

    comparison_df = pd.DataFrame(all_metrics).set_index('model').reindex(columns=['r2', 'mae', 'picp', 'mpiw'])

    formatted_df = comparison_df.copy()
    formatted_df['r2'] = formatted_df['r2'].map('{:.4f}'.format)
    formatted_df['mae'] = formatted_df['mae'].map('{:.2f} dt/ha'.format)
    formatted_df['picp'] = formatted_df['picp'].map('{:.2%}'.format)
    formatted_df['mpiw'] = formatted_df['mpiw'].map('{:.2f} dt/ha'.format)

    print(formatted_df)
    print("=" * 50)
    print("\nDetailed plots and reports are in each model's 'evaluation_report' directory.")


if __name__ == "__main__":
    main()