# File: src/models/backtesting_quantile.py
# Description: A deep-dive diagnostic script to evaluate the QUANTILE model performance
#              at the district level. This version is hardened against data leakage
#              and includes analysis of uncertainty and performance over time.

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

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
MODEL_PATH_LOWER = os.path.join('src/models', 'final_xgb_model_champion_lower.joblib')
MODEL_PATH_MEDIAN = os.path.join('src/models', 'final_xgb_model_champion_median.joblib')
MODEL_PATH_UPPER = os.path.join('src/models', 'final_xgb_model_champion_upper.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = os.path.join('reports', 'figures', 'district_level_diagnostics_quantile')

BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2021
LOW_DATA_THRESHOLD = 10  # Districts with fewer years of data than this will be flagged


def run_quantile_backtest(df: pd.DataFrame, model_lower_template: XGBRegressor, model_median_template: XGBRegressor,
                          model_upper_template: XGBRegressor):
    """
    Performs a rolling forecast origin backtest for the three quantile models,
    ensuring models are re-initialized each year.
    """
    print(f"\n--- Starting Quantile Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")
    all_predictions = []

    # Get feature names from the median model template
    try:
        feature_cols = model_median_template.get_booster().feature_names
    except AttributeError:
        print("Warning: Could not get feature names from booster. Falling back to `feature_names_in_`.")
        feature_cols = model_median_template.feature_names_in_

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Years"):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()

        if test_df.empty or train_df.empty:
            continue

        X_train, y_train = train_df[feature_cols], train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        # --- FIX: Clone all three models for a fresh start each year ---
        model_lower = clone(model_lower_template)
        model_median = clone(model_median_template)
        model_upper = clone(model_upper_template)

        # Fit all three fresh models on historical data
        model_lower.fit(X_train, y_train)
        model_median.fit(X_train, y_train)
        model_upper.fit(X_train, y_train)

        # Predict and add back the trend
        pred_detrended_lower = model_lower.predict(X_test)
        pred_detrended_median = model_median.predict(X_test)
        pred_detrended_upper = model_upper.predict(X_test)

        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name', 'yield_trend']].copy()
        fold_results['predicted_yield_lower'] = pred_detrended_lower + test_df['yield_trend']
        fold_results['predicted_yield_median'] = pred_detrended_median + test_df['yield_trend']
        fold_results['predicted_yield_upper'] = pred_detrended_upper + test_df['yield_trend']
        all_predictions.append(fold_results)

    if not all_predictions:
        print("❌ CRITICAL ERROR: No predictions were made. Check year ranges and data availability.")
        return pd.DataFrame()

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield_median'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    results_df['deviation'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']
    results_df['deviation'] = results_df['deviation'].apply(lambda x: max(x, 0))  # Ensure non-negative
    print("\n✅ Quantile backtest complete.")
    return results_df


def calculate_district_metrics(results_df: pd.DataFrame):
    """Calculates R², MAE (for median), and average deviation for each district."""
    print("Calculating performance and uncertainty metrics for each district...")

    def r2_safe(g):
        return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99

    performance = results_df.groupby('district_no').apply(
        lambda g: pd.Series({
            'r2': r2_safe(g),
            'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
            'avg_deviation': g['deviation'].mean(),
            'name': g['name'].iloc[0] if not g.empty else 'Unknown',
            'data_point_count': len(g)
        })
    ).reset_index()

    performance['is_low_data'] = performance['data_point_count'] < LOW_DATA_THRESHOLD
    save_path = os.path.join(REPORT_DIR, 'district_level_quantile_metrics.csv')
    performance.to_csv(save_path, index=False)
    print(f"✅ Detailed district metrics saved to {save_path}")
    return performance


def analyze_yearly_performance_quantile(results_df: pd.DataFrame):
    """Calculates and plots key quantile metrics for each year in the backtest."""
    print("Calculating yearly performance and uncertainty metrics...")

    def coverage_metric(g):
        # Calculates the percentage of actual values that fall within the prediction interval
        in_interval = ((g['kreisYield'] >= g['predicted_yield_lower']) & (
                    g['kreisYield'] <= g['predicted_yield_upper'])).mean()
        return in_interval

    yearly_perf = results_df.groupby('year').apply(
        lambda g: pd.Series({
            'r2_median': r2_score(g['kreisYield'], g['predicted_yield_median']),
            'mae_median': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
            'avg_deviation': g['deviation'].mean(),
            'coverage_80pct': coverage_metric(g)  # For 0.1 and 0.9 quantiles, the target is 80%
        })
    ).reset_index()

    print("--- Yearly Quantile Performance Summary ---")
    print(yearly_perf)
    print("-------------------------------------------")

    # Plotting R² and Average Deviation over time
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Median R-squared (R²)', color='blue', fontsize=12)
    ax1.plot(yearly_perf['year'], yearly_perf['r2_median'], color='blue', marker='o', label='Median R²')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(0, color='grey', linestyle='--')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Deviation (dt/ha)', color='red', fontsize=12)
    ax2.plot(yearly_perf['year'], yearly_perf['avg_deviation'], color='red', marker='x', linestyle='--',
             label='Avg. Deviation (Uncertainty)')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Model Performance and Uncertainty Over Time', fontsize=16)
    fig.tight_layout()
    # To add a single legend for both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    save_path = os.path.join(REPORT_DIR, '05_quantile_performance_over_time.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Yearly quantile performance plot saved to {save_path}")
    return yearly_perf


# All plotting functions (plot_performance_map_with_hatching, etc.) remain the same
# and can be copied from your previous script. For completeness, they are included here.

def plot_performance_map_with_hatching(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    print("Generating R-squared Map...")
    # (Code is identical to the previous script, no changes needed)
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, legend_kwds={'label': "Median Prediction R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'] == True]
    if not low_data_gdf.empty:
        low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black',
                                 label=f'Low Data (< {LOW_DATA_THRESHOLD} years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability')
    ax.set_title('Model Median Performance (R²) by District', fontsize=16)
    ax.set_axis_off()
    plt.savefig(os.path.join(REPORT_DIR, '01_r_squared_map.png'), bbox_inches='tight')
    plt.close()
    print("✅ R-squared map saved.")


def plot_deviation_map(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    print("Generating Prediction Deviation Map...")
    # (Code is identical, no changes needed)
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='avg_deviation', cmap='plasma', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True,
                    legend_kwds={'label': "Average Prediction Deviation (dt/ha)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'})
    ax.set_title('Model Prediction Uncertainty (Average Deviation) by District', fontsize=16)
    ax.set_axis_off()
    plt.savefig(os.path.join(REPORT_DIR, '02_deviation_map.png'), bbox_inches='tight')
    plt.close()
    print("✅ Deviation map saved.")


def plot_r2_vs_data_count(district_performance: pd.DataFrame):
    print("Generating R² vs. Data Point Count plot...")
    # (Code is identical, ensure ylim is set)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='grey', linestyle='--')
    plt.title("Relationship Between Data Availability and Median Performance", fontsize=16)
    plt.xlabel("Number of Years in Backtest per District")
    plt.ylabel("R-squared (R²)")
    plt.legend(title='Is Low Data?')
    plt.savefig(os.path.join(REPORT_DIR, '03_r2_vs_data_count.png'), bbox_inches='tight')
    plt.close()
    print("✅ R² vs. Count plot saved.")


def plot_deviation_vs_data_count(district_performance: pd.DataFrame):
    print("Generating Deviation vs. Data Point Count plot...")
    # (Code is identical, no changes needed)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='avg_deviation', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.title("Relationship Between Data Availability and Model Uncertainty", fontsize=16)
    plt.xlabel("Number of Years in Backtest per District")
    plt.ylabel("Average Prediction Deviation (dt/ha)")
    plt.legend(title='Is Low Data?')
    plt.savefig(os.path.join(REPORT_DIR, '04_deviation_vs_data_count.png'), bbox_inches='tight')
    plt.close()
    print("✅ Deviation vs. Count plot saved.")


def main():
    """Main function to orchestrate the district-level quantile evaluation pipeline."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting District-Level Quantile Model Evaluation Pipeline ---")

    try:
        model_lower_template = joblib.load(MODEL_PATH_LOWER)
        model_median_template = joblib.load(MODEL_PATH_MEDIAN)
        model_upper_template = joblib.load(MODEL_PATH_UPPER)
        df = pd.read_csv(DATA_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)

        # Data Preparation
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
        print("✅ Models, data, and geo-data loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}")
        return

    # --- FIX: Apply CORRECT Causal Detrending ONCE at the start ---
    print("\n--- Applying Causal (Trailing Mean) Detrending to Prevent Data Leakage ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x
    )
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    # --- Run the Backtest ---
    backtest_results = run_quantile_backtest(df, model_lower_template, model_median_template, model_upper_template)

    if backtest_results.empty:
        print("❌ Backtest did not produce results. Terminating.")
        return

    # --- Run All Analyses ---
    district_performance = calculate_district_metrics(backtest_results)
    yearly_performance = analyze_yearly_performance_quantile(backtest_results)  # New, detailed analysis

    # --- Generate Visualizations ---
    plot_performance_map_with_hatching(district_performance, gdf_districts)
    plot_deviation_map(district_performance, gdf_districts)
    plot_r2_vs_data_count(district_performance)
    plot_deviation_vs_data_count(district_performance)

    # --- Final Summary Metrics on Reliable Data ---
    reliable_districts = district_performance[~district_performance['is_low_data']]
    reliable_results = backtest_results[backtest_results['district_no'].isin(reliable_districts['district_no'])]

    if reliable_results.empty:
        print("⚠️ Warning: No districts were classified as 'reliable'. Cannot compute summary metrics.")
    else:
        mae_reliable = reliable_results['abs_error'].mean()
        r2_reliable = r2_score(reliable_results['kreisYield'], reliable_results['predicted_yield_median'])
        avg_dev_reliable = reliable_results['deviation'].mean()

        print("\n--- Performance Summary on RELIABLE Districts (>= 10 years of data) ---")
        print(f"  Number of reliable districts: {len(reliable_districts)} / {len(district_performance)}")
        print(f"  Median MAE:                   {mae_reliable:.2f} dt/ha")
        print(f"  Median R-squared (R²):        {r2_reliable:.4f}")
        print(f"  Average Deviation (80% PI):   {avg_dev_reliable:.2f} dt/ha")
        print("---------------------------------------------------------------------")


if __name__ == "__main__":
    main()