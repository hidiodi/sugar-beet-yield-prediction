# File: src/models/backtesting.py
# Description: A deep-dive diagnostic script to evaluate model performance
#              at the district level. This version is hardened against data leakage
#              and includes analysis of performance over time to detect non-stationarity.

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

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
# Ensure these paths are correct relative to where you run the script from
MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = os.path.join('reports', 'figures', 'district_level_diagnostics', 'base_model_diagnostics')

BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2024
LOW_DATA_THRESHOLD = 10  # Districts with fewer years of data than this will be flagged
MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT = 1  # Threshold for worst districts plot


def run_backtest(df: pd.DataFrame, model_to_clone: XGBRegressor):
    """
    Performs a rolling forecast origin backtest, ensuring the model is
    re-initialized each year to prevent state leakage.
    """
    print(f"\n--- Starting Robust Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")
    all_predictions = []

    # Get feature names from the model template before the loop
    # This assumes the joblib model has been fit at least once to have feature names
    try:
        feature_cols = model_to_clone.get_booster().feature_names
    except AttributeError:
        # Fallback if the model saved was a scikit-learn pipeline or not yet fit
        # You may need to adjust this based on your exact saved model object
        print("Warning: Could not get feature names from booster. Falling back to `feature_names_in_`.")
        feature_cols = model_to_clone.feature_names_in_

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Years"):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()

        # Skip years where there is no data to train or test on
        if test_df.empty or train_df.empty:
            continue

        X_train, y_train = train_df[feature_cols], train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        # --- FIX: Clone the model for a fresh start each year ---
        # This prevents the model from retaining state from the previous fold.
        model = clone(model_to_clone)
        model.fit(X_train, y_train)

        # Predict the detrended anomaly and add back the trend
        predicted_detrended = model.predict(X_test)
        final_predictions = predicted_detrended + test_df['yield_trend']

        # Store results
        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name']].copy()
        fold_results['predicted_yield'] = final_predictions
        all_predictions.append(fold_results)

    if not all_predictions:
        print("❌ CRITICAL ERROR: No predictions were made. Check year ranges and data availability.")
        return pd.DataFrame()

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\nBacktest complete.")
    return results_df


def calculate_district_metrics(results_df: pd.DataFrame):
    """Calculates R², MAE, and data point count for each district."""
    print("Calculating performance metrics and data counts for each district...")

    def r2_safe(g):
        return r2_score(g['kreisYield'], g['predicted_yield']) if len(g) > 1 else -99

    performance = results_df.groupby('district_no').apply(
        lambda g: pd.Series({
            'r2': r2_safe(g),
            'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield']),
            'name': g['name'].iloc[0] if not g.empty else 'Unknown',
            'data_point_count': len(g)
        })
    ).reset_index()

    performance['is_low_data'] = performance['data_point_count'] < LOW_DATA_THRESHOLD

    save_path = os.path.join(REPORT_DIR, 'district_level_performance_metrics.csv')
    performance.to_csv(save_path, index=False)
    print(f"Detailed district metrics saved to {save_path}")
    return performance


def analyze_yearly_performance(results_df: pd.DataFrame):
    """Calculates and plots R² and MAE for each year in the backtest."""
    print("Calculating performance metrics for each year...")
    yearly_perf = results_df.groupby('year').apply(
        lambda g: pd.Series({
            'r2': r2_score(g['kreisYield'], g['predicted_yield']),
            'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield'])
        })
    ).reset_index()

    print("--- Yearly Performance Summary ---")
    print(yearly_perf)
    print("----------------------------------")

    # Plotting the results
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(yearly_perf['year'], yearly_perf['r2'], color='blue', marker='o', label='Yearly R²')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('R-squared (R²)', fontsize=12)
    ax.axhline(0, color='grey', linestyle='--')
    ax.grid(True, which='both', linestyle='--')
    plt.title('Model Performance Over Time (Backtest)', fontsize=16)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(REPORT_DIR, '03_performance_over_time.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Yearly performance plot saved to {save_path}")
    return yearly_perf


def plot_performance_map_with_hatching(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    """Diagnostic: Geographic map of R² with hatching for low-data districts."""
    print("Generating R-squared Map...")
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'] == True]
    if not low_data_gdf.empty:
        low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black',
                                 label=f'Low Data (< {LOW_DATA_THRESHOLD} years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability')
    ax.set_title('Model Performance (R²) by District', fontsize=16)
    ax.set_axis_off()
    save_path = os.path.join(REPORT_DIR, '01_r_squared_map_with_hatching.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("Performance map saved.")


def plot_r2_vs_data_count(district_performance: pd.DataFrame):
    """Diagnostic: Scatter plot to show the relationship between R² and data count."""
    print("Generating R² vs. Data Count plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1)  # Set a fixed, readable range for R²
    plt.axhline(0, color='grey', linestyle='--')
    plt.title("Relationship Between Data Availability and Model Performance", fontsize=16)
    plt.xlabel("Number of Years in Backtest per District")
    plt.ylabel("R-squared (R²)")
    plt.legend(title='Is Low Data?')
    save_path = os.path.join(REPORT_DIR, '02_r2_vs_data_count.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("R² vs. Count plot saved.")


def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame):
    """Diagnostic: Plot prediction timelines for the 3 best and 3 worst districts."""
    print("Generating timeline plots for best and worst performing districts...")

    # Filter out districts with insufficient data points for a meaningful R^2
    filtered_perf = district_performance[
        (district_performance['data_point_count'] > 1) &
        (district_performance['r2'] != -99)
        ].sort_values('r2', ascending=False)

    best_districts = filtered_perf.head(3)

    # For worst districts, specifically filter by MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT
    worst_districts_filtered = district_performance[
        (district_performance['data_point_count'] >= MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT) &
        (district_performance['r2'] != -99)
        ].sort_values('r2', ascending=True)  # Sort ascending for worst

    worst_districts = worst_districts_filtered.head(3)

    districts_to_plot = pd.concat([best_districts, worst_districts])

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    axes = axes.flatten()

    for i, (_, district_info) in enumerate(districts_to_plot.iterrows()):
        district_no = district_info['district_no']
        district_name = district_info['name']
        district_r2 = district_info['r2']
        district_data = backtest_results[backtest_results['district_no'] == district_no].sort_values('year')

        ax = axes[i]
        ax.plot(district_data['year'], district_data['kreisYield'], label='Actual Yield', color='navy', marker='o',
                markersize=4)
        ax.plot(district_data['year'], district_data['predicted_yield'], label='Predicted Yield', color='red',
                linestyle='--')

        title_prefix = "Best" if i < 3 else "Worst"
        ax.set_title(f"{title_prefix}: {district_name}\n(R² = {district_r2:.2f})", fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle=':')

    plt.suptitle("Prediction Timelines for 3 Best and 3 Worst Performing Districts", fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    save_path = os.path.join(REPORT_DIR, '04_best_worst_district_timelines.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Best/Worst district timelines saved to {save_path}")


def main():
    """Main function to orchestrate the district-level evaluation pipeline."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting District-Level Model Evaluation Pipeline ---")

    try:
        model_template = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)

        # Prepare dataframes
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)

        # Merge district names into the main dataframe
        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
        print("Model template, data, and geo-data loaded successfully.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}")
        return

    # --- FIX: Apply CORRECT Causal Detrending ONCE at the start ---
    print("\n--- Applying Causal (Trailing Mean) Detrending to Prevent Data Leakage ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)

    # Use a trailing (causal) window. shift(1) makes it a pure lookback.
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )

    # Handle NaNs created by the shift and rolling window using ONLY past data (forward fill)
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(method='ffill'))

    # For districts with few data points at the start, fill remaining NaNs with the first valid trend value
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x
    )

    # Drop any rows where a trend could not be computed (e.g., the very first year for each district)
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    # --- Run the Backtest ---
    backtest_results = run_backtest(df, model_template)

    if backtest_results.empty:
        print("❌ Backtest did not produce results. Terminating.")
        return

    # --- Run All Analyses ---
    district_performance = calculate_district_metrics(backtest_results)
    yearly_performance = analyze_yearly_performance(backtest_results)

    # --- Generate Visualizations ---
    plot_performance_map_with_hatching(district_performance, gdf_districts)
    plot_r2_vs_data_count(district_performance)
    plot_best_worst_district_timelines(district_performance, backtest_results)

    # --- Final Summary Metrics ---
    # Overall performance
    mae_total = backtest_results['abs_error'].mean()
    r2_total = r2_score(backtest_results['kreisYield'], backtest_results['predicted_yield'])

    print("\n--- Overall Performance Summary (All Districts) ---")
    print(f"  Mean Absolute Error (MAE):    {mae_total:.2f} dt/ha")
    print(f"  R-squared (R²):               {r2_total:.4f}")
    print("-----------------------------------------------------")

    # On Reliable Data
    reliable_districts = district_performance[~district_performance['is_low_data']]
    reliable_results = backtest_results[backtest_results['district_no'].isin(reliable_districts['district_no'])]

    if reliable_results.empty:
        print("⚠️ Warning: No districts were classified as 'reliable'. Cannot compute summary metrics.")
    else:
        mae_reliable = reliable_results['abs_error'].mean()
        r2_reliable = r2_score(reliable_results['kreisYield'], reliable_results['predicted_yield'])

        print("\n--- Performance Summary on RELIABLE Districts (>= 10 years of data) ---")
        print(f"  Number of reliable districts: {len(reliable_districts)} / {len(district_performance)}")
        print(f"  Mean Absolute Error (MAE):    {mae_reliable:.2f} dt/ha")
        print(f"  R-squared (R²):               {r2_reliable:.4f}")
        print("---------------------------------------------------------------------")


if __name__ == "__main__":
    main()