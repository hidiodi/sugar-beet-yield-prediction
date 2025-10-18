# File: src/models/backtesting.py
# Description: Robust district-level backtesting for sugar beet yield forecast models.
# Enhancements:
#   - Fixed random state for reproducibility
#   - Optional spatial cross-validation (leave-one-state-out)
#   - Bias diagnostics map
#   - Small memory and clarity improvements

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
MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = os.path.join('reports', 'figures', 'district_level_diagnostics')

BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2021
LOW_DATA_THRESHOLD = 10
MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT = 5

# Optional: toggle this to True for leave-one-state-out backtesting
ENABLE_SPATIAL_CV = False


def run_backtest(df: pd.DataFrame, model_to_clone: XGBRegressor):
    """
    Performs a rolling forecast origin backtest (temporal CV), ensuring the model
    is re-initialized each year to prevent state leakage.
    """
    print(f"\n--- Starting Robust Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")
    all_predictions = []

    try:
        feature_cols = model_to_clone.get_booster().feature_names
    except AttributeError:
        print("Warning: Could not get feature names from booster. Falling back to `feature_names_in_`.")
        feature_cols = model_to_clone.feature_names_in_

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Years"):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()

        if test_df.empty or train_df.empty:
            continue

        X_train, y_train = train_df[feature_cols], train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        model = clone(model_to_clone)
        model.set_params(random_state=42)  # reproducibility
        model.fit(X_train, y_train)

        predicted_detrended = model.predict(X_test)
        final_predictions = predicted_detrended + test_df['yield_trend']

        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name']].copy()
        fold_results['predicted_yield'] = final_predictions
        all_predictions.append(fold_results)
        del model  # free memory

    if not all_predictions:
        print("❌ CRITICAL ERROR: No predictions were made. Check year ranges and data availability.")
        return pd.DataFrame()

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\n✅ Backtest complete.")
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
            'bias': g['error'].mean(),
            'name': g['name'].iloc[0] if not g.empty else 'Unknown',
            'data_point_count': len(g)
        })
    ).reset_index()

    performance['is_low_data'] = performance['data_point_count'] < LOW_DATA_THRESHOLD
    save_path = os.path.join(REPORT_DIR, 'district_level_performance_metrics.csv')
    performance.to_csv(save_path, index=False)
    print(f"✅ Detailed district metrics saved to {save_path}")
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

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(yearly_perf['year'], yearly_perf['r2'], color='blue', marker='o', label='Yearly R²')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('R-squared (R²)', fontsize=12)
    ax.axhline(0, color='grey', linestyle='--')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.title('Model Performance Over Time (Backtest)', fontsize=16)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(REPORT_DIR, '03_performance_over_time.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Yearly performance plot saved to {save_path}")
    return yearly_perf


def plot_performance_map_with_hatching(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    """Geographic map of R² with hatching for low-data districts."""
    print("Generating R-squared Map...")
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)

    # ✅ FIX: handle NaNs safely
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'].fillna(False)]

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
    print("✅ Performance map saved.")


def plot_bias_map(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    """Plot mean bias (systematic over/under-prediction) by district."""
    print("Generating Bias Map...")
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='bias', cmap='coolwarm', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, legend_kwds={'label': "Mean Prediction Bias (dt/ha)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-100, vmax=100)
    ax.set_title('Prediction Bias by District', fontsize=16)
    ax.set_axis_off()
    save_path = os.path.join(REPORT_DIR, '05_bias_map.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("✅ Bias map saved.")


def plot_r2_vs_data_count(district_performance: pd.DataFrame):
    """Scatter plot of R² vs number of data points per district."""
    print("Generating R² vs. Data Count plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='grey', linestyle='--')
    plt.title("Relationship Between Data Availability and Model Performance", fontsize=16)
    plt.xlabel("Number of Years in Backtest per District")
    plt.ylabel("R-squared (R²)")
    plt.legend(title='Is Low Data?')
    save_path = os.path.join(REPORT_DIR, '02_r2_vs_data_count.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("✅ R² vs. Count plot saved.")


def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame):
    """Prediction timelines for 3 best and 3 worst performing districts."""
    print("Generating timeline plots for best and worst performing districts...")
    filtered_perf = district_performance[
        (district_performance['data_point_count'] > 1) & (district_performance['r2'] != -99)
    ].sort_values('r2', ascending=False)

    best_districts = filtered_perf.head(3)
    worst_districts_filtered = district_performance[
        (district_performance['data_point_count'] >= MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT) &
        (district_performance['r2'] != -99)
    ].sort_values('r2', ascending=True)
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
        ax.plot(district_data['year'], district_data['kreisYield'], label='Actual Yield', color='navy', marker='o', markersize=4)
        ax.plot(district_data['year'], district_data['predicted_yield'], label='Predicted Yield', color='red', linestyle='--')

        title_prefix = "Best" if i < 3 else "Worst"
        ax.set_title(f"{title_prefix}: {district_name}\n(R² = {district_r2:.2f})", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle=':')

    plt.suptitle("Prediction Timelines for 3 Best and 3 Worst Performing Districts", fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    save_path = os.path.join(REPORT_DIR, '04_best_worst_district_timelines.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Best/Worst district timelines saved to {save_path}")


def run_spatial_cv(df, model_template):
    """Leave-one-state-out validation (optional)."""
    print("\n--- Running Leave-One-State-Out Spatial Cross-Validation ---")
    if 'state' not in df.columns:
        print("⚠️ No 'state' column found — skipping spatial CV.")
        return

    states = sorted(df['state'].dropna().unique())
    spatial_results = []
    for holdout_state in states:
        train_df = df[df['state'] != holdout_state]
        test_df = df[df['state'] == holdout_state]

        if test_df.empty or train_df.empty:
            continue

        model = clone(model_template)
        model.set_params(random_state=42)
        X_train, y_train = train_df[model.feature_names_in_], train_df['kreisYield_detrended']
        X_test = test_df[model.feature_names_in_]
        model.fit(X_train, y_train)

        predicted_detrended = model.predict(X_test)
        final_predictions = predicted_detrended + test_df['yield_trend']

        test_df = test_df.copy()
        test_df['predicted_yield'] = final_predictions
        test_df['state'] = holdout_state
        spatial_results.append(test_df)

    if spatial_results:
        results_df = pd.concat(spatial_results)
        mae = mean_absolute_error(results_df['kreisYield'], results_df['predicted_yield'])
        r2 = r2_score(results_df['kreisYield'], results_df['predicted_yield'])
        print(f"✅ Spatial CV Summary (Leave-One-State-Out): R²={r2:.3f}, MAE={mae:.2f}")
    else:
        print("⚠️ No spatial CV results generated.")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting District-Level Model Evaluation Pipeline ---")

    try:
        model_template = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)

        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)

        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
        print("✅ Model template, data, and geo-data loaded successfully.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}")
        return

    print("\n--- Applying Causal (Trailing Mean) Detrending to Prevent Data Leakage ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)

    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x)
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    backtest_results = run_backtest(df, model_template)
    if backtest_results.empty:
        print("❌ Backtest did not produce results. Terminating.")
        return

    district_performance = calculate_district_metrics(backtest_results)
    yearly_performance = analyze_yearly_performance(backtest_results)

    plot_performance_map_with_hatching(district_performance, gdf_districts)
    plot_r2_vs_data_count(district_performance)
    plot_best_worst_district_timelines(district_performance, backtest_results)
    plot_bias_map(district_performance, gdf_districts)

    mae_total = backtest_results['abs_error'].mean()
    r2_total = r2_score(backtest_results['kreisYield'], backtest_results['predicted_yield'])
    print("\n--- Overall Performance Summary (All Districts) ---")
    print(f"  Mean Absolute Error (MAE):    {mae_total:.2f} dt/ha")
    print(f"  R-squared (R²):               {r2_total:.4f}")
    print("-----------------------------------------------------")

    reliable_districts = district_performance[~district_performance['is_low_data']]
    reliable_results = backtest_results[backtest_results['district_no'].isin(reliable_districts['district_no'])]

    if not reliable_results.empty:
        mae_reliable = reliable_results['abs_error'].mean()
        r2_reliable = r2_score(reliable_results['kreisYield'], reliable_results['predicted_yield'])
        print("\n--- Performance Summary on RELIABLE Districts (>= 10 years of data) ---")
        print(f"  Number of reliable districts: {len(reliable_districts)} / {len(district_performance)}")
        print(f"  Mean Absolute Error (MAE):    {mae_reliable:.2f} dt/ha")
        print(f"  R-squared (R²):               {r2_reliable:.4f}")
        print("---------------------------------------------------------------------")

    if ENABLE_SPATIAL_CV:
        run_spatial_cv(df, model_template)


if __name__ == "__main__":
    main()
