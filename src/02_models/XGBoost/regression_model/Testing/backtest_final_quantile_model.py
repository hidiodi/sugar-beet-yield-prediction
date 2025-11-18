# File: src/models/backtest_final_quantile_model.py
# Description: Backtests the STAT-TREND RESIDUAL model.
# VERSION: 4.0 (Stat-Trend Residual Backtest)

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
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from src import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Use the config dictionaries from the central config file
XGB_CONFIG = config.XGBOOST_TRAINING_CONFIG
BACKTEST_CONFIG = config.BACKTESTING_CONFIG

def run_backtest_residual_model(df: pd.DataFrame, models: dict, full_feature_list: list) -> pd.DataFrame:
    """
    Runs a time-series backtest for a RESIDUAL model using a 5-YEAR ROLLING AVERAGE baseline.
    """
    # --- CHANGE: Update the title to reflect the new baseline ---
    print(
        f"\n--- Starting ROLLING AVERAGE RESIDUAL Model Backtest from {BACKTEST_CONFIG['BACKTEST_START_YEAR']} to {BACKTEST_CONFIG['BACKTEST_END_YEAR']} ---")
    all_predictions = []

    # --- CHANGE: Define the new baseline and the features to exclude from training ---
    baseline_feature = 'yield_trend'  # The name of our rolling average baseline
    target_col = 'trend_residual'

    # Exclude the old baseline AND the new one (which is the target) from predictors to prevent leakage.
    features_to_exclude = ['stat_trend_forecast', baseline_feature]
    actual_training_features = [col for col in full_feature_list if col in df.columns and col not in features_to_exclude]
    print(f" -> Training with {len(actual_training_features)} features.")


    for year_to_predict in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1),
                                desc="Backtesting Years"):
        train_df_full = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()

        if test_df.empty or len(train_df_full) < 100:
            continue

        # --- DATA PREPARATION WITHIN THE LOOP (NEW LOGIC) ---
        # 1. Define the baseline and target residual for the training set
        # The .shift(1) is crucial to prevent the model from seeing the current year's yield
        train_df_full[baseline_feature] = train_df_full.groupby('district_no')['kreisYield'].transform(
            lambda x: x.rolling(window=5, min_periods=3).mean().shift(1)
        )
        train_df_full[target_col] = train_df_full['kreisYield'] - train_df_full[baseline_feature]

        # 2. Drop rows where the baseline or target can't be calculated (early years)
        train_df_full.dropna(subset=[baseline_feature, target_col], inplace=True)

        # 3. Correctly calculate the baseline for the test set using only historical data
        # To do this, we combine historical data with the test data, calculate the rolling
        # average over the whole series, and then merge the result back to the test set.
        temp_df_for_baseline = pd.concat([
            df[df['year'] < year_to_predict],
            test_df
        ]).sort_values(by=['district_no', 'year'])

        temp_df_for_baseline[baseline_feature] = temp_df_for_baseline.groupby('district_no')['kreisYield'].transform(
            lambda x: x.rolling(window=5, min_periods=3).mean().shift(1)
        )

        # Merge the correctly calculated baseline onto the test_df
        test_df = pd.merge(test_df, temp_df_for_baseline[['district_no', 'year', baseline_feature]], on=['district_no', 'year'], how='left')
        test_df.dropna(subset=[baseline_feature], inplace=True)


        if train_df_full.empty or test_df.empty:
            continue

        # 3. Prepare data splits (this part is the same)
        X_train = train_df_full[actual_training_features]
        y_train = train_df_full[target_col]
        X_test = test_df[actual_training_features]

        # 4. Fit models on the residual and predict the residual for the test year (this part is the same)
        fitted_models = {name: clone(model).fit(X_train, y_train) for name, model in models.items()}
        predicted_residuals = {name: model.predict(X_test) for name, model in fitted_models.items()}

        # --- RECONSTRUCT FINAL PREDICTION (this part is the same) ---
        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name', baseline_feature]].copy()

        # Final Prediction = Baseline Forecast + Predicted Residual
        fold_results['predicted_yield_median'] = fold_results[baseline_feature] + predicted_residuals['median']
        fold_results['predicted_yield_lower'] = fold_results[baseline_feature] + predicted_residuals['lower']
        fold_results['predicted_yield_upper'] = fold_results[baseline_feature] + predicted_residuals['upper']

        all_predictions.append(fold_results)

    if not all_predictions:
        return pd.DataFrame()

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
    plt.figure(figsize=(14, 8));
    plt.plot(yearly_avg['year'], yearly_avg['avg_actual_yield'], label='National Average Actual Yield', color='navy',
             marker='o', zorder=3)
    plt.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='Median Prediction', color='red', linestyle='--',
             zorder=4)
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='red',
                     alpha=0.2, label='95% Prediction Interval', zorder=2)
    plt.title("National Average Yield vs. Predicted Interval (Rolling Avg. Residual Model)", fontsize=16);
    plt.xlabel("Year");
    plt.ylabel("Yield (dt/ha)")
    plt.legend();
    plt.grid(True, linestyle=':');
    plt.savefig(os.path.join(report_dir, '01_national_average_timeline.png'), bbox_inches='tight');
    plt.close()


def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame,
                                       report_dir: str):
    print("-> Generating Best vs. Worst District Timelines...")
    reliable_perf = district_performance[
        district_performance['data_point_count'] >= BACKTEST_CONFIG['MIN_DATAPOINTS_FOR_PLOT']]
    if len(reliable_perf) < 6: print(f"   Warning: Not enough reliable districts to plot. Skipping."); return
    best_districts, worst_districts = reliable_perf.nlargest(3, 'r2'), reliable_perf.nsmallest(3, 'r2')
    districts_to_plot = pd.concat([best_districts, worst_districts])
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, (_, district_info) in enumerate(districts_to_plot.iterrows()):
        ax = axes.flatten()[i];
        data = backtest_results[backtest_results['district_no'] == district_info['district_no']]
        ax.plot(data['year'], data['kreisYield'], label='Actual Yield', color='navy', marker='o', markersize=4,
                zorder=3)
        ax.plot(data['year'], data['predicted_yield_median'], label='Median Prediction', color='red', linestyle='--',
                zorder=4)
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'], color='red',
                        alpha=0.2, label='95% Interval')
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})");
        ax.legend();
        ax.grid(True, linestyle=':')
    plt.suptitle("Prediction Timelines for Best & Worst Districts (Rolling Avg. Residual Model)", fontsize=18);
    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    plt.savefig(os.path.join(report_dir, '02_best_worst_districts.png'), bbox_inches='tight');
    plt.close()


def plot_performance_map(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame, report_dir: str):
    print("-> Generating R-squared Map...");
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12));
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8', legend=True,
                    legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'] == True]
    if not low_data_gdf.empty: low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black',
                                 label=f'Low Data (< {BACKTEST_CONFIG["LOW_DATA_THRESHOLD"]} years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability');
    ax.set_title('Model Performance (R²) by District - Rolling Avg. Residual Model', fontsize=16);
    ax.set_axis_off();
    plt.savefig(os.path.join(report_dir, '03_r_squared_map.png'), bbox_inches='tight');
    plt.close()


def plot_r2_vs_data_count(district_performance: pd.DataFrame, report_dir: str):
    print("-> Generating R² vs. Data Count plot...");
    plt.figure(figsize=(10, 6));
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1);
    plt.axhline(0, color='grey', linestyle='--');
    plt.title("Data Availability vs Model Performance - Rolling Avg. Residual Model")
    plt.xlabel("Number of Years in Backtest per District");
    plt.ylabel("R-squared (R²)");
    plt.legend(title='Is Low Data?');
    plt.savefig(os.path.join(report_dir, '04_r2_vs_data_count.png'), bbox_inches='tight');
    plt.close()


def main():
    report_dir = Path(BACKTEST_CONFIG['REPORT_DIR'])
    report_dir.mkdir(parents=True, exist_ok=True)
    print("--- Starting STAT-TREND RESIDUAL Model Evaluation ---")

    try:
        models = {name: joblib.load(XGB_CONFIG[f'{name.upper()}_MODEL_PATH']) for name in ['lower', 'median', 'upper']}
        df = pd.read_csv(XGB_CONFIG['DATA_PATH'])
        gdf = gpd.read_file(BACKTEST_CONFIG['GEOJSON_PATH'])
        gdf.rename(columns={'id': 'district_no'}, inplace=True)
        gdf['district_no'] = gdf['district_no'].astype(str).str.zfill(5)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf[['district_no', 'name']], on='district_no', how='left')
        print("Models and data loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}");
        return

    feature_list = XGB_CONFIG['FEATURE_COLS']

    # NOTE: The Conformalized Quantile Regression (CQR) logic has been removed for simplicity.
    # This backtest now uses the direct quantile outputs.
    results = run_backtest_residual_model(df, models, feature_list)

    if results.empty:
        print("❌ Backtest did not produce results. Terminating.");
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