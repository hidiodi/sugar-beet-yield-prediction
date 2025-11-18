# File: src/02_models/XGBoost/regression_model/Testing/backtest_final_quantile_model.py
# Description: Backtests the ROBUST Risk-Based Residual Model.
# VERSION: 5.0 (Risk-Cone Backtest)

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


def run_backtest_residual_model(df: pd.DataFrame, models: dict, feature_cols: list) -> pd.DataFrame:
    """
    Runs a time-series backtest.
    Strategy:
      1. Baseline = 5-Year Rolling Average (Shifted 1 year).
      2. Model Predicts = Residual (Actual - Baseline).
      3. Final Prediction = Baseline + Predicted Residual.
    """
    print(
        f"\n--- Starting Risk-Based Backtest from {BACKTEST_CONFIG['BACKTEST_START_YEAR']} to {BACKTEST_CONFIG['BACKTEST_END_YEAR']} ---")

    # --- 1. GLOBAL PRE-PROCESSING (Safe due to shift=1) ---
    # We calculate the baseline anchor (Rolling Average) for the whole dataset first.
    # Since it is shifted by 1, the value for Year X only depends on X-1, X-2...
    # This prevents leakage without needing complex loop logic.

    baseline_col = 'yield_rolling_trend'
    target_col = 'trend_residual'

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # 5-Year Rolling Average (Lagged)
    df[baseline_col] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=3).mean().shift(1)
    )

    # The Target the model learns to predict
    df[target_col] = df['kreisYield'] - df[baseline_col]

    # Drop strictly invalid rows for training (where we don't have a history yet)
    # But keep them in 'df' generally, filter later
    valid_mask = df[baseline_col].notna() & df[target_col].notna() & df['stat_trend_forecast'].notna()

    print(f" -> Backtesting with {len(feature_cols)} features.")
    print(f" -> Features: {feature_cols}")

    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1),
                                desc="Backtesting Years"):

        # TRAIN: All valid data strictly BEFORE the target year
        train_df = df[(df['year'] < year_to_predict) & valid_mask].copy()

        # TEST: The target year (Must have baseline available to make a prediction)
        test_df = df[(df['year'] == year_to_predict) & df[baseline_col].notna()].copy()

        if test_df.empty or len(train_df) < 100:
            continue

        # Prepare Features
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]

        # Sanity Check: Ensure 'stat_trend_forecast' is in features if expected
        if 'stat_trend_forecast' not in X_train.columns:
            raise ValueError(
                "CRITICAL: 'stat_trend_forecast' is missing from backtest features. It is required as an anchor.")

        # Retrain Models on Expanding Window
        # We use clone() to reset the model weights every year
        fitted_models = {name: clone(model).fit(X_train, y_train) for name, model in models.items()}

        # Predict Residuals
        pred_residuals = {name: model.predict(X_test) for name, model in fitted_models.items()}

        # Reconstruct Final Yield Prediction
        # Prediction = Rolling_Trend (Baseline) + Predicted_Residual
        fold_results = test_df[
            ['district_no', 'year', 'kreisYield', 'name', baseline_col, 'stat_trend_forecast']].copy()

        for name in ['lower', 'median', 'upper']:
            fold_results[f'predicted_yield_{name}'] = fold_results[baseline_col] + pred_residuals[name]

        all_predictions.append(fold_results)

    if not all_predictions:
        return pd.DataFrame()

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield_median'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()

    # Clip negative predictions to 0 (Physical impossibility)
    for col in ['predicted_yield_lower', 'predicted_yield_median', 'predicted_yield_upper']:
        results_df[col] = results_df[col].clip(lower=0)

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
        avg_pred_lower=('predicted_yield_lower', 'mean'), avg_pred_upper=('predicted_yield_upper', 'mean'),
        avg_stat_trend=('stat_trend_forecast', 'mean')
    ).reset_index()

    plt.figure(figsize=(14, 8))

    # Plot Actual
    plt.plot(yearly_avg['year'], yearly_avg['avg_actual_yield'], label='Actual Yield', color='navy', marker='o',
             zorder=3, linewidth=2)

    # Plot Statistical Trend (The Anchor)
    plt.plot(yearly_avg['year'], yearly_avg['avg_stat_trend'], label='Statistical Trend (Baseline)', color='gray',
             linestyle=':', zorder=2)

    # Plot Model Prediction
    plt.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='Risk-Adjusted Prediction', color='darkorange',
             linestyle='--', linewidth=2, zorder=4)

    # Plot Interval
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='orange',
                     alpha=0.2, label='90% Risk Cone', zorder=1)

    plt.title("National Average Yield: Actual vs Risk-Adjusted Forecast", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Yield (dt/ha)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(report_dir, '01_national_average_timeline.png'), bbox_inches='tight')
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
        ax = axes.flatten()[i]
        data = backtest_results[backtest_results['district_no'] == district_info['district_no']]
        ax.plot(data['year'], data['kreisYield'], label='Actual', color='navy', marker='o', markersize=4)
        ax.plot(data['year'], data['predicted_yield_median'], label='Predicted', color='red', linestyle='--')
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'], color='red',
                        alpha=0.2)
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})")
        ax.legend()
        ax.grid(True, linestyle=':')
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
    plt.title('Model Performance (R²) by District', fontsize=16)
    plt.savefig(os.path.join(report_dir, '03_r_squared_map.png'), bbox_inches='tight')
    plt.close()


def plot_r2_vs_data_count(district_performance: pd.DataFrame, report_dir: str):
    print("-> Generating R² vs. Data Count plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.axhline(0, color='grey', linestyle='--')
    plt.title("Data Availability vs Model Performance")
    plt.xlabel("Backtest Years per District")
    plt.ylabel("R-squared")
    plt.savefig(os.path.join(report_dir, '04_r2_vs_data_count.png'), bbox_inches='tight')
    plt.close()


def main():
    report_dir = Path(BACKTEST_CONFIG['REPORT_DIR'])
    report_dir.mkdir(parents=True, exist_ok=True)
    print("--- Starting Risk-Based Residual Model Evaluation ---")

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
        print(f"❌ CRITICAL ERROR during loading. Details: {e}")
        return

    # Define robust feature set (Must match Training Script)
    # We explicitly INCLUDE 'stat_trend_forecast' as it is the anchor feature.
    # We explicitly EXCLUDE identifiers and targets.

    # Base candidates from config
    candidates = XGB_CONFIG['FEATURE_COLS']

    # Hardcoded safe list to ensure we use exactly what the Risk Model expects
    # (Aligning with the training script logic)
    safe_features = [
        'stat_trend_forecast', 'national_avg_yield_lag1',
        'trend_vs_phys_gap', 'wofost_esp_std', 'wofost_esp_p10', 'wofost_skew',
        'antecedent_precip_sum', 'antecedent_gdd_sum_anomaly', 'winter_cropland_ndvi_anomaly',
        'fertilizer_price_index_lag1', 'producer_price_index_lag1',
        'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_elevation'
    ]

    # Intersect with what is actually in the DF
    final_features = [col for col in safe_features if col in df.columns]

    missing_features = set(safe_features) - set(final_features)
    if missing_features:
        print(f"⚠️ Warning: The following features were expected but missing: {missing_features}")

    # Run Backtest
    results = run_backtest_residual_model(df, models, final_features)

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