# File: src/02_models/XGBoost/regression_model/Testing/backtest_standalone_xgb_model.py
# REFACTORED (v7.0): Standalone Risk-Based Backtest
# Description:
#   Backtests the Standalone Model using the LinearGAM Baseline.

import pandas as pd
import geopandas as gpd
import joblib
from xgboost import XGBRegressor
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from tqdm import tqdm
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


def run_backtest_standalone(df: pd.DataFrame, models: dict, feature_cols: list) -> pd.DataFrame:
    print(f"\n--- Starting Standalone Backtest (Risk/Residual Strategy) ---")

    # 1. Global Baseline Calculation
    # Using the LinearGAM trend (Stat Trend) as the anchor
    baseline_col = 'stat_trend_forecast'
    target_col = 'trend_residual'

    # Calculate Target (Residual)
    df[target_col] = df['kreisYield'] - df[baseline_col]

    # Valid Mask
    valid_mask = df[baseline_col].notna() & df[target_col].notna()

    print(f" -> Backtesting with {len(feature_cols)} features.")

    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1),
                                desc="Backtesting Years"):

        # TRAIN: All valid data strictly BEFORE the target year
        train_df = df[(df['year'] < year_to_predict) & valid_mask].copy()

        # TEST: The target year
        test_df = df[(df['year'] == year_to_predict) & df[baseline_col].notna()].copy()

        if test_df.empty or len(train_df) < 50:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]

        # Retrain Models (Clone to reset)
        # We retrain every year to mimic a real production environment
        fold_preds = test_df[['district_no', 'year', 'kreisYield', 'name', baseline_col]].copy()

        for name in ['lower', 'median', 'upper']:
            model = clone(models[name])
            model.fit(X_train, y_train)
            pred_residual = model.predict(X_test)

            # Final = Baseline + Residual
            fold_preds[f'predicted_yield_{name}'] = fold_preds[baseline_col] + pred_residual
            fold_preds[f'predicted_yield_{name}'] = fold_preds[f'predicted_yield_{name}'].clip(lower=0)

        all_predictions.append(fold_preds)

    if not all_predictions:
        return pd.DataFrame()

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield_median'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()

    print("\nBacktest complete.")
    return results_df


# --- Plotting Functions ---

def analyze_interval_performance(results_df: pd.DataFrame):
    print(f"\n--- Analyzing Prediction Interval Performance (Target: {BACKTEST_CONFIG['NOMINAL_COVERAGE']:.0%}) ---")
    results_df['is_covered'] = (results_df['kreisYield'] >= results_df['predicted_yield_lower']) & \
                               (results_df['kreisYield'] <= results_df['predicted_yield_upper'])
    coverage = results_df['is_covered'].mean()
    print(f"Prediction Interval Coverage (PICP): {coverage:.2%}")
    results_df['interval_width'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']
    print(f"Mean Prediction Interval Width (MPIW): {results_df['interval_width'].mean():.2f} dt/ha")


def plot_national_average_timeline(backtest_results: pd.DataFrame, report_dir: str):
    print("-> Generating National Average Prediction Timeline...")
    yearly_avg = backtest_results.groupby('year').agg(
        avg_actual_yield=('kreisYield', 'mean'),
        avg_pred_median=('predicted_yield_median', 'mean'),
        avg_pred_lower=('predicted_yield_lower', 'mean'),
        avg_pred_upper=('predicted_yield_upper', 'mean')
    ).reset_index()
    plt.figure(figsize=(14, 8))
    plt.plot(yearly_avg['year'], yearly_avg['avg_actual_yield'], label='Actual Yield', color='navy', marker='o',
             zorder=3)
    plt.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='Standalone Prediction', color='mediumorchid',
             linestyle='--', zorder=4)
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'],
                     color='mediumorchid', alpha=0.2, label='90% Interval', zorder=2)
    plt.title("National Average Yield vs. Predicted Interval (Standalone XGB)", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Yield (dt/ha)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(report_dir, '01_national_average_timeline.png'), bbox_inches='tight')
    plt.close()


def main():
    report_dir = Path(BACKTEST_CONFIG['REPORT_DIR'])
    report_dir.mkdir(parents=True, exist_ok=True)
    print("--- Starting Standalone XGB Evaluation ---")

    try:
        models = {name: joblib.load(XGB_CONFIG[f'{name.upper()}_MODEL_PATH']) for name in ['lower', 'median', 'upper']}
        df = pd.read_csv(XGB_CONFIG['DATA_PATH'])
        gdf = gpd.read_file(BACKTEST_CONFIG['GEOJSON_PATH']).rename(columns={'id': 'district_no'})
        gdf['district_no'] = gdf['district_no'].astype(str).str.zfill(5)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf[['district_no', 'name']], on='district_no', how='left')
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return

    # Use robust feature list
    feature_list = [col for col in XGB_CONFIG['FEATURE_COLS'] if col in df.columns]

    # Add new mechanisms if present but not in config
    new_mechanisms = ['nitrogen_leaching_index', 'toxic_carryover_index', 'vector_pressure_local', 'wofost_skew']
    for f in new_mechanisms:
        if f in df.columns and f not in feature_list:
            feature_list.append(f)

    results = run_backtest_standalone(df, models, feature_list)

    if results.empty:
        print("❌ Backtest failed.")
        return

    results_path = report_dir / 'full_backtest_predictions.csv'
    results.to_csv(results_path, index=False)
    print(f"\n✅ Saved results to {results_path}")

    analyze_interval_performance(results)
    plot_national_average_timeline(results, str(report_dir))

    print("\n--- Overall Performance Summary ---")
    print(f"  R-squared (R²): {r2_score(results['kreisYield'], results['predicted_yield_median']):.4f}")
    print(f"  Mean Absolute Error (MAE): {results['abs_error'].mean():.2f} dt/ha")
    print("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    main()