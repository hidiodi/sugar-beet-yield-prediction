# File: src/models/backtesting_hybrid.py
# Description: A deep-dive script to evaluate the TWO-STAGE HYBRID model.
# This script is hardened against data leakage and analyzes the reliability
# of the dynamically generated prediction intervals.

import pandas as pd
import geopandas as gpd
from xgboost import XGBRegressor
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = os.path.join('reports', 'figures', 'district_level_diagnostics_hybrid')
BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2021

# Use the same hyperparameters for both models
BEST_PARAMS = {
    'colsample_bytree': 0.8223306320976561,
    'learning_rate': 0.020282652208788696,
    'max_depth': 4,
    'n_estimators': 448,
    'subsample': 0.8049549320516778
}
feature_cols = [  # Ensure this matches your final feature set
    'avg_elevation', 'avg_soil_pawc', 'lon', 'lat', 'profit_margin_proxy_lag1',
    'cost_of_inputs_lag1', 'producer_price_index_lag1_anomaly',
    'seed_price_index_lag1_anomaly', 'energy_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly', 'plant_protection_price_index_lag1_anomaly',
    'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast',
    'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast',
    'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',
    'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
    'spring_temp_anomaly_forecast_sq', 'summer_temp_anomaly_forecast_sq',
    'spring_precip_anomaly_forecast_sq', 'summer_precip_anomaly_forecast_sq'
]


def run_hybrid_backtest(df: pd.DataFrame, model_median_template: XGBRegressor, model_error_template: XGBRegressor):
    """Performs a rolling backtest of the entire two-stage system."""
    print(f"\n--- Starting Hybrid Model Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")
    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Years"):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()

        if test_df.empty or train_df.empty: continue

        X_train, y_train_detrended = train_df[feature_cols], train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        # --- Stage 1: Train Median Model ---
        model_median = clone(model_median_template)
        model_median.fit(X_train, y_train_detrended)

        # --- Stage 2: Train Error Model ---
        error_targets = abs(y_train_detrended - model_median.predict(X_train))
        model_error = clone(model_error_template)
        model_error.fit(X_train, error_targets)

        # --- Predict on Test Data ---
        predicted_median_detrended = model_median.predict(X_test)
        predicted_error = model_error.predict(X_test)
        predicted_error[predicted_error < 0] = 0

        # --- Construct Final Predictions and Interval ---
        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name', 'yield_trend']].copy()
        fold_results['predicted_yield_median'] = predicted_median_detrended + test_df['yield_trend']
        fold_results['predicted_yield_lower'] = fold_results['predicted_yield_median'] - predicted_error
        fold_results['predicted_yield_upper'] = fold_results['predicted_yield_median'] + predicted_error
        all_predictions.append(fold_results)

    if not all_predictions: return pd.DataFrame()

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['deviation'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']
    print("\n✅ Hybrid backtest complete.")
    return results_df


def analyze_yearly_performance_hybrid(results_df: pd.DataFrame):
    """Calculates and plots key hybrid model metrics for each year."""
    print("Calculating yearly performance and uncertainty calibration...")

    def coverage_metric(g):
        in_interval = ((g['kreisYield'] >= g['predicted_yield_lower']) & (
                    g['kreisYield'] <= g['predicted_yield_upper'])).mean()
        return in_interval

    yearly_perf = results_df.groupby('year').apply(
        lambda g: pd.Series({
            'r2_median': r2_score(g['kreisYield'], g['predicted_yield_median']),
            'avg_deviation': g['deviation'].mean(),
            'coverage': coverage_metric(g)
        })
    ).reset_index()

    print("--- Yearly Hybrid Performance Summary ---")
    print(yearly_perf)
    print("-----------------------------------------")

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

    plt.title('Hybrid Model Performance and Uncertainty Over Time', fontsize=16)
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    save_path = os.path.join(REPORT_DIR, '01_hybrid_performance_over_time.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Yearly hybrid performance plot saved to {save_path}")
    return yearly_perf


def main():
    """Main function to orchestrate the hybrid model evaluation pipeline."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting Hybrid Model Backtesting Pipeline ---")

    try:
        df = pd.read_csv(DATA_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)  # <<< FIX: This line was present but unused

        # <<< FIX: Prepare and merge the district names into the main dataframe
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')

        print("✅ Data and geo-data loaded and merged successfully.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}")
        return

    # --- 1. Causal Detrending (Hardened against data leakage) ---
    print("\n--- Applying Causal (Trailing Mean) Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x)
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    # --- 2. Create Model Templates ---
    model_median_template = XGBRegressor(objective='reg:squarederror', **BEST_PARAMS, random_state=42, n_jobs=-1)
    model_error_template = XGBRegressor(objective='reg:squarederror', **BEST_PARAMS, random_state=42, n_jobs=-1)

    # --- 3. Run the Backtest ---
    backtest_results = run_hybrid_backtest(df, model_median_template, model_error_template)

    if backtest_results.empty:
        print("❌ Backtest did not produce results. Terminating.")
        return

    # --- 4. Run Analysis ---
    yearly_performance = analyze_yearly_performance_hybrid(backtest_results)

    print("\n--- Hybrid Backtesting Complete ---")
    # Calculate and print overall metrics
    overall_r2 = r2_score(backtest_results['kreisYield'], backtest_results['predicted_yield_median'])
    overall_mae = mean_absolute_error(backtest_results['kreisYield'], backtest_results['predicted_yield_median'])
    overall_coverage = ((backtest_results['kreisYield'] >= backtest_results['predicted_yield_lower']) & (
                backtest_results['kreisYield'] <= backtest_results['predicted_yield_upper'])).mean()

    print("\n--- Overall Hybrid Model Performance Summary ---")
    print(f"  Overall R-squared (R²):     {overall_r2:.4f}")
    print(f"  Overall Mean Absolute Error:  {overall_mae:.2f} dt/ha")
    print(f"  Overall Interval Coverage:    {overall_coverage:.2%}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()