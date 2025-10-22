# File: src/models/backtest_conformalized_ensemble.py
# Description: The COMPLETE and FINAL backtest for the Conformalized Deep Ensemble.
# This version includes all standard diagnostic plots in addition to the new uncertainty maps.

import pandas as pd
import numpy as np
import geopandas as gpd
from xgboost import XGBRegressor
import os
import warnings
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = 'reports/figures/district_level_diagnostics/conformalized_deep_ensemble_champion'

# --- Model Parameters ---
N_ENSEMBLE_MODELS = 10
BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2024
CALIBRATION_WINDOW_YEARS = 3
TARGET_COVERAGE = 0.97

BASE_PARAMS = {
    'n_estimators': 914, 'learning_rate': 0.026114, 'max_depth': 5,
    'subsample': 0.922850, 'colsample_bytree': 0.811573, 'gamma': 1.830853,
    'min_child_weight': 2, 'n_jobs': -1
}
BASE_MODEL = XGBRegressor(**BASE_PARAMS)

FEATURE_COLS = [  # Must be identical to all previous models
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly',
    'antecedent_gdd_sum_anomaly', 'spring_temp_anomaly_forecast',
    'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
    'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast',
    'spring_soil_temp_l1_anomaly_forecast', 'spring_snowfall_anomaly_forecast',
    'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast',
    'summer_runoff_anomaly_forecast', 'summer_soil_temp_l1_anomaly_forecast',
    'summer_snowfall_anomaly_forecast', 'spring_temp_prob_warm_forecast',
    'spring_precip_prob_wet_forecast', 'spring_solar_rad_prob_wet_forecast',
    'spring_evaporation_prob_wet_forecast', 'spring_runoff_prob_wet_forecast',
    'spring_soil_temp_l1_prob_warm_forecast', 'spring_snowfall_prob_wet_forecast',
    'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',
    'summer_solar_rad_prob_wet_forecast', 'summer_evaporation_prob_wet_forecast',
    'summer_runoff_prob_wet_forecast', 'summer_soil_temp_l1_prob_warm_forecast',
    'summer_snowfall_prob_wet_forecast', 'lat', 'lon', 'avg_elevation',
    'avg_slope', 'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm',
    'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'avg_bdod_0_100cm',
    'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm',
    'avg_phh2o_0_100cm', 'winter_cropland_ndvi_mean',
    'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean',
    'winter_cropland_LST_anomaly', 'winter_cropland_snow_cover_days',
    'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',
    'is_summer_forecast_dry', 'gdd_x_fertilizer_price',
    'spring_temp_x_spring_precip', 'antecedent_gdd_sum_anomaly_sq',
    'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
    'spring_temp_prob_warm_forecast_sq', 'summer_temp_prob_warm_forecast_sq',
    'spring_precip_prob_wet_forecast_sq', 'summer_precip_prob_wet_forecast_sq'
]


def run_conformalized_ensemble_backtest(df: pd.DataFrame):
    """Performs a rolling backtest with a fully retrained, calibrated ensemble at each step."""
    print("--- Starting Conformalized Deep Ensemble Backtest ---")
    print("WARNING: This process is extremely computationally intensive and may take several hours.")
    all_predictions = []

    start_year = df['year'].min() + CALIBRATION_WINDOW_YEARS
    if BACKTEST_START_YEAR < start_year:
        print(f"Adjusting start year to {start_year} to allow for initial calibration.")

    for year_to_predict in tqdm(range(max(BACKTEST_START_YEAR, start_year), BACKTEST_END_YEAR + 1),
                                desc="Backtesting Year"):

        # 1. --- Data Splitting for this Fold ---
        historical_df = df[df['year'] < year_to_predict]
        test_df = df[df['year'] == year_to_predict]

        calib_start_year = year_to_predict - CALIBRATION_WINDOW_YEARS
        proper_train_df = historical_df[historical_df['year'] < calib_start_year]
        calib_df = historical_df[historical_df['year'] >= calib_start_year]

        if test_df.empty or proper_train_df.empty or calib_df.empty: continue

        # 2. --- Train Full Ensemble on Proper Training Data ---
        ensemble_models = {'lower': [], 'median': [], 'upper': []}
        for i in range(N_ENSEMBLE_MODELS):
            bootstrap_train_df = proper_train_df.sample(frac=1.0, replace=True, random_state=i)
            X_train_boot = bootstrap_train_df[FEATURE_COLS]
            y_train_boot = bootstrap_train_df['kreisYield_detrended']
            for name, alpha in [('lower', 0.025), ('median', 0.5), ('upper', 0.975)]:
                model = clone(BASE_MODEL)
                model.set_params(objective='reg:quantileerror', quantile_alpha=alpha, random_state=i)
                model.fit(X_train_boot, y_train_boot)
                ensemble_models[name].append(model)

        # 3. --- Calibrate on Hold-Out Calibration Data ---
        X_calib = calib_df[FEATURE_COLS]
        y_calib_detrended = calib_df['kreisYield_detrended'].values

        calib_preds = {name: [] for name in ['lower', 'median', 'upper']}
        for name, models in ensemble_models.items():
            for model in models:
                calib_preds[name].append(model.predict(X_calib))

        mean_lower_calib = np.mean(calib_preds['lower'], axis=0)
        mean_upper_calib = np.mean(calib_preds['upper'], axis=0)
        epistemic_calib = np.std(calib_preds['median'], axis=0)

        raw_errors_calib = np.maximum(mean_lower_calib - y_calib_detrended, y_calib_detrended - mean_upper_calib)
        normalized_scores = raw_errors_calib / (epistemic_calib + 1e-6)

        n_calib = len(calib_df)
        alpha = 1 - TARGET_COVERAGE
        q_level = min(np.ceil((1 - alpha) * (n_calib + 1)) / n_calib, 1.0)
        q_multiplier_fold = np.quantile(normalized_scores, q_level)

        # 4. --- Predict on Test Data ---
        X_test = test_df[FEATURE_COLS]

        test_preds = {name: [] for name in ['lower', 'median', 'upper']}
        for name, models in ensemble_models.items():
            for model in models:
                test_preds[name].append(model.predict(X_test))

        mean_lower_test = np.mean(test_preds['lower'], axis=0)
        mean_upper_test = np.mean(test_preds['upper'], axis=0)
        mean_median_test = np.mean(test_preds['median'], axis=0)
        epistemic_test = np.std(test_preds['median'], axis=0)

        adaptive_adjustment = q_multiplier_fold * epistemic_test

        final_lower = (mean_lower_test - adaptive_adjustment) + test_df['yield_trend'].values
        final_upper = (mean_upper_test + adaptive_adjustment) + test_df['yield_trend'].values
        final_median = mean_median_test + test_df['yield_trend'].values

        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name']].copy()
        fold_results['predicted_yield_lower'] = final_lower
        fold_results['predicted_yield_median'] = final_median
        fold_results['predicted_yield_upper'] = final_upper
        fold_results['epistemic_uncertainty'] = epistemic_test
        all_predictions.append(fold_results)

    return pd.concat(all_predictions, ignore_index=True)


# ADDED BACK: Function to calculate per-district metrics needed for plots
def calculate_district_metrics(results_df: pd.DataFrame):
    """Calculates R², MAE, and data count for each district."""
    print("-> Calculating district-level metrics...")

    def r2_safe(g): return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99

    performance = results_df.groupby('district_no').apply(lambda g: pd.Series({
        'r2': r2_safe(g),
        'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
        'name': g['name'].iloc[0],
        'data_point_count': len(g)
    })).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < 10
    return performance


# ADDED BACK: Standard plotting functions, updated for the new model context
def plot_national_average_timeline(backtest_results: pd.DataFrame):
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
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='purple',
                     alpha=0.25, label='95% Conformalized Ensemble Interval', zorder=2)
    plt.title("National Average Yield vs. Predicted Interval (Conformalized Deep Ensemble)", fontsize=16)
    plt.xlabel("Year");
    plt.ylabel("Yield (dt/ha)")
    plt.legend();
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(REPORT_DIR, '03_national_average_timeline.png'), bbox_inches='tight')
    plt.close()


def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame):
    print("-> Generating Best vs. Worst District Timelines...")
    reliable_perf = district_performance[~district_performance['is_low_data']]
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
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'], color='purple',
                        alpha=0.25, label='95% Conformalized Interval')
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})")
        ax.legend();
        ax.grid(True, linestyle=':')
    plt.suptitle("Prediction Timelines for Best & Worst Districts (Conformalized Deep Ensemble)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(REPORT_DIR, '04_best_worst_districts.png'), bbox_inches='tight')
    plt.close()


def plot_performance_map(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    print("-> Generating R-squared Map...")
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data']]
    if not low_data_gdf.empty:
        low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black', label='Low Data (< 10 years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability')
    ax.set_title('Model Performance (R²) by District - Conformalized Deep Ensemble', fontsize=16)
    ax.set_axis_off()
    plt.savefig(os.path.join(REPORT_DIR, '05_r_squared_map.png'), bbox_inches='tight')
    plt.close()


def analyze_and_report(results_df: pd.DataFrame, gdf: gpd.GeoDataFrame):
    """Calculates final metrics and generates ALL planner-focused visualizations."""
    print("\n--- Analyzing Conformalized Ensemble Performance ---")
    os.makedirs(REPORT_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(REPORT_DIR, 'conformalized_ensemble_results.csv'), index=False)

    # --- Performance Metrics ---
    results_df['is_covered'] = (results_df['kreisYield'] >= results_df['predicted_yield_lower']) & \
                               (results_df['kreisYield'] <= results_df['predicted_yield_upper'])
    results_df['interval_width'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']

    picp = results_df['is_covered'].mean()
    mpiw = results_df['interval_width'].mean()
    r2 = r2_score(results_df['kreisYield'], results_df['predicted_yield_median'])
    mae = mean_absolute_error(results_df['kreisYield'], results_df['predicted_yield_median'])

    print(f"Prediction Interval Coverage (PICP): {picp:.2%}")
    print(f"Mean Prediction Interval Width (MPIW): {mpiw:.2f} dt/ha")
    print(f"Overall R-squared (R²): {r2:.4f}")
    print(f"Overall Mean Absolute Error (MAE): {mae:.2f} dt/ha")

    # --- Uncertainty Analysis ---
    avg_epistemic = results_df['epistemic_uncertainty'].mean()
    width_std = results_df['interval_width'].std()
    print(f"\nAverage Epistemic Uncertainty (Model Ignorance): {avg_epistemic:.2f} dt/ha")
    print(f"Standard Deviation of Final Interval Widths: {width_std:.2f} dt/ha (Higher is more adaptive)")

    # --- Generate NEW Planner-Focused Visualizations ---
    print("\n-> Generating new uncertainty maps...")
    district_uncertainty = results_df.groupby('district_no')['epistemic_uncertainty'].mean().reset_index()
    merged_gdf = gdf.merge(district_uncertainty, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='epistemic_uncertainty', cmap='Reds', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True,
                    legend_kwds={'label': "Average Epistemic Uncertainty (dt/ha)", 'orientation': "horizontal"})
    ax.set_title('Model Ignorance by District (Where is the model guessing?)', fontsize=16)
    ax.set_axis_off()
    plt.savefig(os.path.join(REPORT_DIR, '01_epistemic_uncertainty_map.png'), bbox_inches='tight')
    plt.close()

    district_width = results_df.groupby('district_no')['interval_width'].mean().reset_index()
    merged_gdf_width = gdf.merge(district_width, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf_width.plot(column='interval_width', cmap='Oranges', linewidth=0.5, ax=ax, edgecolor='0.8',
                          legend=True,
                          legend_kwds={'label': "Average Final Interval Width (dt/ha)", 'orientation': "horizontal"})
    ax.set_title('Total Forecast Uncertainty by District (Final Actionable Risk)', fontsize=16)
    ax.set_axis_off()
    plt.savefig(os.path.join(REPORT_DIR, '02_total_uncertainty_map.png'), bbox_inches='tight')
    plt.close()

    # --- Generate STANDARD Diagnostic Visualizations ---
    print("\n-> Generating standard diagnostic plots...")
    district_performance = calculate_district_metrics(results_df)
    plot_national_average_timeline(results_df)
    plot_best_worst_district_timelines(district_performance, results_df)
    plot_performance_map(district_performance, gdf)


def main():
    """Main function to orchestrate the Conformalized Deep Ensemble backtest."""
    df = pd.read_csv(DATA_PATH)
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
    gdf['district_no'] = gdf['district_no'].astype(str).str.zfill(5)
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)

    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    df = pd.merge(df, gdf[['district_no', 'name']], on='district_no', how='left')

    results = run_conformalized_ensemble_backtest(df)
    analyze_and_report(results, gdf)

    print(f"\n--- Evaluation Complete ---")
    print(f"All reports and plots saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()