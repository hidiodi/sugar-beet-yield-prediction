# File: src/models/backtest_final_mapie_model.py
# Description: Backtesting and diagnostics for the STABLE RF model
#              using MAPIE for calibrated quantile intervals.

import pandas as pd
import geopandas as gpd
import joblib
from sklearn.ensemble import RandomForestRegressor  # <-- Standard RF
from mapie.regression import MapieRegressor  # <-- KEY IMPORT
from mapie.quantile_regression import MapieQuantileRegressor  # <-- Even better!
import numpy as np
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

# --- Configuration for the Final MAPIE Model ---
MODEL_PATH = os.path.join('src/models', 'final_rf_model.joblib')  # <-- Load the sklearn RF model
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
# Create a new report directory
REPORT_DIR = os.path.join('reports', 'figures', 'district_level_diagnostics', 'final_mapie_champion')

BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2024  # Let's run it up to the present
LOW_DATA_THRESHOLD = 10
MIN_DATAPOINTS_FOR_PLOT = 10
CALIBRATION_SET_SIZE = 0.15  # Use 15% of training data for calibration
ALPHA = 0.05  # This is our 1 - 95% = 5% error rate


def run_backtest(df: pd.DataFrame, feature_cols: list, model_base_clone: RandomForestRegressor):
    """Performs a rolling forecast using MAPIE to get calibrated quantiles."""
    print(f"\n--- Starting MAPIE Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")
    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Years"):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()
        if test_df.empty or len(train_df) < 50: continue  # Need enough data to train + calibrate

        X_train_cal_all = train_df[feature_cols]
        y_train_cal_all = train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        # --- KEY MAPIE LOGIC ---
        # We need a base model (for median) and two quantile models (for the interval)
        # We use MapieQuantileRegressor, which is designed for this!

        # 1. Clone the base model structure
        base_model = clone(model_base_clone)

        # 2. Wrap it with MapieQuantileRegressor (CQR method)
        # This model will be trained on a subset of data and calibrated on the rest.
        # "split" CV automatically creates a train/calibration split
        mapie_cqr = MapieQuantileRegressor(
            base_model,
            method="cqr",
            cv="split",
            test_size=CALIBRATION_SET_SIZE,  # 15% for calibration
            random_state=42
        )

        # 3. Fit MAPIE on the full training data for this fold
        mapie_cqr.fit(X_train_cal_all, y_train_cal_all, alpha=ALPHA)

        # 4. Predict quantiles on the test set
        # This returns a 3D array: [n_samples, n_quantiles, n_alpha]
        y_pis = mapie_cqr.predict(X_test)

        # We also need a separate median prediction
        # Let's train a model on ALL data for the best median
        median_model = clone(model_base_clone)
        median_model.fit(X_train_cal_all, y_train_cal_all)
        q_median = median_model.predict(X_test)

        # Extract the lower and upper bounds
        q_lower = y_pis[:, 0, 0]  # [:, 0, 0] is the lower quantile
        q_upper = y_pis[:, 1, 0]  # [:, 1, 0] is the upper quantile
        # --- END OF KEY CHANGE ---

        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name']].copy()
        # Add trend back to the detrended quantile predictions
        fold_results['predicted_yield_lower'] = q_lower + test_df['yield_trend']
        fold_results['predicted_yield_median'] = q_median + test_df['yield_trend']
        fold_results['predicted_yield_upper'] = q_upper + test_df['yield_trend']
        all_predictions.append(fold_results)

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield_median'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\nBacktest complete.")
    return results_df


# =============================================================================
# ALL FUNCTIONS BELOW ARE *IDENTICAL* TO YOUR PREVIOUS SCRIPTS
# Only the output titles in plots have been changed for clarity.
# =============================================================================

def analyze_interval_performance(results_df: pd.DataFrame):
    """Analyzes the performance of the 95% prediction interval."""
    print("\n--- Analyzing Prediction Interval Performance (Target: 95%) ---")
    results_df['is_covered'] = (results_df['kreisYield'] >= results_df['predicted_yield_lower']) & \
                               (results_df['kreisYield'] <= results_df['predicted_yield_upper'])

    coverage = results_df['is_covered'].mean()
    print(f"Prediction Interval Coverage (PICP): {coverage:.2%}")

    results_df['interval_width'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']
    avg_width = results_df['interval_width'].mean()
    print(f"Mean Prediction Interval Width (MPIW): {avg_width:.2f} dt/ha")


def calculate_district_metrics(results_df: pd.DataFrame, report_dir: str):
    """Calculates R², MAE, and data count for each district."""
    print("-> Calculating district-level metrics...")

    def r2_safe(g): return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99

    performance = results_df.groupby('district_no').apply(lambda g: pd.Series(
        {'r2': r2_safe(g), 'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
         'name': g['name'].iloc[0], 'data_point_count': len(g)})).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < LOW_DATA_THRESHOLD
    performance.to_csv(os.path.join(report_dir, 'district_level_metrics.csv'), index=False)
    return performance


# --- PLOTTING FUNCTIONS ---

def plot_national_average_timeline(backtest_results: pd.DataFrame, report_dir: str):
    """Generates a national average plot with the 95% prediction interval."""
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
    plt.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='Median Prediction', color='red', linestyle='--',
             zorder=4)
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='red',
                     alpha=0.2, label='95% Prediction Interval', zorder=2)
    plt.title("National Average Yield vs. Predicted Interval (Stable RF + MAPIE)", fontsize=16)  # <-- Title Changed
    plt.xlabel("Year");
    plt.ylabel("Yield (dt/ha)")
    plt.legend();
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(report_dir, '01_national_average_timeline.png'), bbox_inches='tight')
    plt.close()


def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame,
                                       report_dir: str):
    """Generates timeline plots for the 3 best and 3 worst reliable districts."""
    print("-> Generating Best vs. Worst District Timelines...")
    reliable_perf = district_performance[district_performance['data_point_count'] >= MIN_DATAPOINTS_FOR_PLOT]
    if len(reliable_perf) < 6:
        print(f"   Warning: Not enough reliable districts (found {len(reliable_perf)}) to plot. Skipping.")
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
        ax.fill_between(data['year'], data['predicted_yield_lower'], data['predicted_yield_upper'], color='red',
                        alpha=0.2, label='95% Interval')
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})")
        ax.legend();
        ax.grid(True, linestyle=':')
    plt.suptitle(f"Prediction Timelines for Best & Worst Districts (Stable RF + MAPIE)",
                 fontsize=18)  # <-- Title Changed
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(report_dir, '02_best_worst_districts.png'), bbox_inches='tight')
    plt.close()


def plot_performance_map(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame, report_dir: str):
    """Geographic map of R² with hatching for low-data districts."""
    print("-> Generating R-squared Map...")
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
    ax.set_title('Model Performance (R²) by District - Stable RF + MAPIE', fontsize=16);  # <-- Title Changed
    ax.set_axis_off()
    plt.savefig(os.path.join(report_dir, '03_r_squared_map.png'), bbox_inches='tight');
    plt.close()


def plot_r2_vs_data_count(district_performance: pd.DataFrame, report_dir: str):
    """Scatter plot of R² vs number of data points per district."""
    print("-> Generating R² vs. Data Count plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1);
    plt.axhline(0, color='grey', linestyle='--')
    plt.title("Data Availability vs Model Performance - Stable RF + MAPIE")  # <-- Title Changed
    plt.xlabel("Number of Years in Backtest per District");
    plt.ylabel("R-squared (R²)")
    plt.legend(title='Is Low Data?')
    plt.savefig(os.path.join(report_dir, '04_r2_vs_data_count.png'), bbox_inches='tight');
    plt.close()


def main():
    """Main function to orchestrate the Final MAPIE model evaluation."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting Final MAPIE (Stable RF) Model Evaluation Pipeline ---")

    try:
        model_base = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
        print("Model, data, and geo-data loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}");
        return

    # ENSURE this feature list is IDENTICAL to the one in the training script
    feature_cols = [
        'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
        'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
        'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
        'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
        'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
        'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast', 'spring_temp_prob_warm_forecast',
        'spring_precip_prob_wet_forecast', 'spring_solar_rad_prob_wet_forecast', 'spring_evaporation_prob_wet_forecast',
        'spring_runoff_prob_wet_forecast', 'spring_soil_temp_l1_prob_warm_forecast',
        'spring_snowfall_prob_wet_forecast',
        'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast', 'summer_solar_rad_prob_wet_forecast',
        'summer_evaporation_prob_wet_forecast', 'summer_runoff_prob_wet_forecast',
        'summer_soil_temp_l1_prob_warm_forecast', 'summer_snowfall_prob_wet_forecast', 'lat', 'lon', 'avg_elevation',
        'avg_slope', 'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm',
        'avg_bdod_0_100cm', 'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_1cm', 'avg_phh2o_0_100cm',
        'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean',
        'winter_cropland_LST_anomaly', 'winter_cropland_snow_cover_days', 'fertilizer_price_index_lag1_anomaly_capped',
        'is_fertilizer_price_extreme', 'is_summer_forecast_dry', 'gdd_x_fertilizer_price',
        'spring_temp_x_spring_precip', 'antecedent_gdd_sum_anomaly_sq', 'summer_heat_x_profit_margin',
        'summer_precip_x_input_costs', 'spring_temp_prob_warm_forecast_sq', 'summer_temp_prob_warm_forecast_sq',
        'spring_precip_prob_wet_forecast_sq', 'summer_precip_prob_wet_forecast_sq'
    ]
    # Handle a potential column name typo I saw
    if 'avg_som_0_1cm' in feature_cols:
        feature_cols[feature_cols.index('avg_som_0_1cm')] = 'avg_som_0_100cm'

    print("\n--- Applying Causal Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    backtest_results = run_backtest(df, feature_cols, model_base)
    if backtest_results.empty:
        print("❌ Backtest did not produce results. Terminating.");
        return

    # --- Analysis & Reporting (This part is unchanged) ---
    analyze_interval_performance(backtest_results)
    district_performance = calculate_district_metrics(backtest_results, REPORT_DIR)

    plot_national_average_timeline(backtest_results, REPORT_DIR)
    plot_best_worst_district_timelines(district_performance, backtest_results, REPORT_DIR)
    plot_performance_map(district_performance, gdf_districts, REPORT_DIR)
    plot_r2_vs_data_count(district_performance, REPORT_DIR)

    print("\n--- Overall Performance Summary (All Districts, Median Prediction) ---")
    r2_total = r2_score(backtest_results['kreisYield'], backtest_results['predicted_yield_median'])
    mae_total = backtest_results['abs_error'].mean()
    print(f"  R-squared (R²): {r2_total:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae_total:.2f} dt/ha")
    print("\n--- Evaluation Complete ---")
    print(f"All reports and plots saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()