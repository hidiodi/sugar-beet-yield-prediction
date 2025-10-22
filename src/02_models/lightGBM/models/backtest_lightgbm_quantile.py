# File: src/models/backtest_lightgbm_quantile.py
# Description: Definitive backtesting and diagnostic script for the LightGBM
#              multi-model quantile champion.

import pandas as pd
import geopandas as gpd
import joblib
import lightgbm as lgb
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

# --- Configuration for the LightGBM Champion Model ---
LOWER_MODEL_PATH = os.path.join('src/models', 'final_lightgbm_model_lower.joblib')
MEDIAN_MODEL_PATH = os.path.join('src/models', 'final_lightgbm_model_median.joblib')
UPPER_MODEL_PATH = os.path.join('src/models', 'final_lightgbm_model_upper.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = os.path.join('reports', 'figures', 'district_level_diagnostics', 'lightgbm_quantile_champion')

BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2024
LOW_DATA_THRESHOLD = 10
MIN_DATAPOINTS_FOR_PLOT = 5


def run_backtest(df: pd.DataFrame, feature_cols: list, model_lower_clone: lgb.LGBMRegressor,
                 model_median_clone: lgb.LGBMRegressor, model_upper_clone: lgb.LGBMRegressor):
    """Performs a rolling forecast for the three LightGBM quantile models."""
    print(f"\n--- Starting LightGBM Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")
    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Years"):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()
        if test_df.empty or train_df.empty: continue

        X_train, y_train = train_df[feature_cols], train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        model_lower, model_median, model_upper = clone(model_lower_clone), clone(model_median_clone), clone(
            model_upper_clone)
        model_lower.fit(X_train, y_train)
        model_median.fit(X_train, y_train)
        model_upper.fit(X_train, y_train)

        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name']].copy()
        fold_results['predicted_yield_lower'] = model_lower.predict(X_test) + test_df['yield_trend']
        fold_results['predicted_yield_median'] = model_median.predict(X_test) + test_df['yield_trend']
        fold_results['predicted_yield_upper'] = model_upper.predict(X_test) + test_df['yield_trend']
        all_predictions.append(fold_results)

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield_median'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\nBacktest complete.")
    return results_df


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
    plt.title("National Average Yield vs. Predicted Interval (LightGBM Quantile Model)", fontsize=16)
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
    plt.suptitle(f"Prediction Timelines for Best & Worst Districts (LightGBM Quantile Model)", fontsize=18)
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
    ax.set_title('Model Performance (R²) by District - LightGBM Quantile Model', fontsize=16);
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
    plt.title("Data Availability vs Model Performance - LightGBM Quantile Model")
    plt.xlabel("Number of Years in Backtest per District");
    plt.ylabel("R-squared (R²)")
    plt.legend(title='Is Low Data?')
    plt.savefig(os.path.join(report_dir, '04_r2_vs_data_count.png'), bbox_inches='tight');
    plt.close()


def main():
    """Main function to orchestrate the LightGBM Champion model evaluation."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting LightGBM Champion Model Evaluation Pipeline ---")

    try:
        model_lower = joblib.load(LOWER_MODEL_PATH)
        model_median = joblib.load(MEDIAN_MODEL_PATH)
        model_upper = joblib.load(UPPER_MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
        print("Models, data, and geo-data loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}");
        return

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
        'avg_bdod_0_100cm', 'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm', 'avg_phh2o_0_100cm',
        'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean',
        'winter_cropland_LST_anomaly', 'winter_cropland_snow_cover_days', 'fertilizer_price_index_lag1_anomaly_capped',
        'is_fertilizer_price_extreme', 'is_summer_forecast_dry', 'gdd_x_fertilizer_price',
        'spring_temp_x_spring_precip', 'antecedent_gdd_sum_anomaly_sq', 'summer_heat_x_profit_margin',
        'summer_precip_x_input_costs', 'spring_temp_prob_warm_forecast_sq', 'summer_temp_prob_warm_forecast_sq',
        'spring_precip_prob_wet_forecast_sq', 'summer_precip_prob_wet_forecast_sq'
    ]

    print("\n--- Applying Causal Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    backtest_results = run_backtest(df, feature_cols, model_lower, model_median, model_upper)
    if backtest_results.empty:
        print("❌ Backtest did not produce results. Terminating.");
        return

    # --- Analysis & Reporting ---
    analyze_interval_performance(backtest_results)
    district_performance = calculate_district_metrics(backtest_results, REPORT_DIR)

    # Generate all diagnostic plots
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