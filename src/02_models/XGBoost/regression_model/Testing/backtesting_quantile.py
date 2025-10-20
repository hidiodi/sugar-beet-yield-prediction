# File: src/models/backtesting_quantile.py
# Description: A deep-dive diagnostic script to evaluate the V3 QUANTILE model's performance,
#              which uses an enhanced feature set and a 95% prediction interval.

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

# --- V3 Configuration: Updated for the new models and reports ---
LOWER_MODEL_PATH = os.path.join('src/models', 'final_xgb_model_lower.joblib')
MEDIAN_MODEL_PATH = os.path.join('src/models', 'final_xgb_model_median.joblib')
UPPER_MODEL_PATH = os.path.join('src/models', 'final_xgb_model_upper.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
# Save V3 results to a new directory to compare with the previous run
REPORT_DIR = os.path.join('reports', 'figures', 'district_level_diagnostics', 'quantile_model_diagnostics')

BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2024
LOW_DATA_THRESHOLD = 10
MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT = 5


def run_quantile_backtest(df: pd.DataFrame, feature_cols: list, model_lower_clone: XGBRegressor, model_median_clone: XGBRegressor, model_upper_clone: XGBRegressor):
    """Performs a rolling forecast for the quantile models using the specified features."""
    print(f"\n--- Starting Quantile Model Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")
    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Years"):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()
        if test_df.empty or train_df.empty: continue

        # Use the provided full feature list
        X_train, y_train = train_df[feature_cols], train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        model_lower, model_median, model_upper = clone(model_lower_clone), clone(model_median_clone), clone(model_upper_clone)
        model_lower.fit(X_train, y_train)
        model_median.fit(X_train, y_train)
        model_upper.fit(X_train, y_train)

        pred_lower_detrended = model_lower.predict(X_test)
        pred_median_detrended = model_median.predict(X_test)
        pred_upper_detrended = model_upper.predict(X_test)

        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name']].copy()
        fold_results['predicted_yield_lower'] = pred_lower_detrended + test_df['yield_trend']
        fold_results['predicted_yield_median'] = pred_median_detrended + test_df['yield_trend']
        fold_results['predicted_yield_upper'] = pred_upper_detrended + test_df['yield_trend']
        all_predictions.append(fold_results)

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield_median'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\nQuantile backtest complete.")
    return results_df


def analyze_interval_outliers(results_df: pd.DataFrame, report_dir: str):
    """Identifies and analyzes outliers for the 95% prediction interval."""
    print("\n--- Analyzing Prediction Interval Outliers (95% Interval) ---")
    results_df['is_outside_interval'] = (results_df['kreisYield'] < results_df['predicted_yield_lower']) | \
                                        (results_df['kreisYield'] > results_df['predicted_yield_upper'])

    def calculate_outlier_magnitude(row):
        if row['kreisYield'] < row['predicted_yield_lower']: return row['kreisYield'] - row['predicted_yield_lower']
        elif row['kreisYield'] > row['predicted_yield_upper']: return row['kreisYield'] - row['predicted_yield_upper']
        return 0
    results_df['outlier_magnitude'] = results_df.apply(calculate_outlier_magnitude, axis=1)

    outliers_df = results_df[results_df['is_outside_interval']].copy()
    outlier_percentage = (len(outliers_df) / len(results_df)) * 100 if len(results_df) > 0 else 0
    print(f"Total predictions made: {len(results_df)}")
    print(f"Number of times actual yield was outside the 95% interval: {len(outliers_df)}")
    print(f"Outlier Percentage: {outlier_percentage:.2f}% (Target is ~5%)") # <-- MODIFIED Target

    outliers_df.sort_values(by='abs_error', ascending=False, inplace=True)
    save_path = os.path.join(report_dir, 'interval_outliers_analysis.csv')
    outliers_df.to_csv(save_path, index=False)
    print(f"Detailed outlier analysis saved to {save_path}")


def plot_national_average_timeline(backtest_results: pd.DataFrame, report_dir: str):
    """Generates a national average plot with the 95% prediction interval."""
    print("\n--- Generating National Average Prediction Timeline ---")
    yearly_avg = backtest_results.groupby('year').agg(
        avg_actual_yield=('kreisYield', 'mean'),
        avg_pred_median=('predicted_yield_median', 'mean'),
        avg_pred_lower=('predicted_yield_lower', 'mean'),
        avg_pred_upper=('predicted_yield_upper', 'mean')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(yearly_avg['year'], yearly_avg['avg_actual_yield'], label='National Average Actual Yield', color='navy', marker='o', zorder=3)
    ax.plot(yearly_avg['year'], yearly_avg['avg_pred_median'], label='National Average Median Prediction', color='red', linestyle='--', zorder=4)
    ax.fill_between(yearly_avg['year'], yearly_avg['avg_pred_lower'], yearly_avg['avg_pred_upper'], color='red',
                    alpha=0.2, label='95% Prediction Interval (National Avg.)', zorder=2) # <-- MODIFIED Label

    ax.set_title("National Average Yield vs. Predicted Interval (Backtest - V3 Model)", fontsize=16)
    ax.set_xlabel("Year"); ax.set_ylabel("Yield (dt/ha)")
    ax.legend(); ax.grid(True, which='both', linestyle=':')
    plt.tight_layout()

    save_path = os.path.join(report_dir, '05_national_average_timeline.png')
    plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)
    print(f"National average plot saved to {save_path}")

# (Other plotting functions are updated with correct labels)
def plot_best_worst_district_timelines_with_intervals(district_performance: pd.DataFrame,
                                                      backtest_results: pd.DataFrame):
    """
    Generates and saves timeline plots for the 3 best and 3 worst performing reliable districts,
    including the 95% prediction interval.
    """
    print("\n--- Generating timeline plots for best and worst districts with prediction intervals ---")

    # First, create a pool of reliable districts that meet the minimum data requirement
    reliable_perf = district_performance[
        (district_performance['data_point_count'] >= MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT) &
        (district_performance['r2'] != -99)
        ]

    # Now select the best and worst from this reliable pool
    best_districts = reliable_perf.sort_values('r2', ascending=False).head(3)
    worst_districts = reliable_perf.sort_values('r2', ascending=True).head(3)

    # Check if there are enough districts to plot
    if len(best_districts) < 3 or len(worst_districts) < 3:
        print(
            f"  Warning: Could not generate best/worst plot. Not enough reliable districts found with at least {MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT} data points.")
        return

    districts_to_plot = pd.concat([best_districts, worst_districts])

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    axes = axes.flatten()

    for i, (_, district_info) in enumerate(districts_to_plot.iterrows()):
        district_data = backtest_results[backtest_results['district_no'] == district_info['district_no']].sort_values(
            'year')
        ax = axes[i]
        ax.plot(district_data['year'], district_data['kreisYield'], label='Actual Yield', color='navy', marker='o',
                markersize=4, zorder=3)
        ax.plot(district_data['year'], district_data['predicted_yield_median'], label='Median Prediction', color='red',
                linestyle='--', zorder=4)
        ax.fill_between(district_data['year'], district_data['predicted_yield_lower'],
                        district_data['predicted_yield_upper'], color='red', alpha=0.2, label='95% Prediction Interval',
                        zorder=2)
        ax.set_title(f"{'Best' if i < 3 else 'Worst'}: {district_info['name']}\n(R² = {district_info['r2']:.2f})")
        ax.legend()
        ax.grid(True, which='both', linestyle=':')

    # <-- MODIFIED LINE: Added the minimum data points value to the main title -->
    plt.suptitle(
        f"Prediction Timelines for 3 Best and 3 Worst Reliable Districts (Min. {MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT} Years)\n(V3 Quantile Model)",
        fontsize=18,
        y=1.03  # Slightly increased y to give the two-line title space
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(REPORT_DIR, '04_best_worst_district_timelines.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Best/Worst district timelines saved to {save_path}")

# (The core metric/utility functions below require no changes in logic)
def calculate_district_metrics(results_df: pd.DataFrame):
    print("\n--- Calculating performance metrics (based on median prediction) ---")
    def r2_safe(g): return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99
    performance = results_df.groupby('district_no').apply(lambda g: pd.Series({'r2': r2_safe(g), 'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']), 'name': g['name'].iloc[0] if not g.empty else 'Unknown', 'data_point_count': len(g)})).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < LOW_DATA_THRESHOLD
    save_path = os.path.join(REPORT_DIR, 'district_level_performance_metrics.csv')
    performance.to_csv(save_path, index=False)
    print(f"Detailed district metrics saved to {save_path}")
    return performance

def analyze_yearly_performance(results_df: pd.DataFrame):
    print("\n--- Calculating yearly performance metrics (based on median prediction) ---")
    yearly_perf = results_df.groupby('year').apply(lambda g: pd.Series({'r2': r2_score(g['kreisYield'], g['predicted_yield_median']), 'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median'])})).reset_index()
    print("--- Yearly Performance Summary (V3 Quantile Median) ---\n", yearly_perf, "\n----------------------------------------------------")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(yearly_perf['year'], yearly_perf['r2'], color='blue', marker='o', label='Yearly R² (Median)')
    ax.set_title('Median Model Performance Over Time (Backtest - V3 Model)'); plt.legend(); plt.tight_layout()
    # (other plotting settings are fine)
    ax.set_xlabel('Year'); ax.set_ylabel('R-squared (R²)')
    ax.axhline(0, color='grey', linestyle='--'); ax.grid(True, which='both', linestyle=':')
    save_path = os.path.join(REPORT_DIR, '03_performance_over_time.png')
    plt.savefig(save_path, bbox_inches='tight'); plt.close()
    print(f"Yearly performance plot saved to {save_path}")

def plot_performance_map_with_hatching(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    print("\n--- Generating R-squared Map (based on median prediction) ---")
    # ... (No logical changes needed here, just filenames)
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8', legend=True, legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"}, missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'] == True]
    if not low_data_gdf.empty: low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black', label=f'Low Data (< {LOW_DATA_THRESHOLD} years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability')
    ax.set_title('Median Model Performance (R²) by District - V3 Model'); ax.set_axis_off()
    save_path = os.path.join(REPORT_DIR, '01_r_squared_map_with_hatching.png')
    plt.savefig(save_path, bbox_inches='tight'); plt.close()
    print("Performance map saved.")

def plot_r2_vs_data_count(district_performance: pd.DataFrame):
    print("\n--- Generating R² vs. Data Count plot ---")
    # ... (No logical changes needed here, just filenames)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data', palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1); plt.axhline(0, color='grey', linestyle='--')
    plt.title("Data Availability vs Median Model Performance - V3 Model")
    plt.xlabel("Number of Years in Backtest per District"); plt.ylabel("R-squared (R²)")
    plt.legend(title='Is Low Data?')
    save_path = os.path.join(REPORT_DIR, '02_r2_vs_data_count.png')
    plt.savefig(save_path, bbox_inches='tight'); plt.close()
    print("R² vs. Count plot saved.")


def main():
    """Main function to orchestrate the V3 quantile model evaluation pipeline."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting V3 Quantile Model District-Level Evaluation Pipeline ---")

    try:
        model_lower, model_median, model_upper = joblib.load(LOWER_MODEL_PATH), joblib.load(MEDIAN_MODEL_PATH), joblib.load(UPPER_MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
        print("V3 Quantile models, data, and geo-data loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}"); return

    # --- V3 Feature List (Must match the training script) ---
    feature_cols = [
        'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
        'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
        'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
        'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
        'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
        'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast', 'spring_temp_prob_warm_forecast',
        'spring_precip_prob_wet_forecast', 'spring_solar_rad_prob_wet_forecast', 'spring_evaporation_prob_wet_forecast',
        'spring_runoff_prob_wet_forecast', 'spring_soil_temp_l1_prob_warm_forecast', 'spring_snowfall_prob_wet_forecast',
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
        'spring_precip_prob_wet_forecast_sq', 'summer_precip_prob_wet_forecast_sq',
        'is_extreme_heat_forecast', 'is_extreme_drought_forecast', 'drought_x_heat'
    ]

    print("\n--- Applying Causal (Trailing Mean) Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x)
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    backtest_results = run_quantile_backtest(df, feature_cols, model_lower, model_median, model_upper)
    if backtest_results.empty: print("❌ Backtest did not produce results. Terminating."); return

    district_performance = calculate_district_metrics(backtest_results)
    analyze_interval_outliers(backtest_results, REPORT_DIR)

    # --- BUG FIX STARTS HERE ---
    # Save the full predictions dataframe so the deep-dive analysis script can use it
    full_predictions_save_path = os.path.join(REPORT_DIR, 'full_backtest_predictions.csv')
    backtest_results.to_csv(full_predictions_save_path, index=False)
    print(f"\nFull backtest predictions saved to {full_predictions_save_path}")
    # --- BUG FIX ENDS HERE ---

    analyze_yearly_performance(backtest_results)
    plot_performance_map_with_hatching(district_performance, gdf_districts)
    plot_r2_vs_data_count(district_performance)
    plot_best_worst_district_timelines_with_intervals(district_performance, backtest_results)
    plot_national_average_timeline(backtest_results, REPORT_DIR)

    mae_total = backtest_results['abs_error'].mean()
    r2_total = r2_score(backtest_results['kreisYield'], backtest_results['predicted_yield_median'])
    print("\n--- Overall Performance Summary (All Districts, V3 Median Prediction) ---")
    print(f"  Mean Absolute Error (MAE):    {mae_total:.2f} dt/ha")
    print(f"  R-squared (R²):               {r2_total:.4f}")

    reliable_districts = district_performance[~district_performance['is_low_data']]
    reliable_results = backtest_results[backtest_results['district_no'].isin(reliable_districts['district_no'])]
    if not reliable_results.empty:
        mae_reliable = reliable_results['abs_error'].mean()
        r2_reliable = r2_score(reliable_results['kreisYield'], reliable_results['predicted_yield_median'])
        print("\n--- Performance on RELIABLE Districts (>= 10 years, V3 Median Prediction) ---")
        print(f"  Number of reliable districts: {len(reliable_districts)} / {len(district_performance)}")
        print(f"  Mean Absolute Error (MAE):    {mae_reliable:.2f} dt/ha")
        print(f"  R-squared (R²):               {r2_reliable:.4f}")

if __name__ == "__main__":
    main()