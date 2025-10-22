# File: src/models/backtest_conformalized_model.py
# Description: Backtesting for ADAPTIVE CQR+ (Normalized Intervals).

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

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
LOWER_MODEL_PATH = os.path.join('src/models', 'final_quantile_model_lower.joblib')
MEDIAN_MODEL_PATH = os.path.join('src/models', 'final_quantile_model_median.joblib')
UPPER_MODEL_PATH = os.path.join('src/models', 'final_quantile_model_upper.joblib')
# Load the MULTIPLIER now
MULTIPLIER_PATH = os.path.join('src/models', 'conformal_multiplier.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
# New report directory for clarity
REPORT_DIR = os.path.join('reports', 'figures', 'district_level_diagnostics', 'adaptive_cqr_plus_champion')

BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2024
LOW_DATA_THRESHOLD = 10
MIN_DATAPOINTS_FOR_PLOT = 10


def run_adaptive_backtest(df: pd.DataFrame, feature_cols: list, model_lower_clone: XGBRegressor,
                          model_median_clone: XGBRegressor, model_upper_clone: XGBRegressor, q_mult: float):
    print(f"\n--- Starting ADAPTIVE CQR+ Backtest ({BACKTEST_START_YEAR}-{BACKTEST_END_YEAR}) ---")
    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting"):
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

        # 1. Get Raw Predictions
        raw_lower_detrended = model_lower.predict(X_test)
        raw_upper_detrended = model_upper.predict(X_test)

        # 2. Calculate specific width for EACH prediction
        raw_widths = np.maximum(raw_upper_detrended - raw_lower_detrended, 1.0)

        # 3. Calculate ADAPTIVE adjustment for EACH prediction
        # The adjustment is proportional to the raw width.
        # If the model is unsure (wide raw width), we add MORE safety margin.
        adaptive_adjustment = q_mult * raw_widths

        # 4. Apply adjustment and re-add trend
        fold_results['predicted_yield_lower'] = (raw_lower_detrended - adaptive_adjustment) + test_df['yield_trend']
        fold_results['predicted_yield_upper'] = (raw_upper_detrended + adaptive_adjustment) + test_df['yield_trend']
        fold_results['predicted_yield_median'] = model_median.predict(X_test) + test_df['yield_trend']

        # Save raw width for diagnostics (to see if it's actually adapting)
        fold_results['raw_interval_width'] = raw_widths

        all_predictions.append(fold_results)

    return pd.concat(all_predictions, ignore_index=True)


def analyze_interval_performance(results_df: pd.DataFrame):
    print("\n--- Analyzing ADAPTIVE CQR+ Performance ---")
    results_df['is_covered'] = (results_df['kreisYield'] >= results_df['predicted_yield_lower']) & \
                               (results_df['kreisYield'] <= results_df['predicted_yield_upper'])
    results_df['final_width'] = results_df['predicted_yield_upper'] - results_df['predicted_yield_lower']

    print(f"Prediction Interval Coverage (PICP): {results_df['is_covered'].mean():.2%}")
    print(f"Mean Prediction Interval Width (MPIW): {results_df['final_width'].mean():.2f} dt/ha")

    # Diagnostic: Check if width is actually varying
    width_std = results_df['final_width'].std()
    print(f"Standard Deviation of Interval Widths: {width_std:.2f} (Higher means more adaptive)")
    print(f"Min Width: {results_df['final_width'].min():.2f}, Max Width: {results_df['final_width'].max():.2f}")


# --- PLOTTING FUNCTIONS (Updated colors/titles for CQR+) ---
def plot_national_average_timeline(backtest_results: pd.DataFrame, report_dir: str):
    yearly_avg = backtest_results.groupby('year').agg(
        avg_actual=('kreisYield', 'mean'), avg_med=('predicted_yield_median', 'mean'),
        avg_low=('predicted_yield_lower', 'mean'), avg_high=('predicted_yield_upper', 'mean')
    ).reset_index()
    plt.figure(figsize=(14, 8))
    plt.plot(yearly_avg['year'], yearly_avg['avg_actual'], label='Actual Yield', color='navy', marker='o', zorder=3)
    plt.plot(yearly_avg['year'], yearly_avg['avg_med'], label='Median Prediction', color='red', linestyle='--',
             zorder=4)
    plt.fill_between(yearly_avg['year'], yearly_avg['avg_low'], yearly_avg['avg_high'], color='blue', alpha=0.2,
                     label='Adaptive CQR+ Interval', zorder=2)
    plt.title("National Average: Adaptive CQR+ Model", fontsize=16);
    plt.grid(True, linestyle=':')
    plt.legend();
    plt.savefig(os.path.join(report_dir, '01_adaptive_national_timeline.png'), bbox_inches='tight');
    plt.close()


def plot_best_worst_districts(district_perf: pd.DataFrame, results: pd.DataFrame, report_dir: str):
    reliable = district_perf[district_perf['data_point_count'] >= MIN_DATAPOINTS_FOR_PLOT]
    if len(reliable) < 6: return
    top3 = reliable.nlargest(3, 'r2');
    bot3 = reliable.nsmallest(3, 'r2')
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, (idx, row) in enumerate(pd.concat([top3, bot3]).iterrows()):
        ax = axes.flatten()[i]
        d = results[results['district_no'] == row['district_no']]
        ax.plot(d['year'], d['kreisYield'], color='navy', marker='o', label='Actual')
        ax.plot(d['year'], d['predicted_yield_median'], color='red', linestyle='--', label='Median')
        ax.fill_between(d['year'], d['predicted_yield_lower'], d['predicted_yield_upper'], color='blue', alpha=0.2,
                        label='Adaptive Interval')
        ax.set_title(f"{row['name']} (R²={row['r2']:.2f})");
        ax.grid(True, linestyle=':')
        if i == 0: ax.legend()
    plt.tight_layout();
    plt.savefig(os.path.join(report_dir, '02_adaptive_best_worst.png'));
    plt.close()


# ... (Include standard calculate_district_metrics, plot_performance_map, plot_r2_vs_data_count here if needed, they are unchanged except for titles) ...
def calculate_district_metrics(results_df: pd.DataFrame, report_dir: str):
    def r2_safe(g): return r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else -99

    perf = results_df.groupby('district_no').apply(lambda g: pd.Series(
        {'r2': r2_safe(g), 'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield_median']),
         'name': g['name'].iloc[0], 'data_point_count': len(g)})).reset_index()
    perf['is_low_data'] = perf['data_point_count'] < LOW_DATA_THRESHOLD
    return perf


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting ADAPTIVE CQR+ Evaluation ---")
    try:
        q_mult = joblib.load(MULTIPLIER_PATH)
        df = pd.read_csv(DATA_PATH)
        gdf = gpd.read_file(GEOJSON_PATH)
        # ... (Standard data loading and merging) ...
        gdf.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf['district_no'] = gdf['district_no'].astype(str).str.zfill(5)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf[['district_no', 'name']], on='district_no', how='left')
        print(f"✅ Loaded. Using ADAPTIVE Multiplier: {q_mult:.4f}")
    except Exception as e:
        print(f"❌ Error: {e}"); return

    # ... (Standard feature list and detrending) ...
    FEATURE_COLS = [  # MUST MATCH TRAINING SCRIPT EXACTLY
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
        'summer_soil_temp_l1_prob_warm_forecast',
        'summer_snowfall_prob_wet_forecast', 'lat', 'lon', 'avg_elevation', 'avg_slope', 'avg_bdod_0_30cm',
        'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'avg_bdod_0_100cm',
        'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm', 'avg_phh2o_0_100cm', 'winter_cropland_ndvi_mean',
        'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days', 'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',
        'is_summer_forecast_dry', 'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip',
        'antecedent_gdd_sum_anomaly_sq', 'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
        'spring_temp_prob_warm_forecast_sq', 'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq',
        'summer_precip_prob_wet_forecast_sq'
    ]

    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']

    model_lower = joblib.load(LOWER_MODEL_PATH);
    model_median = joblib.load(MEDIAN_MODEL_PATH);
    model_upper = joblib.load(UPPER_MODEL_PATH)

    results = run_adaptive_backtest(df, FEATURE_COLS, model_lower, model_median, model_upper, q_mult)
    analyze_interval_performance(results)

    dist_perf = calculate_district_metrics(results, REPORT_DIR)
    plot_national_average_timeline(results, REPORT_DIR)
    plot_best_worst_districts(dist_perf, results, REPORT_DIR)
    # plot_performance_map(dist_perf, gdf, REPORT_DIR) # Include if defined

    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()