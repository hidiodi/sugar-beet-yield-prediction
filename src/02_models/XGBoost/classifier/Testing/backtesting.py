# File: src/models/backtesting.py
# Description: A deep-dive diagnostic script to evaluate model performance
#              at the district level, with a focus on the impact of data availability.

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
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion_final.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = os.path.join('../../../../03_analysis/visualization/reports', 'figures', 'district_level_diagnostics')

BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2021
LOW_DATA_THRESHOLD = 10  # Districts with fewer years of data than this will be flagged


def run_backtest(df: pd.DataFrame, model_template: XGBRegressor):
    """Performs a rolling forecast origin backtest."""
    print(f"\n--- Starting Robust Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")
    all_predictions = []
    feature_cols = model_template.feature_names_in_

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Years"):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()
        if test_df.empty: continue

        X_train, y_train = train_df[feature_cols], train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        model = model_template
        model.fit(X_train, y_train)

        final_predictions = model.predict(X_test) + test_df['yield_trend']

        fold_results = test_df[['district_no', 'year', 'kreisYield', 'name']].copy()
        fold_results['predicted_yield'] = final_predictions
        all_predictions.append(fold_results)

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\n✅ Backtest complete.")
    return results_df


def calculate_district_metrics(results_df: pd.DataFrame):
    """Calculates R², MAE, and data point count for each district."""
    print("Calculating performance metrics and data counts for each district...")

    def r2_safe(g):
        return r2_score(g['kreisYield'], g['predicted_yield']) if len(
            g) > 1 else -99  # Use a sentinel for single points

    def mae_safe(g):
        return mean_absolute_error(g['kreisYield'], g['predicted_yield'])

    performance = results_df.groupby('district_no').apply(
        lambda g: pd.Series({
            'r2': r2_safe(g),
            'mae': mae_safe(g),
            'name': g['name'].iloc[0],
            'data_point_count': len(g)  # Calculate the number of years for this district
        })
    ).reset_index()

    # Flag districts with low data availability
    performance['is_low_data'] = performance['data_point_count'] < LOW_DATA_THRESHOLD

    save_path = os.path.join(REPORT_DIR, 'district_level_performance_metrics.csv')
    performance.to_csv(save_path, index=False)
    print(f"✅ Detailed district metrics saved to {save_path}")
    return performance


def plot_performance_map_with_hatching(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    """Diagnostic 1: Geographic map of R² with hatching for low-data districts."""
    print("Generating Diagnostic 1: R-squared Map with Data Availability Hatching...")

    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # 1. Base Plot: Color all districts by their R² score
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)

    # 2. Overlay Plot: Add hatching ONLY to the low-data districts
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'] == True]
    if not low_data_gdf.empty:
        low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)

    # 3. Create a custom legend for the hatching
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black',
                                 label=f'Low Data (< {LOW_DATA_THRESHOLD} years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability')

    ax.set_title('Model Performance (R²) by District', fontsize=16)
    ax.set_axis_off()

    save_path = os.path.join(REPORT_DIR, '01_r_squared_map_with_hatching.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("✅ Performance map saved.")


def plot_r2_vs_data_count(district_performance: pd.DataFrame):
    """Diagnostic 2: Scatter plot to show the direct relationship between R² and data count."""
    print("Generating Diagnostic 2: R² vs. Data Point Count (The 'Smoking Gun' Plot)...")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)

    plt.axhline(0, color='grey', linestyle='--')
    plt.title("The Relationship Between Data Availability and Model Performance", fontsize=16)
    plt.xlabel("Number of Years in Backtest per District")
    plt.ylabel("R-squared (R²)")
    plt.legend(title='Is Low Data?')

    save_path = os.path.join(REPORT_DIR, '02_r2_vs_data_count.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("✅ R² vs. Count plot saved.")


def main():
    """Main function to orchestrate the district-level evaluation pipeline."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting District-Level Model Evaluation Pipeline ---")

    try:
        model_template = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)

        gdf_districts = gpd.read_file(GEOJSON_PATH)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        district_name_lookup = gdf_districts[['district_no', 'name']]

        df = pd.merge(df, district_name_lookup, on='district_no', how='left')
        print("✅ Model template, data, and district names loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}")
        return

    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    ).fillna(method='ffill').fillna(method='bfill')
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']

    backtest_results = run_backtest(df, model_template)
    district_performance = calculate_district_metrics(backtest_results)

    # --- Generate New, Insightful Visualizations ---
    plot_performance_map_with_hatching(district_performance, gdf_districts)
    plot_r2_vs_data_count(district_performance)

    # --- Final Summary Metrics (Calculated on reliable data) ---
    reliable_districts = district_performance[district_performance['is_low_data'] == False]
    reliable_results = backtest_results[backtest_results['district_no'].isin(reliable_districts['district_no'])]

    mae_reliable = reliable_results['abs_error'].mean()
    r2_reliable = r2_score(reliable_results['kreisYield'], reliable_results['predicted_yield'])

    print("\n--- Performance Summary on RELIABLE Districts (>= 10 years of data) ---")
    print(f"  Number of reliable districts: {len(reliable_districts)} / {len(district_performance)}")
    print(f"  Mean Absolute Error (MAE):    {mae_reliable:.2f} dt/ha")
    print(f"  R-squared (R²):               {r2_reliable:.4f}")
    print("---------------------------------------------------------------------")


if __name__ == "__main__":
    main()