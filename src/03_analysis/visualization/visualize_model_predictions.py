# File: src/models/visualize_model_predictions.py
# Description: Loads the final trained model and visualizes its performance
#              on BOTH the Validation (2009-2018) and Test (2019+) splits.

import pandas as pd
import geopandas as gpd
import joblib
from xgboost import XGBRegressor
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# --- Define Paths and Splits ---
MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion_final.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = os.path.join('reports', 'figures', 'final_model_visuals')

VALIDATION_START_YEAR = 2009
TEST_START_YEAR = 2019


def create_results_df(df_split: pd.DataFrame, model: XGBRegressor, trend_df: pd.DataFrame):
    """Makes predictions and creates a standardized results DataFrame."""
    X_split = df_split[model.feature_names_in_]
    y_split_actual = df_split['kreisYield']

    detrended_predictions = model.predict(X_split)
    final_predictions = detrended_predictions + trend_df['yield_trend']

    results_df = pd.DataFrame({
        'district_no': df_split['district_no'],
        'year': df_split['year'],
        'actual_yield': y_split_actual,
        'predicted_yield': final_predictions
    })
    results_df['error'] = results_df['predicted_yield'] - results_df['actual_yield']
    results_df['data_split'] = 'Validation' if df_split['year'].min() < TEST_START_YEAR else 'Test'
    return results_df


def plot_performance_metrics(results_df: pd.DataFrame, split_name: str):
    """Generates Scatter and Histogram plots for a given split."""

    # --- Plot 1: Predicted vs. Actual Scatter Plot ---
    r2 = r2_score(results_df['actual_yield'], results_df['predicted_yield'])
    mae = mean_absolute_error(results_df['actual_yield'], results_df['predicted_yield'])
    plt.figure(figsize=(8, 8))
    plt.scatter(results_df['actual_yield'], results_df['predicted_yield'], alpha=0.3)
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Forecast")
    plt.title(f"{split_name} Set Performance (R² = {r2:.3f} | MAE = {mae:.2f} dt/ha)")
    plt.xlabel("Actual Yield (dt/ha)")
    plt.ylabel("Predicted Yield (dt/ha)")
    plt.legend();
    plt.grid(True);
    plt.axis('equal');
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, f'predicted_vs_actual_scatter_{split_name.lower()}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Scatter plot saved to {save_path}")

    # --- Plot 2: Error Distribution Histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['error'], bins=50, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    mean_error = results_df['error'].mean()
    plt.axvline(x=mean_error, color='k', linestyle=':', label=f'Mean Error: {mean_error:.2f}')
    plt.title(f"{split_name} Error Distribution (Predicted - Actual)")
    plt.xlabel("Prediction Error (dt/ha)")
    plt.ylabel("Frequency")
    plt.legend();
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, f'error_distribution_histogram_{split_name.lower()}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Error histogram saved to {save_path}")


def plot_geographic_error_map(results_df: pd.DataFrame, gdf_districts: gpd.GeoDataFrame, split_name: str):
    """Generates Geographic Error Map for a given split."""

    avg_error_by_district = results_df.groupby('district_no')['error'].mean().reset_index()
    avg_error_by_district['district_no'] = avg_error_by_district['district_no'].astype(str).str.zfill(5)

    # Merge on the GeoDataFrame. We've ensured gdf_districts is clean in main().
    merged_gdf = gdf_districts.merge(avg_error_by_district, on='district_no', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged_gdf.plot(column='error', cmap='RdBu_r', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, legend_kwds={'label': f"Average Prediction Error ({split_name}) (dt/ha)",
                                              'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey', 'label': 'No Data'})
    ax.set_title(f'Average Prediction Error by District ({split_name})')
    ax.set_axis_off()
    save_path = os.path.join(REPORT_DIR, f'geographic_error_map_{split_name.lower()}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Geographic map saved to {save_path}")


def plot_national_average_timeseries(all_results: pd.DataFrame):
    """Generates a combined time series plot for the entire validation/test period."""

    national_avg = all_results.groupby('year')[['actual_yield', 'predicted_yield']].mean().reset_index()

    plt.figure(figsize=(12, 6))

    # Plot Actual Yield (solid line for the whole period)
    plt.plot(national_avg['year'], national_avg['actual_yield'], marker='o',
             linestyle='-', color='black', label='Actual National Average')

    # Plot Predicted Yield (distinguished by split)
    for split_name, group in all_results.groupby('data_split'):
        avg_split = group.groupby('year')[['predicted_yield']].mean().reset_index()
        plt.plot(avg_split['year'], avg_split['predicted_yield'], marker='x',
                 linestyle='--', label=f'Predicted ({split_name} Set)')

    # Add a vertical line to mark the split between Validation and Test
    plt.axvline(x=TEST_START_YEAR - 0.5, color='grey', linestyle=':',
                label=f'Validation/Test Split ({TEST_START_YEAR})')

    plt.title(f'National Average Yield: Actual vs. Predicted (Years {VALIDATION_START_YEAR}+)')
    plt.xlabel("Year")
    plt.ylabel("Yield (dt/ha)")
    plt.xticks(national_avg['year'].astype(int))
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    save_path = os.path.join(REPORT_DIR, 'national_average_timeseries_combined.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Combined Time Series saved to {save_path}")


def main():
    """Main function to orchestrate the visualization."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting Final Model Performance Visualization ---")

    # --- 1. Load Model, Data, and Geojson ---
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)
        gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        print("Model, data, and geojson loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: A required file was not found. Details: {e}")
        return

    # --- 2. Recreate Splits and Trend ---
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    ).fillna(method='ffill').fillna(method='bfill')

    val_df = df[(df['year'] >= VALIDATION_START_YEAR) & (df['year'] < TEST_START_YEAR)].copy()
    test_df = df[df['year'] >= TEST_START_YEAR].copy()

    # --- 3. Generate Results DataFrames ---
    val_results = create_results_df(val_df, model, val_df)
    test_results = create_results_df(test_df, model, test_df)

    # Combine for the national time series plot
    all_results = pd.concat([val_results, test_results], ignore_index=True)

    # --- 4. Generate Visualizations for Validation Split ---
    print(f"\n--- Generating Visuals for VALIDATION Set ({VALIDATION_START_YEAR}-{TEST_START_YEAR - 1}) ---")
    plot_performance_metrics(val_results, "Validation")
    plot_geographic_error_map(val_results, gdf_districts, "Validation")

    # --- 5. Generate Visualizations for Test Split ---
    print(f"\n--- Generating Visuals for TEST Set ({TEST_START_YEAR}+) ---")
    plot_performance_metrics(test_results, "Test")
    plot_geographic_error_map(test_results, gdf_districts, "Test")

    # --- 6. Generate Combined Time Series Plot ---
    print("\n--- Generating Combined National Time Series Plot ---")
    plot_national_average_timeseries(all_results)

    print("\n--- All visualizations complete. Check the reports/figures/final_model_visuals directory. ---")


if __name__ == "__main__":
    main()