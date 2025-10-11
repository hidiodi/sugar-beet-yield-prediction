# File: src/planning/demonstrate_value.py
# Description: Demonstrates the business value of the Stage 1 model by comparing its forecast
# against a naive "last year's yield" forecast for a historical test year.

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration ---
MODEL_PATH = os.path.join('src/models', 'stage1_preseason_xgb_model.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
OUTPUT_DIR = os.path.join('reports', 'stage1_planning_demonstration')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The year we will use for our historical "what if" simulation
TEST_YEAR = 2021


def run_simulation():
    """Performs a head-to-head comparison of the model vs. a naive forecast."""
    print(f"--- Running Business Value Simulation for the year {TEST_YEAR} ---")

    # --- Step 1: Load Data and Trained Model ---
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: Missing required file. {e}. Please ensure model is trained and features are generated.")
        return

    # --- Step 2: Prepare the Datasets for the Simulation ---
    # a) Ground Truth: The actual results from our test year
    ground_truth_df = df[df['year'] == TEST_YEAR][['district_no', 'kreisYield']].copy()
    ground_truth_df.rename(columns={'kreisYield': 'actual_yield'}, inplace=True)

    # b) Naive Forecast: Last year's (2020) actual yield
    naive_forecast_df = df[df['year'] == TEST_YEAR - 1][['district_no', 'kreisYield']].copy()
    naive_forecast_df.rename(columns={'kreisYield': 'naive_forecast'}, inplace=True)

    # c) Model Forecast: Run our model on the features available in March of the test year
    model_input_df = df[df['year'] == TEST_YEAR].copy()
    feature_cols = model.get_booster().feature_names
    model_input_df['model_forecast'] = model.predict(model_input_df[feature_cols])

    # --- Step 3: Combine into a Single Comparison DataFrame ---
    comparison_df = pd.merge(ground_truth_df, naive_forecast_df, on='district_no', how='inner')
    comparison_df = pd.merge(comparison_df, model_input_df[['district_no', 'model_forecast']], on='district_no',
                             how='inner')

    # --- Step 4: Quantify the Advantage (Calculate Errors) ---
    rmse_naive = np.sqrt(mean_squared_error(comparison_df['actual_yield'], comparison_df['naive_forecast']))
    rmse_model = np.sqrt(mean_squared_error(comparison_df['actual_yield'], comparison_df['model_forecast']))
    error_reduction = ((rmse_naive - rmse_model) / rmse_naive) * 100

    print("\n--- Quantitative Comparison ---")
    print(f"Naive Forecast (Last Year's Yield) RMSE: {rmse_naive:.2f} dt/ha")
    print(f"Stage 1 Model Forecast RMSE:             {rmse_model:.2f} dt/ha")
    print("--------------------------------------------------")
    print(f"Clear Advantage: The model reduced the forecast error by {error_reduction:.1f}%")

    # --- Step 5: Visualize the Advantage ---
    # a) Scatter Plot: Actual vs. Forecast
    plt.figure(figsize=(10, 10))
    plt.scatter(comparison_df['actual_yield'], comparison_df['naive_forecast'], alpha=0.5,
                label=f'Naive Forecast (RMSE: {rmse_naive:.2f})', color='red')
    plt.scatter(comparison_df['actual_yield'], comparison_df['model_forecast'], alpha=0.5,
                label=f'Model Forecast (RMSE: {rmse_model:.2f})', color='green')
    plt.plot([comparison_df['actual_yield'].min(), comparison_df['actual_yield'].max()],
             [comparison_df['actual_yield'].min(), comparison_df['actual_yield'].max()], 'k--',
             label='Perfect Forecast')
    plt.title(f'Forecast Accuracy Showdown ({TEST_YEAR})', fontsize=16)
    plt.xlabel('Actual Yield (dt/ha)', fontsize=12)
    plt.ylabel('Forecasted Yield (dt/ha)', fontsize=12)
    plt.legend()
    plt.grid(True)
    scatter_path = os.path.join(OUTPUT_DIR, 'forecast_accuracy_comparison.png')
    plt.savefig(scatter_path, dpi=300)
    print(f"\n✅ Accuracy comparison scatter plot saved to {scatter_path}")
    plt.close()

    # b) Histogram of Errors
    comparison_df['error_naive'] = comparison_df['actual_yield'] - comparison_df['naive_forecast']
    comparison_df['error_model'] = comparison_df['actual_yield'] - comparison_df['model_forecast']

    plt.figure(figsize=(12, 6))
    plt.hist(comparison_df['error_naive'], bins=30, alpha=0.6, label='Naive Forecast Error', color='red')
    plt.hist(comparison_df['error_model'], bins=30, alpha=0.6, label='Model Forecast Error', color='green')
    plt.axvline(0, color='k', linestyle='--')
    plt.title('Distribution of Forecast Errors', fontsize=16)
    plt.xlabel('Error (Actual - Forecast) in dt/ha', fontsize=12)
    plt.ylabel('Number of Districts', fontsize=12)
    plt.legend()
    error_hist_path = os.path.join(OUTPUT_DIR, 'forecast_error_distribution.png')
    plt.savefig(error_hist_path, dpi=300)
    print(f"✅ Error distribution histogram saved to {error_hist_path}")
    plt.close()

    print("\n--- Business Value Demonstration Complete ---")


if __name__ == "__main__":
    run_simulation()