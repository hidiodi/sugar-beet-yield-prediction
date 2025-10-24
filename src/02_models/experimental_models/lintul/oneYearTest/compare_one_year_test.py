# File: src/models/experimental_models/lintul/analyze_one_year_test_results.py
# Description: The definitive analysis script for the one-year test.
#              It merges the results from the historical simulation and the forecast
#              simulation to directly compare both against the actual observed yield.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# --- Configuration ---
TEST_YEAR = 2018
HISTORICAL_RESULTS_PATH = f'data/03_primary/historical_residuals_{TEST_YEAR}_TEST.csv'
FORECAST_RESULTS_PATH = f'data/03_primary/forecast_simulation_{TEST_YEAR}_TEST.csv'


def analyze_final_comparison():
    """
    Loads, merges, and analyzes the results to provide the ultimate comparison.
    """
    logging.info(f"--- Running Definitive Analysis for ONE-YEAR TEST ({TEST_YEAR}) ---")

    # --- 1. Load Data ---
    try:
        df_hist = pd.read_csv(HISTORICAL_RESULTS_PATH)
        df_fcst = pd.read_csv(FORECAST_RESULTS_PATH)
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required input file was not found. {e}")
        logging.error(
            "Please run both 'generate_historical_residuals_ONE_YEAR_TEST.py' and 'run_forecast_simulation_ONE_YEAR_TEST.py' first.")
        return

    # --- 2. Prepare and Merge DataFrames ---
    # Rename columns for clarity in the final merged table
    df_hist.rename(columns={'lintul_yield': 'lintul_yield_perfect_weather'}, inplace=True)
    df_fcst.rename(columns={'lintul_forecast_baseline': 'lintul_yield_forecast_weather'}, inplace=True)

    # Merge the two results into a single, comprehensive DataFrame
    df_final = pd.merge(
        df_hist[['year', 'district_no', 'actual_yield', 'lintul_yield_perfect_weather']],
        df_fcst[['year', 'district_no', 'lintul_yield_forecast_weather']],
        on=['year', 'district_no']
    )

    if df_final.empty:
        print("Merged DataFrame is empty. Cannot proceed with analysis.")
        return

    print("--- Final Merged Data Sample ---")
    print(df_final.head())

    # --- 3. Quantitative Analysis (The Numbers) ---
    print("\n--- Quantitative Performance Metrics ---")
    mae_perfect = mean_absolute_error(df_final['actual_yield'], df_final['lintul_yield_perfect_weather'])
    mae_forecast = mean_absolute_error(df_final['actual_yield'], df_final['lintul_yield_forecast_weather'])

    r2_perfect = r2_score(df_final['actual_yield'], df_final['lintul_yield_perfect_weather'])
    r2_forecast = r2_score(df_final['actual_yield'], df_final['lintul_yield_forecast_weather'])

    print(f"  MAE (Perfect Weather vs Actual): {mae_perfect:.2f} dt/ha")
    print(f"  MAE (Forecast Weather vs Actual): {mae_forecast:.2f} dt/ha")
    print("-" * 20)
    print(f"  R² (Perfect Weather vs Actual): {r2_perfect:.2f}")
    print(f"  R² (Forecast Weather vs Actual): {r2_forecast:.2f}")

    print("\nCorrelation Matrix:")
    print(df_final[['actual_yield', 'lintul_yield_perfect_weather', 'lintul_yield_forecast_weather']].corr())

    # --- 4. Visual Analysis (The Plots) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle(f'Crop Model Performance Comparison for {TEST_YEAR}', fontsize=16)

    # Plot 1: Perfect Weather Simulation vs. Actual Yield
    sns.scatterplot(data=df_final, x='actual_yield', y='lintul_yield_perfect_weather', ax=axes[0], alpha=0.7)
    axes[0].plot([df_final['actual_yield'].min(), df_final['actual_yield'].max()],
                 [df_final['actual_yield'].min(), df_final['actual_yield'].max()],
                 'r--', lw=2, label='1:1 Line')
    axes[0].set_title('Performance with PERFECT Historical Weather')
    axes[0].set_xlabel('Actual Yield (dt/ha)')
    axes[0].set_ylabel('Simulated Yield (dt/ha)')
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: Forecast Weather Simulation vs. Actual Yield
    sns.scatterplot(data=df_final, x='actual_yield', y='lintul_yield_forecast_weather', ax=axes[1], alpha=0.7)
    axes[1].plot([df_final['actual_yield'].min(), df_final['actual_yield'].max()],
                 [df_final['actual_yield'].min(), df_final['actual_yield'].max()],
                 'r--', lw=2, label='1:1 Line')
    axes[1].set_title('Performance with FORECAST Weather')
    axes[1].set_xlabel('Actual Yield (dt/ha)')
    axes[1].set_ylabel('')  # Remove redundant y-axis label
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    analyze_final_comparison()