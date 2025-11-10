# File: src/03_analysis/analyze_performance_degradation.py
# Description: This script analyzes the backtest results of a model to determine
#              if its performance is systematically degrading over time, likely due
#              to climate-driven distributional drift.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import linregress
import numpy as np

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The backtest results of our best model
BACKTEST_FILE = 'reports/figures/district_level_diagnostics/final_ensemble_champion/full_backtest_predictions.csv'
OUTPUT_DIR = Path('reports/figures/performance_degradation_analysis')

NOMINAL_COVERAGE_PERCENT = 95.0
ALPHA = 1 - (NOMINAL_COVERAGE_PERCENT / 100.0)


def calculate_interval_score(y_true, lower, upper, alpha):
    """Calculates the Winkler Interval Score."""
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return width + penalty_lower + penalty_upper


def main():
    """Main function to analyze and visualize performance degradation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"--- Starting Performance Degradation Analysis for {Path(BACKTEST_FILE).parent.name} ---")

    try:
        df = pd.read_csv(BACKTEST_FILE)
    except FileNotFoundError as e:
        logging.error(f"❌ FATAL: Backtest file not found. Ensure the champion model has been evaluated. Details: {e}")
        return

    # --- 1. Calculate Yearly Performance Metrics ---
    logging.info("Calculating performance metrics for each year...")
    df['abs_error'] = (df['predicted_yield_median'] - df['kreisYield']).abs()
    df['score'] = calculate_interval_score(df['kreisYield'], df['predicted_yield_lower'], df['predicted_yield_upper'],
                                           ALPHA)

    yearly_mae = df.groupby('year')['abs_error'].mean()
    yearly_score = df.groupby('year')['score'].mean()

    yearly_r2 = df.groupby('year').apply(
        lambda g: r2_score(g['kreisYield'], g['predicted_yield_median']) if len(g) > 1 else np.nan
    ).dropna()

    metrics = pd.DataFrame({'MAE': yearly_mae, 'Interval Score': yearly_score, 'R-squared': yearly_r2}).reset_index()

    # --- 2. Fit Trend Lines and Analyze Significance ---
    logging.info("Fitting linear trends to performance metrics...")
    results = {}
    for metric in ['MAE', 'R-squared', 'Interval Score']:
        slope, intercept, r_value, p_value, std_err = linregress(metrics['year'], metrics[metric])
        results[metric] = {'slope': slope, 'p_value': p_value}
        metrics[f'{metric}_trend'] = intercept + slope * metrics['year']

    print("\n" + "=" * 80)
    print("      ANALYSIS OF PERFORMANCE DEGRADATION OVER TIME")
    print("=" * 80)
    for metric, res in results.items():
        print(f"Metric: {metric}")
        print(f"  - Annual Trend (Slope): {res['slope']:.4f} per year")
        print(f"  - P-value: {res['p_value']:.4f}")
        if res['p_value'] < 0.05:
            print("  - Verdict: Statistically SIGNIFICANT degradation detected.")
        else:
            print("  - Verdict: No statistically significant degradation detected.")
    print("=" * 80 + "\n")

    # --- 3. Generate Plots ---
    logging.info("Generating degradation plots...")

    # MAE Plot
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=metrics, x='year', y='MAE')
    sns.lineplot(data=metrics, x='year', y='MAE_trend', color='red', label=f"Trend (p={results['MAE']['p_value']:.3f})")
    plt.title('Model Accuracy (MAE) Degradation Over Time', fontsize=16)
    plt.ylabel('Mean Absolute Error (Lower is Better)')
    plt.savefig(OUTPUT_DIR / '01_mae_degradation.png', dpi=300)
    plt.close()

    # R-squared Plot
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=metrics, x='year', y='R-squared')
    sns.lineplot(data=metrics, x='year', y='R-squared_trend', color='red',
                 label=f"Trend (p={results['R-squared']['p_value']:.3f})")
    plt.title('Model Accuracy (R-squared) Degradation Over Time', fontsize=16)
    plt.ylabel('R-squared (Higher is Better)')
    plt.savefig(OUTPUT_DIR / '02_r2_degradation.png', dpi=300)
    plt.close()

    # Interval Score Plot
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=metrics, x='year', y='Interval Score')
    sns.lineplot(data=metrics, x='year', y='Interval Score_trend', color='red',
                 label=f"Trend (p={results['Interval Score']['p_value']:.3f})")
    plt.title('Overall Model Quality (Interval Score) Degradation Over Time', fontsize=16)
    plt.ylabel('Interval Score (Lower is Better)')
    plt.savefig(OUTPUT_DIR / '03_score_degradation.png', dpi=300)
    plt.close()

    logging.info(f"✓ Degradation analysis complete. Plots saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()