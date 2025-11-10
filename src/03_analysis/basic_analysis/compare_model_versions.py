# File: src/03_analysis/compare_model_versions.py
# Description: Generates final comparison plots for all models, including the new standalone XGB model.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
import sys

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = config.MODEL_COMPARISON_CONFIG
NOMINAL_COVERAGE_PERCENT = CONFIG['NOMINAL_COVERAGE_PERCENT']
ALPHA = 1 - (NOMINAL_COVERAGE_PERCENT / 100.0)
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])
WOFOST_FORECAST_FILE = project_root / 'data/05_model_input/wofost_walkforward/final_honest_forecasts.csv'


# --- Function Definitions ---
def calculate_interval_score(y_true, lower, upper, alpha):
    """Calculates the Winkler Interval Score."""
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return width + penalty_lower + penalty_upper


def plot_accuracy_comparison(yearly_stats, models, start_year, end_year):
    """Generates the MAE comparison plot with a proper legend."""
    plt.figure(figsize=(18, 8))
    for name, spec in models.items():
        plot_spec = spec.copy()
        plot_spec['label'] = name
        plt.plot(yearly_stats['year'], yearly_stats[f'{name}_mae'], **plot_spec)

    plt.title(f'Point Accuracy Comparison: Mean Absolute Error ({start_year}-{end_year})', fontsize=20)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend(fontsize=14, title="Model", title_fontsize=14)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(yearly_stats['year'], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'comparison_01_accuracy_{start_year}-{end_year}.png', dpi=300)
    plt.close()


def plot_sharpness_comparison(yearly_stats, models_quantile, df_quantile_summary, start_year, end_year):
    """Generates the Interval Width comparison plot."""
    plt.figure(figsize=(18, 8))
    for name, spec in models_quantile.items():
        plot_spec = spec.copy()
        coverage = df_quantile_summary.loc[df_quantile_summary['Model'] == name, 'Coverage (%)'].iloc[0]
        plot_spec['label'] = f'{name} — Coverage: {coverage:.1f}%'
        plt.plot(yearly_stats['year'], yearly_stats[f'{name}_width'], **plot_spec)

    plt.title(f'Interval Sharpness Comparison: Average Width ({start_year}-{end_year})', fontsize=20)
    plt.ylabel('Average Interval Width', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend(fontsize=14, title="Model & Achieved Coverage", title_fontsize=14)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(yearly_stats['year'], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'comparison_02_sharpness_{start_year}-{end_year}.png', dpi=300)
    plt.close()


def plot_score_comparison(yearly_stats, models_quantile, start_year, end_year):
    """Generates the Interval Score comparison plot."""
    plt.figure(figsize=(18, 8))
    for name, spec in models_quantile.items():
        plot_spec = spec.copy()
        plot_spec['label'] = name
        plt.plot(yearly_stats['year'], yearly_stats[f'{name}_score'], **plot_spec)

    plt.title(f'Overall Interval Quality: {int(NOMINAL_COVERAGE_PERCENT)}% Interval Score ({start_year}-{end_year})',
              fontsize=20)
    plt.ylabel('Mean Interval Score (Lower is Better)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend(fontsize=14, title="Model", title_fontsize=14)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(yearly_stats['year'], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'comparison_03_score_{start_year}-{end_year}.png', dpi=300)
    plt.close()


def plot_national_average_timelines(national_avg_df, models_quantile, start_year, end_year, output_dir,
                                    nominal_coverage):
    """
    Generates a 2x2 grid plot comparing national average actual yield vs.
    predicted intervals for the four main quantile models, matching the example plot style.
    """
    logging.info("Generating 2x2 national average timeline comparison plot...")

    # Ensure we have exactly 4 models for a 2x2 grid
    if len(models_quantile) != 4:
        logging.warning(f"Expected 4 models for 2x2 grid, but got {len(models_quantile)}. Plotting first 4.")
        models_to_plot = models_quantile[:4]
    else:
        models_to_plot = models_quantile

    fig, axes = plt.subplots(2, 2, figsize=(22, 18), sharex=True, sharey=True)
    axes = axes.flatten()  # To iterate

    # Data for plotting
    years = national_avg_df['year']
    actual_yield = national_avg_df['kreisYield']

    for i, model_name in enumerate(models_to_plot):
        ax = axes[i]

        # Get data for this model
        median_pred = national_avg_df[f'{model_name}_pred']
        lower_pred = national_avg_df[f'{model_name}_lower']
        upper_pred = national_avg_df[f'{model_name}_upper']

        # 1. Plot 95% Prediction Interval (shaded area) - matching example image
        ax.fill_between(years, lower_pred, upper_pred, color='pink', alpha=0.6,
                        label=f'{int(nominal_coverage)}% Prediction Interval')

        # 2. Plot National Average Actual Yield (blue line with dots) - matching example image
        ax.plot(years, actual_yield, 'o-', color='darkblue', linewidth=2,
                label='National Average Actual Yield', markersize=8)

        # 3. Plot Median Prediction (red dashed line) - matching example image
        ax.plot(years, median_pred, '--', color='red', linewidth=2.0,
                label='Median Prediction')

        # --- Subplot Formatting ---
        ax.set_title(model_name, fontsize=18, fontweight='bold')
        ax.grid(True, linestyle=':', which='both', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Add legend to each subplot
        handles, labels = ax.get_legend_handles_labels()
        # Reorder legend to match example: Actual, Median, Interval
        order = [1, 2, 0]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14, loc='upper left')

        # Set x-ticks to be integer years
        ax.set_xticks(np.arange(start_year, end_year + 1, 5))  # Tick every 5 years
        ax.tick_params(axis='x', rotation=0)  # No rotation as it's shared

    # --- Figure-wide Formatting ---
    fig.suptitle(f'National Average Yield vs. Predicted Interval ({start_year}-{end_year})', fontsize=26, y=1.03)

    # Common axis labels
    fig.text(0.5, 0.04, 'Year', ha='center', va='center', fontsize=20)
    fig.text(0.04, 0.5, 'Yield (dt/ha)', ha='center', va='center', rotation='vertical', fontsize=20)

    plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.96])  # Adjust rect for suptitle and common labels

    save_path = output_dir / f'comparison_04_national_average_timeline_{start_year}-{end_year}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main function to execute the model comparison workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("--- Loading Final Model Predictions for Comparison ---")

    try:
        # Load predictions from all models
        df_hybrid_xgb = pd.read_csv(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
        df_standalone_xgb = pd.read_csv(CONFIG['STANDALONE_XGB_PREDICTIONS_FILE'])  # New model
        df_cqr = pd.read_csv(CONFIG['ADAPTIVE_CQR_PREDICTIONS_FILE'])
        df_ngb = pd.read_csv(CONFIG['NGBOOST_PREDICTIONS_FILE'])
        df_wofost = pd.read_csv(WOFOST_FORECAST_FILE)
    except Exception as e:
        logging.error(f"❌ FATAL: Input file not found. Ensure all backtests have been run. Details: {e}")
        sys.exit(1)

    # --- Data Merging and Preparation ---
    # Use the hybrid model's output as the base for ground truth and alignment
    df_merged = df_hybrid_xgb[['year', 'district_no', 'kreisYield']].copy()

    # Merge predictions from all models onto the base DataFrame
    df_merged['Hybrid (TS+XGB)_pred'] = df_hybrid_xgb['predicted_yield_median']
    df_merged['Hybrid (TS+XGB)_lower'] = df_hybrid_xgb['predicted_yield_lower']
    df_merged['Hybrid (TS+XGB)_upper'] = df_hybrid_xgb['predicted_yield_upper']

    df_merged = pd.merge(df_merged, df_standalone_xgb[
        ['year', 'district_no', 'predicted_yield_median', 'predicted_yield_lower', 'predicted_yield_upper']],
                         on=['year', 'district_no'], how='left', suffixes=('', '_standalone'))
    df_merged.rename(
        columns={'predicted_yield_median': 'Standalone XGB_pred', 'predicted_yield_lower': 'Standalone XGB_lower',
                 'predicted_yield_upper': 'Standalone XGB_upper'}, inplace=True)

    df_merged['Adaptive CQR_pred'] = df_cqr['predicted_yield_median']
    df_merged['Adaptive CQR_lower'] = df_cqr['predicted_yield_lower']
    df_merged['Adaptive CQR_upper'] = df_cqr['predicted_yield_upper']

    df_merged['NGBoost_pred'] = df_ngb['predicted_yield_median']
    df_merged['NGBoost_lower'] = df_ngb['predicted_yield_lower']
    df_merged['NGBoost_upper'] = df_ngb['predicted_yield_upper']

    # Merge WOFOST separately as it may not have district_no
    df_merged['Pure WOFOST_pred'] = df_wofost['final_corrected_forecast']

    # --- Model Lists ---
    models_point = ["Pure WOFOST", "Hybrid (TS+XGB)", "Standalone XGB", "Adaptive CQR", "NGBoost"]
    models_quantile = ["Hybrid (TS+XGB)", "Standalone XGB", "Adaptive CQR", "NGBoost"]  # 4 models for 2x2 grid

    # --- Performance Summaries ---
    point_results = [{'Model': name, 'MAE': (df_merged[f'{name}_pred'] - df_merged['kreisYield']).abs().mean(),
                      'R-squared': r2_score(df_merged['kreisYield'], df_merged[f'{name}_pred'])} for name in
                     models_point]
    df_summary = pd.DataFrame(point_results).sort_values('MAE')
    print("\n" + "=" * 80);
    print("      POINT FORECAST ACCURACY");
    print("=" * 80)
    print(df_summary.to_string(index=False, float_format="%.4f"));
    print("=" * 80)

    quantile_results = []
    for name in models_quantile:
        score = calculate_interval_score(df_merged['kreisYield'], df_merged[f'{name}_lower'],
                                         df_merged[f'{name}_upper'], ALPHA).mean()
        coverage = ((df_merged['kreisYield'] >= df_merged[f'{name}_lower']) & (
                df_merged['kreisYield'] <= df_merged[f'{name}_upper'])).mean() * 100
        width = (df_merged[f'{name}_upper'] - df_merged[f'{name}_lower']).mean()
        quantile_results.append({'Model': name, 'Interval Score': score, 'Coverage (%)': coverage, 'Width': width})
    df_quantile_summary = pd.DataFrame(quantile_results).sort_values('Interval Score')
    print("\n" + "=" * 80);
    print(f"      PREDICTION INTERVAL QUALITY ({int(NOMINAL_COVERAGE_PERCENT)}%)");
    print("=" * 80)
    print(df_quantile_summary.to_string(index=False, float_format="%.4f"));
    print("=" * 80)

    # --- NEW: Aggregate national average yields for timeline plot ---
    logging.info("Aggregating national average yields for timeline plot...")
    avg_cols = ['year', 'kreisYield']
    for name in models_quantile:
        avg_cols.extend([f'{name}_pred', f'{name}_lower', f'{name}_upper'])

    # Filter to only columns that actually exist in df_merged
    valid_avg_cols = [col for col in avg_cols if col in df_merged.columns]
    national_avg_df = df_merged[valid_avg_cols].groupby('year').mean().reset_index()

    # --- Yearly Statistics for Metric Plotting ---
    logging.info("Aggregating metrics for yearly plots...")
    for name in models_point: df_merged[f'{name}_mae'] = (df_merged[f'{name}_pred'] - df_merged['kreisYield']).abs()
    for name in models_quantile:
        df_merged[f'{name}_width'] = df_merged[f'{name}_upper'] - df_merged[f'{name}_lower']
        df_merged[f'{name}_score'] = calculate_interval_score(df_merged['kreisYield'], df_merged[f'{name}_lower'],
                                                              df_merged[f'{name}_upper'], ALPHA)

    agg_dict = {f'{name}_mae': 'mean' for name in models_point}
    agg_dict.update({f'{name}_width': 'mean' for name in models_quantile})
    agg_dict.update({f'{name}_score': 'mean' for name in models_quantile})
    yearly_stats = df_merged.groupby('year').agg(agg_dict).reset_index()

    start_year, end_year = yearly_stats['year'].min(), yearly_stats['year'].max()

    # --- Plotting ---
    models_to_plot_styles = {
        "Pure WOFOST": {'marker': 'x', 'color': 'royalblue', 'linewidth': 2.0},
        "NGBoost": {'marker': 'D', 'color': 'firebrick', 'linewidth': 2.0},
        "Hybrid (TS+XGB)": {'marker': 's', 'color': 'darkorange', 'linewidth': 2.5},
        "Standalone XGB": {'marker': 'p', 'color': 'mediumorchid', 'linewidth': 2.0},  # New style
        "Adaptive CQR": {'marker': '*', 'markersize': 10, 'color': 'darkgreen', 'linewidth': 2.0},
    }

    models_point_plot = {k: v for k, v in models_to_plot_styles.items() if k in models_point}
    models_quantile_plot = {k: v for k, v in models_to_plot_styles.items() if k in models_quantile}

    plot_accuracy_comparison(yearly_stats, models_point_plot, start_year, end_year)
    plot_sharpness_comparison(yearly_stats, models_quantile_plot, df_quantile_summary, start_year, end_year)
    plot_score_comparison(yearly_stats, models_quantile_plot, start_year, end_year)

    # --- NEW: Call the 4-plot function ---
    plot_national_average_timelines(national_avg_df, models_quantile, start_year, end_year, OUTPUT_DIR,
                                    NOMINAL_COVERAGE_PERCENT)

    logging.info(f"✓ Final comparison plots saved successfully to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()