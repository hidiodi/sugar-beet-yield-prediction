# File: src/03_analysis/compare_model_versions.py
# Description: Generates final comparison plots for all models, including the champion ensemble.
# Refactored to use central configuration from src.config

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

OUTPUT_DIR = CONFIG['OUTPUT_DIR']


# --- Function Definitions ---

def validate_dataframe_columns(df, required_cols, filename):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ FATAL: Missing columns in {filename}: {missing_cols}")
        sys.exit(1)
    return True


def calculate_interval_score(y_true, lower, upper, alpha):
    """Calculates the Winkler Interval Score."""
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return width + penalty_lower + penalty_upper


def plot_accuracy_comparison(yearly_stats, models, start_year, end_year):
    """Generates the MAE comparison plot."""
    plt.figure(figsize=(18, 8))
    for name, spec in models.items():
        plt.plot(yearly_stats['year'], yearly_stats[f'{name}_mae'], **spec)

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
        coverage = df_quantile_summary.loc[df_quantile_summary['Model'] == name, 'Coverage (%)'].iloc[0]
        spec['label'] = f'{name} — Coverage: {coverage:.1f}%'
        if name == 'Hybrid Ensemble':
            spec['label'] += ' (CHAMPION)'
        plt.plot(yearly_stats['year'], yearly_stats[f'{name}_width'], **spec)

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
        spec['label'] = name
        if name == 'Hybrid Ensemble':
            spec['label'] += ' (CHAMPION)'
        plt.plot(yearly_stats['year'], yearly_stats[f'{name}_score'], **spec)

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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("--- Loading Final Model Predictions for Comparison ---")

    try:
        df_ensemble = pd.read_csv(CONFIG['FINAL_ENSEMBLE_PREDICTIONS_FILE'])
        df_xgb = pd.read_csv(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
        df_cqr = pd.read_csv(CONFIG['ADAPTIVE_CQR_PREDICTIONS_FILE'])
        df_ngb = pd.read_csv(CONFIG['NGBOOST_PREDICTIONS_FILE'])
    except FileNotFoundError as e:
        logging.error(f"❌ FATAL: Input file not found. Ensure all backtests have been run. Details: {e}")
        sys.exit(1)

    df_merged = df_ensemble.rename(
        columns={'predicted_yield_median': 'Hybrid Ensemble_pred', 'predicted_yield_lower': 'Hybrid Ensemble_lower',
                 'predicted_yield_upper': 'Hybrid Ensemble_upper'})
    df_merged['Hybrid (TS+XGB)_pred'] = df_xgb['predicted_yield_median']
    df_merged['Hybrid (TS+XGB)_lower'] = df_xgb['predicted_yield_lower']
    df_merged['Hybrid (TS+XGB)_upper'] = df_xgb['predicted_yield_upper']
    df_merged['Adaptive CQR_pred'] = df_cqr['predicted_yield_median']
    df_merged['Adaptive CQR_lower'] = df_cqr['predicted_yield_lower']
    df_merged['Adaptive CQR_upper'] = df_cqr['predicted_yield_upper']
    df_merged['NGBoost_pred'] = df_ngb['predicted_yield_median']
    df_merged['NGBoost_lower'] = df_ngb['predicted_yield_lower']
    df_merged['NGBoost_upper'] = df_ngb['predicted_yield_upper']

    models_point = ["Hybrid Ensemble", "Hybrid (TS+XGB)", "Adaptive CQR", "NGBoost"]
    point_results = [{'Model': name, 'MAE': (df_merged[f'{name}_pred'] - df_merged['kreisYield']).abs().mean(),
                      'R-squared': r2_score(df_merged['kreisYield'], df_merged[f'{name}_pred'])} for name in
                     models_point]
    df_summary = pd.DataFrame(point_results).sort_values('MAE')
    print("\n" + "=" * 80);
    print("      POINT FORECAST ACCURACY");
    print("=" * 80);
    print(df_summary.to_string(index=False, float_format="%.4f"));
    print("=" * 80)

    models_quantile = ["Hybrid Ensemble", "Adaptive CQR", "Hybrid (TS+XGB)", "NGBoost"]
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
    print("=" * 80);
    print(df_quantile_summary.to_string(index=False, float_format="%.4f"));
    print("=" * 80)

    logging.info("Generating final comparison plots...")
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

    models_to_plot = {
        "NGBoost": {'marker': 'D', 'color': 'firebrick', 'linewidth': 2.0},
        "Hybrid (TS+XGB)": {'marker': 's', 'color': 'darkorange', 'linewidth': 2.5},
        "Adaptive CQR": {'marker': '*', 'markersize': 10, 'color': 'darkgreen', 'linewidth': 2.0},
        "Hybrid Ensemble": {'marker': 'P', 'markersize': 12, 'color': 'purple', 'linewidth': 3.5}
    }

    plot_accuracy_comparison(yearly_stats, models_to_plot, start_year, end_year)
    plot_sharpness_comparison(yearly_stats, models_to_plot, df_quantile_summary, start_year, end_year)
    plot_score_comparison(yearly_stats, models_to_plot, start_year, end_year)

    logging.info(f"✓ Final comparison plots saved successfully to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
