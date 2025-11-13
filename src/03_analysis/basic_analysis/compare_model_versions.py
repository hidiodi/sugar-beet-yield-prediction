# File: src/03_analysis/compare_model_versions.py
# Description: (V3 - FIXED) Generates final comparison plots for all models.
#              Correctly loads and labels "Statistical Trend" and "Pure WOFOST"
#              as separate point-forecast baselines.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import sys
import math

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = config.MODEL_COMPARISON_CONFIG
WOFOST_CONFIG = config.WOFOST_CONFIG
NOMINAL_COVERAGE_PERCENT = CONFIG['NOMINAL_COVERAGE_PERCENT']
ALPHA = 1 - (NOMINAL_COVERAGE_PERCENT / 100.0)
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])


# --- Function Definitions ---
def calculate_interval_score(y_true, lower, upper, alpha):
    """Calculates the Winkler Interval Score."""
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return width + penalty_lower + penalty_upper


def plot_accuracy_comparison(yearly_stats, models, start_year, end_year):
    """Generates the MAE comparison plot with a proper legend."""
    if not models:
        logging.warning("No models found for accuracy plot. Skipping.")
        return

    plt.figure(figsize=(18, 8))
    for name, spec in models.items():
        if f'{name}_mae' in yearly_stats.columns:
            plot_spec = spec.copy()
            plot_spec['label'] = name
            plt.plot(yearly_stats['year'], yearly_stats[f'{name}_mae'], **plot_spec)

    plt.title(f'Point Accuracy Comparison: Mean Absolute Error ({start_year}-{end_year})', fontsize=20)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend(fontsize=14, title="Model", title_fontsize=14)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(yearly_stats['year'][::2], rotation=45, ha="right")  # Show every 2nd year
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'comparison_01_accuracy_{start_year}-{end_year}.png', dpi=300)
    plt.close()


def plot_sharpness_comparison(yearly_stats, models_quantile, df_quantile_summary, start_year, end_year):
    """Generates the Interval Width comparison plot."""
    if not models_quantile:
        logging.warning("No quantile models found for sharpness plot. Skipping.")
        return

    plt.figure(figsize=(18, 8))
    for name, spec in models_quantile.items():
        if f'{name}_width' in yearly_stats.columns:
            plot_spec = spec.copy()
            try:
                coverage = df_quantile_summary.loc[df_quantile_summary['Model'] == name, 'Coverage (%)'].iloc[0]
                plot_spec['label'] = f'{name} — Coverage: {coverage:.1f}%'
            except IndexError:
                plot_spec['label'] = f'{name} (Coverage N/A)'
            plt.plot(yearly_stats['year'], yearly_stats[f'{name}_width'], **plot_spec)

    plt.title(f'Interval Sharpness Comparison: Average Width ({start_year}-{end_year})', fontsize=20)
    plt.ylabel('Average Interval Width', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend(fontsize=14, title="Model & Achieved Coverage", title_fontsize=14)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(yearly_stats['year'][::2], rotation=45, ha="right")  # Show every 2nd year
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'comparison_02_sharpness_{start_year}-{end_year}.png', dpi=300)
    plt.close()


def plot_score_comparison(yearly_stats, models_quantile, start_year, end_year):
    """Generates the Interval Score comparison plot."""
    if not models_quantile:
        logging.warning("No quantile models found for score plot. Skipping.")
        return

    plt.figure(figsize=(18, 8))
    for name, spec in models_quantile.items():
        if f'{name}_score' in yearly_stats.columns:
            plot_spec = spec.copy()
            plot_spec['label'] = name
            plt.plot(yearly_stats['year'], yearly_stats[f'{name}_score'], **plot_spec)

    plt.title(f'Overall Interval Quality: {int(NOMINAL_COVERAGE_PERCENT)}% Interval Score ({start_year}-{end_year})',
              fontsize=20)
    plt.ylabel('Mean Interval Score (Lower is Better)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend(fontsize=14, title="Model", title_fontsize=14)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(yearly_stats['year'][::2], rotation=45, ha="right")  # Show every 2nd year
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'comparison_03_score_{start_year}-{end_year}.png', dpi=300)
    plt.close()


def plot_national_average_timelines(national_avg_df, models_quantile, start_year, end_year, output_dir,
                                    nominal_coverage):
    """
    Generates a dynamic grid plot comparing national average actual yield vs.
    predicted intervals for all available quantile models.
    """
    logging.info("Generating national average timeline comparison plot...")

    if not models_quantile:
        logging.warning("No models found for national average timeline plot. Skipping.")
        return

    n_models = len(models_quantile)
    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11 * n_cols, 9 * n_rows), sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    years = national_avg_df['year']
    actual_yield = national_avg_df['kreisYield']

    for i, model_name in enumerate(models_quantile):
        ax = axes[i]
        median_pred = national_avg_df[f'{model_name}_pred']
        lower_pred = national_avg_df[f'{model_name}_lower']
        upper_pred = national_avg_df[f'{model_name}_upper']

        ax.fill_between(years, lower_pred, upper_pred, color='pink', alpha=0.6,
                        label=f'{int(nominal_coverage)}% Prediction Interval')
        ax.plot(years, actual_yield, 'o-', color='darkblue', linewidth=2,
                label='National Average Actual Yield', markersize=8)
        ax.plot(years, median_pred, '--', color='red', linewidth=2.0,
                label='Median Prediction')

        ax.set_title(model_name, fontsize=18, fontweight='bold')
        ax.grid(True, linestyle=':', which='both', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=12)
        handles, labels = ax.get_legend_handles_labels()
        order = [1, 2, 0]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14, loc='upper left')
        ax.set_xticks(np.arange(start_year, end_year + 1, 5))
        ax.tick_params(axis='x', rotation=0)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'National Average Yield vs. Predicted Interval ({start_year}-{end_year})', fontsize=26, y=1.03)
    fig.text(0.5, 0.04, 'Year', ha='center', va='center', fontsize=20)
    fig.text(0.04, 0.5, 'Yield (dt/ha)', ha='center', va='center', rotation='vertical', fontsize=20)
    plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.96])

    save_path = output_dir / f'comparison_04_national_average_timeline_{start_year}-{end_year}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def load_quantile_model_data(config_key, model_name, base_df):
    """Helper function to load quantile model data and merge it."""
    file_path = Path(CONFIG.get(config_key))
    if not file_path or not file_path.exists():
        logging.warning(f"File not found for model: {model_name}. Skipping.")
        return base_df, None, None

    logging.info(f"Loading data for model: {model_name}")
    df = pd.read_csv(file_path)

    cols_to_merge = ['year', 'district_no', 'predicted_yield_median', 'predicted_yield_lower', 'predicted_yield_upper']
    if not all(col in df.columns for col in cols_to_merge):
        logging.error(f"File for {model_name} is missing required columns. Skipping.")
        return base_df, None, None

    rename_dict = {
        'predicted_yield_median': f'{model_name}_pred',
        'predicted_yield_lower': f'{model_name}_lower',
        'predicted_yield_upper': f'{model_name}_upper'
    }
    df = df[cols_to_merge].rename(columns=rename_dict)

    base_df = pd.merge(base_df, df, on=['year', 'district_no'], how='left')
    return base_df, model_name, model_name  # Returns names for point and quantile lists


def load_statistical_trend(base_df):
    """Loads the statistical trend model (Script 2)"""
    model_name = "Statistical Trend"
    file_path = Path(CONFIG.get('STATISTICAL_TREND_FILE'))
    if not file_path or not file_path.exists():
        logging.warning(f"File not found for model: {model_name}. Skipping.")
        return base_df, None

    logging.info(f"Loading data for model: {model_name}")
    df = pd.read_csv(file_path)
    cols_to_merge = ['year', 'district_no', 'final_corrected_forecast']
    if not all(col in df.columns for col in cols_to_merge):
        logging.error(f"File for {model_name} is missing 'final_corrected_forecast'. Skipping.")
        return base_df, None

    df.rename(columns={'final_corrected_forecast': f'{model_name}_pred'}, inplace=True)
    base_df = pd.merge(base_df, df[['year', 'district_no', f'{model_name}_pred']], on=['year', 'district_no'],
                       how='left')
    return base_df, model_name


def load_pure_wofost(base_df):
    """Loads the 'Pure WOFOST' model (Script 1)"""
    model_name = "Pure WOFOST"
    file_path = Path(CONFIG.get('PURE_WOFOST_ENSEMBLE_FILE'))
    if not file_path or not file_path.exists():
        logging.warning(f"File not found for model: {model_name}. Skipping.")
        return base_df, None

    logging.info(f"Loading data for model: {model_name} (aggregating from ensemble...)")
    df_wofost_raw = pd.read_csv(file_path)

    # Aggregate, convert, and rename
    df_wofost_agg = df_wofost_raw.groupby(['year', 'district_no']).agg(
        wofost_pred_dry_kgha=('yield_water_limited_dry_kgha', 'mean')
    ).reset_index()

    DMC_SUGARBEET = WOFOST_CONFIG['CONSTANTS']['DMC_SUGARBEET']
    df_wofost_agg[f'{model_name}_pred'] = (df_wofost_agg['wofost_pred_dry_kgha'] / DMC_SUGARBEET) / 100.0

    # Standardize keys for merging
    df_wofost_agg['district_no'] = df_wofost_agg['district_no'].astype(str).str.zfill(5)
    df_wofost_agg['year'] = df_wofost_agg['year'].astype(int)

    # Need to make base_df keys compatible
    base_df['year'] = base_df['year'].astype(int)
    base_df['district_no'] = base_df['district_no'].astype(str).str.zfill(5)

    base_df = pd.merge(base_df, df_wofost_agg[['year', 'district_no', f'{model_name}_pred']],
                       on=['year', 'district_no'], how='left')
    return base_df, model_name


def main():
    """Main function to execute the model comparison workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("--- Loading Final Model Predictions for Comparison ---")

    # --- Data Merging and Preparation ---
    base_file_path = Path(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
    if not base_file_path.exists():
        logging.error(f"❌ FATAL: Base 'Hybrid XGB' file not found at {base_file_path}. Cannot proceed.")
        sys.exit(1)

    df_hybrid_xgb = pd.read_csv(base_file_path)
    df_merged = df_hybrid_xgb[['year', 'district_no', 'kreisYield']].drop_duplicates().copy()

    # Define model name and rename dict for the base model
    model_name = "Hybrid XGB"
    df_hybrid_xgb.rename(columns={
        'predicted_yield_median': f'{model_name}_pred',
        'predicted_yield_lower': f'{model_name}_lower',
        'predicted_yield_upper': f'{model_name}_upper'
    }, inplace=True)

    # Merge base model (which has intervals)
    df_merged = pd.merge(df_merged, df_hybrid_xgb[
        ['year', 'district_no', f'{model_name}_pred', f'{model_name}_lower', f'{model_name}_upper']],
                         on=['year', 'district_no'], how='left')

    models_point = [model_name]  # For MAE/R2
    models_quantile = [model_name]  # For Interval plots

    # --- Dynamically load all other quantile models ---
    model_definitions = {
        'STANDALONE_XGB_PREDICTIONS_FILE': "Standalone XGB",
        'ADAPTIVE_CQR_PREDICTIONS_FILE': "Adaptive CQR",
        'NGBOOST_PREDICTIONS_FILE': "NGBoost"
    }

    for config_key, model_name in model_definitions.items():
        df_merged, point_name, quantile_name = load_quantile_model_data(config_key, model_name, df_merged)
        if point_name:
            models_point.append(point_name)
        if quantile_name:
            models_quantile.append(quantile_name)

    # --- Load Point-Forecast Baselines ---
    df_merged, stat_trend_name = load_statistical_trend(df_merged)
    if stat_trend_name:
        models_point.append(stat_trend_name)

    df_merged, pure_wofost_name = load_pure_wofost(df_merged)
    if pure_wofost_name:
        models_point.append(pure_wofost_name)

    # --- Performance Summaries ---
    point_results = []
    for name in models_point:
        if f'{name}_pred' in df_merged.columns:
            df_comp = df_merged[['kreisYield', f'{name}_pred']].dropna()
            if not df_comp.empty:
                point_results.append({
                    'Model': name,
                    'MAE': mean_absolute_error(df_comp['kreisYield'], df_comp[f'{name}_pred']),
                    'R-squared': r2_score(df_comp['kreisYield'], df_comp[f'{name}_pred'])
                })

    df_summary = pd.DataFrame(point_results).sort_values('MAE')
    print("\n" + "=" * 80);
    print("      POINT FORECAST ACCURACY");
    print("=" * 80)
    print(df_summary.to_string(index=False, float_format="%.4f"));
    print("=" * 80)

    quantile_results = []
    for name in models_quantile:
        if f'{name}_lower' in df_merged.columns:
            df_comp = df_merged[['kreisYield', f'{name}_lower', f'{name}_upper']].dropna()
            if not df_comp.empty:
                score = calculate_interval_score(df_comp['kreisYield'], df_comp[f'{name}_lower'],
                                                 df_comp[f'{name}_upper'], ALPHA).mean()
                coverage = ((df_comp['kreisYield'] >= df_comp[f'{name}_lower']) & (
                        df_comp['kreisYield'] <= df_comp[f'{name}_upper'])).mean() * 100
                width = (df_comp[f'{name}_upper'] - df_comp[f'{name}_lower']).mean()
                quantile_results.append(
                    {'Model': name, 'Interval Score': score, 'Coverage (%)': coverage, 'Width': width})

    df_quantile_summary = pd.DataFrame(quantile_results).sort_values('Interval Score')
    print("\n" + "=" * 80);
    print(f"      PREDICTION INTERVAL QUALITY ({int(NOMINAL_COVERAGE_PERCENT)}%)");
    print("=" * 80)
    print(df_quantile_summary.to_string(index=False, float_format="%.4f"));
    print("=" * 80)

    # --- Aggregate national average yields ---
    logging.info("Aggregating national average yields for timeline plot...")
    avg_cols = ['year', 'kreisYield']
    # Add all available model predictions (point and quantile)
    for name in models_point:
        avg_cols.append(f'{name}_pred')
    for name in models_quantile:
        avg_cols.extend([f'{name}_lower', f'{name}_upper'])

    valid_avg_cols = list(set([col for col in avg_cols if col in df_merged.columns]))  # Use set for unique
    national_avg_df = df_merged[valid_avg_cols].groupby('year').mean().reset_index()

    # --- Yearly Statistics for Metric Plotting ---
    logging.info("Aggregating metrics for yearly plots...")
    for name in models_point:
        if f'{name}_pred' in df_merged.columns:
            df_merged[f'{name}_mae'] = (df_merged[f'{name}_pred'] - df_merged['kreisYield']).abs()
    for name in models_quantile:
        if f'{name}_lower' in df_merged.columns:
            df_merged[f'{name}_width'] = df_merged[f'{name}_upper'] - df_merged[f'{name}_lower']
            df_merged[f'{name}_score'] = calculate_interval_score(df_merged['kreisYield'], df_merged[f'{name}_lower'],
                                                                  df_merged[f'{name}_upper'], ALPHA)

    agg_dict = {f'{name}_mae': 'mean' for name in models_point if f'{name}_mae' in df_merged.columns}
    agg_dict.update({f'{name}_width': 'mean' for name in models_quantile if f'{name}_width' in df_merged.columns})
    agg_dict.update({f'{name}_score': 'mean' for name in models_quantile if f'{name}_score' in df_merged.columns})

    if not agg_dict:
        logging.error("No valid model data to aggregate for yearly stats. Stopping.")
        return

    yearly_stats = df_merged.groupby('year').agg(agg_dict).reset_index()
    start_year, end_year = yearly_stats['year'].min(), yearly_stats['year'].max()

    # --- Plotting ---
    models_to_plot_styles = {
        "Hybrid XGB": {'marker': 's', 'color': 'darkorange', 'linewidth': 2.5},
        "Standalone XGB": {'marker': 'p', 'color': 'mediumorchid', 'linewidth': 2.0},
        "Adaptive CQR": {'marker': '*', 'markersize': 10, 'color': 'darkgreen', 'linewidth': 2.0},
        "NGBoost": {'marker': 'D', 'color': 'firebrick', 'linewidth': 2.0},
        "Statistical Trend": {'marker': 'o', 'color': 'black', 'linewidth': 2.0, 'linestyle': '--'},  # NEW
        "Pure WOFOST": {'marker': '^', 'color': 'blue', 'linewidth': 2.0, 'linestyle': ':'},  # NEW
    }

    models_point_plot = {k: v for k, v in models_to_plot_styles.items() if k in models_point}
    models_quantile_plot = {k: v for k, v in models_to_plot_styles.items() if k in models_quantile}

    plot_accuracy_comparison(yearly_stats, models_point_plot, start_year, end_year)
    plot_sharpness_comparison(yearly_stats, models_quantile_plot, df_quantile_summary, start_year, end_year)
    plot_score_comparison(yearly_stats, models_quantile_plot, start_year, end_year)
    plot_national_average_timelines(national_avg_df, models_quantile_plot.keys(), start_year, end_year, OUTPUT_DIR,
                                    NOMINAL_COVERAGE_PERCENT)

    logging.info(f"✓ Final comparison plots saved successfully to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()