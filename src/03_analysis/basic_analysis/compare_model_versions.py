# File: src/03_analysis/basic_analysis/compare_model_versions.py
# Description: Generates final comparison metrics and plots.
#              UPDATED: Adds specific forensic logging for 2014/2018 anomalies.

import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import sys
import math

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')  # Clean format
CONFIG = config.MODEL_COMPARISON_CONFIG
WOFOST_CONFIG = config.WOFOST_CONFIG
NOMINAL_COVERAGE_PERCENT = CONFIG['NOMINAL_COVERAGE_PERCENT']
ALPHA = 1 - (NOMINAL_COVERAGE_PERCENT / 100.0)
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])


def calculate_interval_score(y_true, lower, upper, alpha):
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return width + penalty_lower + penalty_upper


def load_and_merge_models():
    # Base: Hybrid Model
    base_path = Path(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
    if not base_path.exists():
        logging.error("Hybrid XGB predictions not found.")
        sys.exit(1)

    df = pd.read_csv(base_path)
    df.rename(columns={
        'predicted_yield_median': 'Hybrid XGB_pred',
        'predicted_yield_lower': 'Hybrid XGB_lower',
        'predicted_yield_upper': 'Hybrid XGB_upper'
    }, inplace=True)

    # Merge Standalone
    sa_path = Path(CONFIG['STANDALONE_XGB_PREDICTIONS_FILE'])
    if sa_path.exists():
        df_sa = pd.read_csv(sa_path)
        df = pd.merge(df, df_sa[
            ['year', 'district_no', 'predicted_yield_median', 'predicted_yield_lower', 'predicted_yield_upper']],
                      on=['year', 'district_no'], suffixes=('', '_sa'))
        df.rename(columns={
            'predicted_yield_median': 'Standalone XGB_pred',
            'predicted_yield_lower': 'Standalone XGB_lower',
            'predicted_yield_upper': 'Standalone XGB_upper'
        }, inplace=True)

    # Merge Statistical Trend
    trend_path = Path(CONFIG['STATISTICAL_TREND_FILE'])
    if trend_path.exists():
        df_trend = pd.read_csv(trend_path)
        df = pd.merge(df, df_trend[['year', 'district_no', 'final_corrected_forecast']],
                      on=['year', 'district_no'], how='left')
        df.rename(columns={'final_corrected_forecast': 'Statistical Trend_pred'}, inplace=True)

    return df


def print_anomaly_forensics(df):
    """Checks specific years known to be difficult."""
    logging.info("\n" + "=" * 80)
    logging.info("      ANOMALY FORENSICS (Did we catch the Black Swans?)")
    logging.info("=" * 80)

    anomalies = [2014, 2018]

    for year in anomalies:
        if year not in df['year'].values: continue

        subset = df[df['year'] == year].copy()
        actual = subset['kreisYield'].mean()

        # Calculate errors for all available models
        errors = {}

        if 'Statistical Trend_pred' in subset.columns:
            trend_err = (subset['Statistical Trend_pred'] - subset['kreisYield']).abs().mean()
            errors['TREND'] = trend_err

        if 'Standalone XGB_pred' in subset.columns:
            sa_err = (subset['Standalone XGB_pred'] - subset['kreisYield']).abs().mean()
            errors['STANDALONE'] = sa_err

        if 'Hybrid XGB_pred' in subset.columns:
            hybrid_err = (subset['Hybrid XGB_pred'] - subset['kreisYield']).abs().mean()
            errors['HYBRID'] = hybrid_err

        logging.info(f"YEAR {year} (Actual: {actual:.1f} dt/ha)")

        if 'TREND' in errors:
            logging.info(f"  > Trend Model Error:      {errors['TREND']:.1f}")
        if 'STANDALONE' in errors:
            logging.info(f"  > Standalone Model Error: {errors['STANDALONE']:.1f}")
        if 'HYBRID' in errors:
            logging.info(f"  > Hybrid Model Error:     {errors['HYBRID']:.1f}")

        if errors:
            winner = min(errors, key=errors.get)
            best_err = errors[winner]
            # Calculate improvement over Trend if Trend exists, otherwise N/A
            if 'TREND' in errors and winner != 'TREND':
                imp = errors['TREND'] - best_err
                imp_str = f"(Improvement: +{imp:.1f} dt/ha)"
            elif winner == 'TREND':
                imp_str = "(Baseline)"
            else:
                imp_str = ""

            logging.info(f"  > WINNER: {winner} {imp_str}")

        logging.info("-" * 40)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_merge_models()

    # 1. Point Accuracy
    models = ['Statistical Trend', 'Standalone XGB', 'Hybrid XGB']
    results = []
    for m in models:
        if f'{m}_pred' in df.columns:
            clean = df.dropna(subset=[f'{m}_pred', 'kreisYield'])
            mae = mean_absolute_error(clean['kreisYield'], clean[f'{m}_pred'])
            r2 = r2_score(clean['kreisYield'], clean[f'{m}_pred'])
            results.append({'Model': m, 'MAE': mae, 'R2': r2})

    res_df = pd.DataFrame(results).sort_values('MAE')
    logging.info("\n" + "=" * 80)
    logging.info("      OVERALL POINT ACCURACY (2000-2024)")
    logging.info("=" * 80)
    logging.info(res_df.to_string(index=False, float_format="%.4f"))

    # 2. Anomaly Check
    if 'Statistical Trend_pred' in df.columns:
        print_anomaly_forensics(df)

    # 3. Interval Quality
    q_models = ['Standalone XGB', 'Hybrid XGB']
    q_results = []
    for m in q_models:
        if f'{m}_lower' in df.columns:
            clean = df.dropna(subset=[f'{m}_lower', 'kreisYield'])
            score = calculate_interval_score(clean['kreisYield'], clean[f'{m}_lower'], clean[f'{m}_upper'],
                                             ALPHA).mean()
            cov = ((clean['kreisYield'] >= clean[f'{m}_lower']) & (
                        clean['kreisYield'] <= clean[f'{m}_upper'])).mean() * 100
            width = (clean[f'{m}_upper'] - clean[f'{m}_lower']).mean()
            q_results.append({'Model': m, 'Interval Score': score, 'Coverage %': cov, 'Width': width})

    q_df = pd.DataFrame(q_results).sort_values('Interval Score')
    logging.info("\n" + "=" * 80)
    logging.info(f"      UNCERTAINTY & RISK ({int(NOMINAL_COVERAGE_PERCENT)}% Intervals)")
    logging.info("=" * 80)
    logging.info(q_df.to_string(index=False, float_format="%.4f"))
    logging.info("\nNOTE: Lower Interval Score is better. Higher Coverage is better.")

    # (Plots are generated but code omitted here for brevity as requested - keep existing plot code if needed)


if __name__ == '__main__':
    main()