import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
import importlib

config = importlib.import_module("src.03_analysis.config")
analysis_config = importlib.import_module("src.03_analysis.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])

# Strict Paths to the New Stacking Architecture
TREND_PATH = config.DATA_DIR / "05_model_input/wofost_walkforward/final_honest_forecasts.csv"
XGB_PATH = Path(analysis_config.STANDALONE_BACKTESTING_CONFIG['REPORT_DIR']) / 'full_backtest_predictions.csv'
LINEAR_PATH = config.DATA_DIR / '06_model_output/recovery_models/model_c_linear_forecasts.csv'
SUPER_ENSEMBLE_PATH = OUTPUT_DIR / 'super_ensemble_final_forecast_TSCV.csv'


def load_and_merge_models():
    df = pd.DataFrame()

    # 1. Statistical Trend (The Baseline)
    if TREND_PATH.exists():
        df_trend = pd.read_csv(TREND_PATH)
        df = df_trend[['year', 'district_no', 'final_corrected_forecast', 'actual_yield']].copy()
        df.rename(columns={'final_corrected_forecast': 'Statistical Trend_pred', 'actual_yield': 'kreisYield'},
                  inplace=True)
        df['district_no'] = df['district_no'].astype(int)

    # 2. Model A: Standalone XGBoost
    if XGB_PATH.exists():
        df_xgb = pd.read_csv(XGB_PATH)
        df_xgb['district_no'] = df_xgb['district_no'].astype(int)
        if not df.empty:
            df = pd.merge(df, df_xgb[['year', 'district_no', 'predicted_yield_median']], on=['year', 'district_no'],
                          how='left')
        else:
            df = df_xgb[['year', 'district_no', 'predicted_yield_median', 'kreisYield']].copy()
        df.rename(columns={'predicted_yield_median': 'Model A (XGBoost)_pred'}, inplace=True)

    # 3. Model C: Robust Ridge
    if LINEAR_PATH.exists():
        df_lin = pd.read_csv(LINEAR_PATH)
        df_lin['district_no'] = df_lin['district_no'].astype(int)
        df = pd.merge(df, df_lin[['year', 'district_no', 'linear_pred']], on=['year', 'district_no'], how='left')
        df.rename(columns={'linear_pred': 'Model C (Ridge)_pred'}, inplace=True)

    # 4. Super Ensemble (The Stack)
    if SUPER_ENSEMBLE_PATH.exists():
        df_sup = pd.read_csv(SUPER_ENSEMBLE_PATH)
        df_sup['district_no'] = df_sup['district_no'].astype(int)
        df = pd.merge(df, df_sup[['year', 'district_no', 'Super_Ensemble_pred']], on=['year', 'district_no'],
                      how='left')
        df.rename(columns={'Super_Ensemble_pred': 'Super Ensemble_pred'}, inplace=True)

    return df


def evaluate_timeframe(df, start_year, end_year, title):
    """Calculates strict apples-to-apples metrics for a specific time window."""
    mask = (df['year'] >= start_year) & (df['year'] <= end_year)
    subset = df[mask].copy()

    if subset.empty:
        logging.warning(f"No data for {title}")
        return

    models = [
        'Model A (XGBoost)',
        'Model C (Ridge)',
        'Super Ensemble'
    ]

    results = []

    # Calculate Trend MAE Baseline
    clean_trend = subset.dropna(subset=['Statistical Trend_pred', 'kreisYield'])
    trend_mae_global = mean_absolute_error(clean_trend['kreisYield'], clean_trend['Statistical Trend_pred'])

    # Add Trend to results
    results.append({
        'Model': 'Statistical Trend',
        'MAE': trend_mae_global,
        'R2': r2_score(clean_trend['kreisYield'], clean_trend['Statistical Trend_pred']),
        'Skill (%)': 0.00
    })

    for m in models:
        col = f'{m}_pred'
        if col in subset.columns:
            # STRICT: Only evaluate where BOTH the model and the Trend exist to prevent fake skill
            clean = subset.dropna(subset=[col, 'Statistical Trend_pred', 'kreisYield'])
            if clean.empty: continue

            mae = mean_absolute_error(clean['kreisYield'], clean[col])
            r2 = r2_score(clean['kreisYield'], clean[col])

            # Recalculate exact comparative trend MAE for this specific subset
            exact_trend_mae = mean_absolute_error(clean['kreisYield'], clean['Statistical Trend_pred'])

            # Skill Score: % Improvement over Trend
            skill = 0.0
            if exact_trend_mae > 0:
                skill = (1 - (mae / exact_trend_mae)) * 100

            results.append({
                'Model': m,
                'MAE': mae,
                'R2': r2,
                'Skill (%)': skill
            })

    res_df = pd.DataFrame(results).sort_values('MAE')

    logging.info("\n" + "=" * 80)
    logging.info(f"      {title}  (N={len(subset)})")
    logging.info("=" * 80)
    logging.info(res_df.to_string(index=False, float_format="%.4f"))


def print_anomaly_forensics(df):
    logging.info("\n" + "=" * 80)
    logging.info("      ANOMALY FORENSICS (Black Swan Events)")
    logging.info("=" * 80)

    anomalies = [2003, 2014, 2018, 2022]
    model_map = {
        'TREND': 'Statistical Trend_pred',
        'MODEL_A_XGB': 'Model A (XGBoost)_pred',
        'MODEL_C_RIDGE': 'Model C (Ridge)_pred',
        'SUPER_ENSEMBLE': 'Super Ensemble_pred'
    }

    for year in anomalies:
        if year not in df['year'].values: continue
        subset = df[df['year'] == year].copy()
        actual = subset['kreisYield'].mean()

        logging.info(f"YEAR {year} (Actual National Yield: {actual:.1f} dt/ha)")

        errors = []
        for label, col in model_map.items():
            if col in subset.columns:
                pred_mean = subset[col].mean()
                mae = (subset[col] - subset['kreisYield']).abs().mean()
                errors.append((label, pred_mean, mae))

        errors.sort(key=lambda x: x[2])

        for label, pred, mae in errors:
            marker = "  "
            if label == 'SUPER_ENSEMBLE': marker = "->"
            logging.info(f"{marker} {label:<16}: Pred {pred:.1f} (Avg MAE: {mae:.1f})")

        logging.info("-" * 40)


def plot_time_series(df):
    output_path = project_root / 'docs/paper_latex/figures/fig2_time_series.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate to National Level
    plot_cols = ['year', 'kreisYield', 'Statistical Trend_pred']
    if 'Super Ensemble_pred' in df.columns: plot_cols.append('Super Ensemble_pred')
    if 'Model A (XGBoost)_pred' in df.columns: plot_cols.append('Model A (XGBoost)_pred')

    national = df.groupby('year')[[c for c in plot_cols if c != 'year']].mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(national['year'], national['kreisYield'], 'k-o', label='Actual Yield', linewidth=2.5)
    plt.plot(national['year'], national['Statistical Trend_pred'], 'b--', label='Statistical Trend (Baseline)')

    if 'Model A (XGBoost)_pred' in national.columns:
        plt.plot(national['year'], national['Model A (XGBoost)_pred'], 'g-s', label='Model A (XGBoost)', alpha=0.5)

    if 'Super Ensemble_pred' in national.columns:
        plt.plot(national['year'], national['Super Ensemble_pred'], 'r-^', label='Super Ensemble (Stack)', alpha=0.9,
                 linewidth=2)

    plt.title('National Sugarbeet Yield Performance: Trend vs Stacking Ensemble', fontsize=14)
    plt.ylabel('Yield (dt/ha)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logging.info(f"✅ Time Series Figure saved to: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_merge_models()
    if df.empty:
        logging.error("No data loaded. Check file paths.")
        return

    # 1. Long-Term Stability Test (2000-2024)
    evaluate_timeframe(df, 2005, 2024, "LONG-TERM STABILITY (2005-2024)")

    # 2. Recent Volatility Test (2014-2024)
    evaluate_timeframe(df, 2014, 2024, "RECENT VOLATILITY (2014-2024)")

    # 3. Anomaly Check
    print_anomaly_forensics(df)

    # 4. Plot Time Series
    plot_time_series(df)


if __name__ == '__main__':
    main()