import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import sys
import numpy as np

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])

# Additional File Paths
NATIVE_ENSEMBLE_PATH = project_root / 'src/models/native_ensemble_champion/native_ensemble_forecasts.csv'
SWITCHED_MODEL_PATH = config.DATA_DIR / '06_model_output' / 'final_switched_forecast.csv'
SUPER_ENSEMBLE_PATH = OUTPUT_DIR / 'super_ensemble_final_forecast_TSCV.csv'
ROBUST_LINEAR_PATH = config.DATA_DIR / '06_model_output' / 'recovery_models' / 'stage2_forecasts.csv'


def load_and_merge_models():
    # 1. Base: Hybrid Model (Legacy)
    base_path = Path(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
    df = pd.DataFrame()
    if base_path.exists():
        df = pd.read_csv(base_path)
        df.rename(columns={'predicted_yield_median': 'Hybrid XGB_pred'}, inplace=True)

    # 2. Statistical Trend (The Baseline)
    trend_path = Path(CONFIG['STATISTICAL_TREND_FILE'])
    if trend_path.exists():
        df_trend = pd.read_csv(trend_path)
        if df.empty:
            df = df_trend[['year', 'district_no', 'final_corrected_forecast', 'kreisYield']].copy()
        else:
            df = pd.merge(df, df_trend[['year', 'district_no', 'final_corrected_forecast']],
                          on=['year', 'district_no'], how='left')
        df.rename(columns={'final_corrected_forecast': 'Statistical Trend_pred'}, inplace=True)

    # 3. Native Ensemble
    if NATIVE_ENSEMBLE_PATH.exists():
        df_ens = pd.read_csv(NATIVE_ENSEMBLE_PATH)
        df_ens['district_no'] = df_ens['district_no'].astype(int)
        df = pd.merge(df, df_ens[['year', 'district_no', 'Ensemble_Pred']],
                      on=['year', 'district_no'], how='left')
        df.rename(columns={'Ensemble_Pred': 'Native Ensemble_pred'}, inplace=True)

    # 4. Regime Switch V10
    if SWITCHED_MODEL_PATH.exists():
        df_sw = pd.read_csv(SWITCHED_MODEL_PATH)
        df_sw['district_no'] = df_sw['district_no'].astype(int)
        df = pd.merge(df, df_sw[['year', 'district_no', 'Final_Pred']],
                      on=['year', 'district_no'], how='left')
        df.rename(columns={'Final_Pred': 'Regime Switch V10_pred'}, inplace=True)

    # 5. Super Ensemble
    if SUPER_ENSEMBLE_PATH.exists():
        df_sup = pd.read_csv(SUPER_ENSEMBLE_PATH)
        df_sup['district_no'] = df_sup['district_no'].astype(int)
        df = pd.merge(df, df_sup[['year', 'district_no', 'Super_Ensemble_pred']],
                      on=['year', 'district_no'], how='left')
        df.rename(columns={'Super_Ensemble_pred': 'Super Ensemble_pred'}, inplace=True)

    # 6. Robust Linear (Stage 2)
    if ROBUST_LINEAR_PATH.exists():
        df_rob = pd.read_csv(ROBUST_LINEAR_PATH)
        df_rob['district_no'] = df_rob['district_no'].astype(int)
        df = pd.merge(df, df_rob[['year', 'district_no', 'stage2_pred']],
                      on=['year', 'district_no'], how='left')
        df.rename(columns={'stage2_pred': 'Robust Linear (Stage 2)_pred'}, inplace=True)

    return df


def evaluate_timeframe(df, start_year, end_year, title):
    """Calculates metrics for a specific time window."""
    mask = (df['year'] >= start_year) & (df['year'] <= end_year)
    subset = df[mask].copy()

    if subset.empty:
        logging.warning(f"No data for {title}")
        return

    models = [
        'Statistical Trend',
        'Native Ensemble',
        'Regime Switch V10',
        'Super Ensemble',
        'Robust Linear (Stage 2)'
    ]

    results = []
    # Calculate Trend MAE first for Skill Score
    trend_mae = 0
    if 'Statistical Trend_pred' in subset.columns:
        clean_t = subset.dropna(subset=['Statistical Trend_pred', 'kreisYield'])
        if not clean_t.empty:
            trend_mae = mean_absolute_error(clean_t['kreisYield'], clean_t['Statistical Trend_pred'])

    for m in models:
        col = f'{m}_pred'
        if col in subset.columns:
            clean = subset.dropna(subset=[col, 'kreisYield'])
            if clean.empty: continue

            mae = mean_absolute_error(clean['kreisYield'], clean[col])
            r2 = r2_score(clean['kreisYield'], clean[col])

            # Skill Score: % Improvement over Trend
            skill = 0.0
            if trend_mae > 0:
                skill = (1 - (mae / trend_mae)) * 100

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

    anomalies = [2003, 2014, 2018]  # Added 2003 for long-term check
    model_map = {
        'TREND': 'Statistical Trend_pred',
        'SUPER_ENSEMBLE': 'Super Ensemble_pred',
        'ROBUST_LINEAR': 'Robust Linear (Stage 2)_pred'
    }

    for year in anomalies:
        if year not in df['year'].values: continue
        subset = df[df['year'] == year].copy()
        actual = subset['kreisYield'].mean()

        logging.info(f"YEAR {year} (Actual: {actual:.1f} dt/ha)")

        # Calculate Error for ranking
        errors = []
        for label, col in model_map.items():
            if col in subset.columns:
                pred_mean = subset[col].mean()
                mae = (subset[col] - subset['kreisYield']).abs().mean()
                errors.append((label, pred_mean, mae))

        # Sort by MAE
        errors.sort(key=lambda x: x[2])

        for label, pred, mae in errors:
            marker = "  "
            if label == 'ROBUST_LINEAR': marker = "->"
            logging.info(f"{marker} {label:<16}: Pred {pred:.1f} (MAE: {mae:.1f})")

        logging.info("-" * 40)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_merge_models()
    if df.empty:
        logging.error("No data loaded.")
        return

    # 1. Long-Term Stability Test (2000-2024)
    evaluate_timeframe(df, 2000, 2024, "LONG-TERM STABILITY (2000-2024)")

    # 2. Recent Volatility Test (2010-2024)
    evaluate_timeframe(df, 2010, 2024, "RECENT VOLATILITY (2010-2024)")

    # 3. Anomaly Check
    print_anomaly_forensics(df)


if __name__ == '__main__':
    main()