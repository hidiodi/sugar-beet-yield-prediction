import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import sys

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = config.MODEL_COMPARISON_CONFIG
NOMINAL_COVERAGE_PERCENT = CONFIG['NOMINAL_COVERAGE_PERCENT']
ALPHA = 1 - (NOMINAL_COVERAGE_PERCENT / 100.0)
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])

# Additional File Paths
NATIVE_COMPARISON_PATH = project_root / 'src/models/native_physics_comparison/native_model_comparison_v2_v8.csv'
NATIVE_ENSEMBLE_PATH = project_root / 'src/models/native_ensemble_champion/native_ensemble_forecasts.csv'
SWITCHED_MODEL_PATH = config.DATA_DIR / '06_model_output' / 'final_switched_forecast.csv'


def calculate_interval_score(y_true, lower, upper, alpha):
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return width + penalty_lower + penalty_upper


def load_and_merge_models():
    # 1. Base: Hybrid Model
    base_path = Path(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
    if not base_path.exists():
        logging.error("Hybrid XGB predictions not found.")
        return pd.DataFrame()

    df = pd.read_csv(base_path)
    df.rename(columns={
        'predicted_yield_median': 'Hybrid XGB_pred',
        'predicted_yield_lower': 'Hybrid XGB_lower',
        'predicted_yield_upper': 'Hybrid XGB_upper'
    }, inplace=True)

    # 2. Merge Standalone
    sa_path = Path(CONFIG['STANDALONE_XGB_PREDICTIONS_FILE'])
    if sa_path.exists():
        df_sa = pd.read_csv(sa_path)
        df = pd.merge(df, df_sa[
            ['year', 'district_no', 'predicted_yield_median', 'predicted_yield_lower', 'predicted_yield_upper']],
                      on=['year', 'district_no'], suffixes=('', '_sa'),
                      how='left')
        df.rename(columns={
            'predicted_yield_median': 'Standalone XGB_pred',
            'predicted_yield_lower': 'Standalone XGB_lower',
            'predicted_yield_upper': 'Standalone XGB_upper'
        }, inplace=True)

    # 3. Merge Statistical Trend
    trend_path = Path(CONFIG['STATISTICAL_TREND_FILE'])
    if trend_path.exists():
        df_trend = pd.read_csv(trend_path)
        df = pd.merge(df, df_trend[['year', 'district_no', 'final_corrected_forecast']],
                      on=['year', 'district_no'], how='left')
        df.rename(columns={'final_corrected_forecast': 'Statistical Trend_pred'}, inplace=True)

    # 4. Merge Native Physics V2 & V8
    if NATIVE_COMPARISON_PATH.exists():
        df_native = pd.read_csv(NATIVE_COMPARISON_PATH)
        df_native['district_no'] = df_native['district_no'].astype(int)
        df = pd.merge(df, df_native[['year', 'district_no', 'Native_V2', 'Native_V8']],
                      on=['year', 'district_no'], how='left')
        df.rename(columns={
            'Native_V2': 'Native V2 (Anchored)_pred',
            'Native_V8': 'Native V8 (Unanchored)_pred'
        }, inplace=True)

    # 5. Merge Native Ensemble
    if NATIVE_ENSEMBLE_PATH.exists():
        df_ensemble = pd.read_csv(NATIVE_ENSEMBLE_PATH)
        df_ensemble['district_no'] = df_ensemble['district_no'].astype(int)
        df = pd.merge(df, df_ensemble[['year', 'district_no', 'Ensemble_Pred']],
                      on=['year', 'district_no'], how='left')
        df.rename(columns={'Ensemble_Pred': 'Native Ensemble_pred'}, inplace=True)

    # 6. Merge Regime Switch V10 (Now includes Strategy_Mode)
    if SWITCHED_MODEL_PATH.exists():
        df_switch = pd.read_csv(SWITCHED_MODEL_PATH)
        df_switch['district_no'] = df_switch['district_no'].astype(int)

        # Load Strategy Mode if available
        cols = ['year', 'district_no', 'Final_Pred']
        if 'Strategy_Mode' in df_switch.columns:
            cols.append('Strategy_Mode')

        df = pd.merge(df, df_switch[cols], on=['year', 'district_no'], how='left')
        df.rename(columns={'Final_Pred': 'Regime Switch V10_pred'}, inplace=True)
        logging.info("✓ Regime Switch V10 loaded (with Strategy Modes).")

    return df


def analyze_strategy_effectiveness(df):
    """New: Breakdown of performance by Strategy Mode (V10 feature)."""
    if 'Strategy_Mode' not in df.columns or 'Regime Switch V10_pred' not in df.columns:
        return

    logging.info("\n" + "=" * 80)
    logging.info("      STRATEGY MODE ANALYSIS (The 'Switch' Effectiveness)")
    logging.info("=" * 80)

    # Define required columns for this analysis
    req_cols = ['Strategy_Mode', 'Regime Switch V10_pred', 'kreisYield', 'Native Ensemble_pred']

    # Ensure they exist in the dataframe
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        logging.warning(f"Skipping Strategy Analysis due to missing columns: {missing}")
        return

    # Drop NaNs only for the columns we are actually using here to prevent crashes
    clean = df.dropna(subset=req_cols).copy()

    if clean.empty:
        logging.warning("No overlapping data found for Strategy Analysis.")
        return

    # Iterate manually to compute metrics (safer than complex lambdas)
    results = []
    for mode, group in clean.groupby('Strategy_Mode'):
        count = len(group)
        mae_switch = mean_absolute_error(group['kreisYield'], group['Regime Switch V10_pred'])
        mae_base = mean_absolute_error(group['kreisYield'], group['Native Ensemble_pred'])
        gain = mae_base - mae_switch
        results.append({
            'Strategy Mode': mode,
            'Count': count,
            'MAE': mae_switch,
            'Base MAE': mae_base,
            'Gain': gain
        })

    # Sort for readability (maybe by Count or Gain)
    res_df = pd.DataFrame(results).sort_values('Count', ascending=False)

    logging.info(f"{'Strategy Mode':<35} | {'Count':<6} | {'MAE':<6} | {'Base MAE':<8} | {'Gain':<6}")
    logging.info("-" * 80)

    for _, row in res_df.iterrows():
        logging.info(
            f"{row['Strategy Mode']:<35} | {int(row['Count']):<6} | {row['MAE']:.2f}  | {row['Base MAE']:.2f}     | {row['Gain']:+.2f}")


def print_anomaly_forensics(df):
    """Checks specific years known to be difficult."""
    logging.info("\n" + "=" * 80)
    logging.info("      ANOMALY FORENSICS (Did we catch the Black Swans?)")
    logging.info("=" * 80)

    anomalies = [2014, 2018]
    model_map = {
        'TREND': 'Statistical Trend_pred',
        'STANDALONE': 'Standalone XGB_pred',
        'HYBRID': 'Hybrid XGB_pred',
        'NATIVE_V2': 'Native V2 (Anchored)_pred',
        'NATIVE_V8': 'Native V8 (Unanchored)_pred',
        'NATIVE_ENSEMBLE': 'Native Ensemble_pred',
        'REGIME_SWITCH': 'Regime Switch V10_pred'
    }

    for year in anomalies:
        if year not in df['year'].values: continue

        subset = df[df['year'] == year].copy()
        actual = subset['kreisYield'].mean()

        # Determine the active strategy for this year
        mode_info = ""
        if 'Strategy_Mode' in subset.columns:
            mode = subset['Strategy_Mode'].mode()
            if not mode.empty:
                mode_info = f" [Mode: {mode[0]}]"

        logging.info(f"YEAR {year} (Actual: {actual:.1f} dt/ha){mode_info}")

        errors = {}
        for key, col in model_map.items():
            if col in subset.columns:
                errors[key] = (subset[col] - subset['kreisYield']).abs().mean()

        sorted_errors = sorted(errors.items(), key=lambda x: x[1])

        for m, err in sorted_errors:
            marker = "  >"
            # Highlight our V10 model
            if m == 'REGIME_SWITCH': marker = "  >>"
            logging.info(f"{marker} {m:<18}: {err:.1f}")

        logging.info("-" * 40)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_merge_models()
    if df.empty:
        logging.error("No data loaded.")
        return

    # 1. Point Accuracy
    models = [
        'Statistical Trend',
        'Standalone XGB',
        'Hybrid XGB',
        'Native V2 (Anchored)',
        'Native V8 (Unanchored)',
        'Native Ensemble',
        'Regime Switch V10'
    ]

    results = []
    for m in models:
        col = f'{m}_pred'
        if col in df.columns:
            clean = df.dropna(subset=[col, 'kreisYield'])
            if clean.empty: continue
            mae = mean_absolute_error(clean['kreisYield'], clean[col])
            r2 = r2_score(clean['kreisYield'], clean[col])
            results.append({'Model': m, 'MAE': mae, 'R2': r2})

    res_df = pd.DataFrame(results).sort_values('MAE')
    logging.info("\n" + "=" * 80)
    logging.info("      OVERALL POINT ACCURACY (2000-2024)")
    logging.info("=" * 80)
    logging.info(res_df.to_string(index=False, float_format="%.4f"))

    # 2. Strategy Analysis
    analyze_strategy_effectiveness(df)

    # 3. Anomaly Check
    if 'Statistical Trend_pred' in df.columns:
        print_anomaly_forensics(df)


if __name__ == '__main__':
    main()