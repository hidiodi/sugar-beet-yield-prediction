import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib

analysis_config = importlib.import_module("src.03_analysis.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
OUTPUT_DIR = Path(analysis_config.MODEL_COMPARISON_CONFIG['OUTPUT_DIR'])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# THRESHOLDS
GATE_THRESHOLD = 0.040  # Lowered to 4.0% deviation to be more responsive
DOWNSIDE_TRUST = 1.0  # 100% Trust. If XGB calls a crash based on physical features, believe it.
UPSIDE_TRUST = 0.50  # 50% Trust. Bumpers are harder to guarantee in March.


def load_oos_predictions():
    df_trend = pd.read_csv(global_config.DATA_DIR / "05_model_input/wofost_walkforward/final_honest_forecasts.csv")
    df_trend = df_trend[['year', 'district_no', 'final_corrected_forecast', 'actual_yield']].rename(
        columns={'final_corrected_forecast': 'trend_pred', 'actual_yield': 'kreisYield'})
    df_trend['district_no'] = df_trend['district_no'].astype(str).str.zfill(5)

    xgb_path = Path(analysis_config.STANDALONE_BACKTESTING_CONFIG['REPORT_DIR']) / 'full_backtest_predictions.csv'
    if xgb_path.exists():
        df_xgb = pd.read_csv(xgb_path)[['year', 'district_no', 'predicted_yield_median']].rename(
            columns={'predicted_yield_median': 'xgb_pred'})
        df_xgb['district_no'] = df_xgb['district_no'].astype(str).str.zfill(5)
        df_trend = pd.merge(df_trend, df_xgb, on=['year', 'district_no'], how='left')

    return df_trend


def generate_gated_forecast():
    logging.info("--- Generating Asymmetric Gated Forecast (Full Trust Mode) ---")
    df = load_oos_predictions()

    df['xgb_pred'] = df['xgb_pred'].fillna(df['trend_pred'])

    def apply_gate(row):
        trend = row['trend_pred']
        xgb = row['xgb_pred']

        delta_pct = (xgb - trend) / trend

        if delta_pct < -GATE_THRESHOLD:
            # Full trust on downside risk
            return trend + ((xgb - trend) * DOWNSIDE_TRUST)
        elif delta_pct > GATE_THRESHOLD:
            # Partial trust on upside potential
            return trend + ((xgb - trend) * UPSIDE_TRUST)
        else:
            return trend

    df['Super_Ensemble_pred'] = df.apply(apply_gate, axis=1)
    df['Predicted_Best_Model'] = np.where(df['Super_Ensemble_pred'] == df['trend_pred'], 'Statistical_Trend_pred',
                                          'Hybrid_XGB_pred')

    df.to_csv(OUTPUT_DIR / 'super_ensemble_final_forecast_TSCV.csv', index=False)

    df_prod = df[['year', 'district_no', 'kreisYield', 'trend_pred', 'Super_Ensemble_pred', 'Predicted_Best_Model']]
    df_prod.to_csv(OUTPUT_DIR / 'super_ensemble_production_forecast.csv', index=False)

    logging.info(f"✓ Saved Full Trust Gated Forecasts to {OUTPUT_DIR}")


if __name__ == '__main__':
    generate_gated_forecast()