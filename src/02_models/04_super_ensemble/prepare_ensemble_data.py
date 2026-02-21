import pandas as pd
import logging
from pathlib import Path
import sys
import numpy as np
import geopandas as gpd

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib

analysis_config = importlib.import_module("src.03_analysis.config")
models_config = importlib.import_module("src.02_models.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = analysis_config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])
OUTPUT_FILENAME = 'super_ensemble_training_data.csv'

# --- Paths ---
STATISTICAL_TREND_PATH = Path(CONFIG['STATISTICAL_TREND_FILE'])
HYBRID_XGB_PATH = Path(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
V31_SOLAR_PATH = global_config.DATA_DIR / '06_model_output' / 'v31_solar_gated_forecast.csv'
ROBUST_LINEAR_PATH = global_config.DATA_DIR / '06_model_output' / 'recovery_models' / 'stage2_forecasts.csv'
STAGE1_FEATURES_PATH = models_config.XGBOOST_TRAINING_CONFIG['DATA_PATH']

ANALYSIS_START_YEAR = 2000
ANALYSIS_END_YEAR = 2024
ANCHOR_MODEL = 'Statistical Trend'

COMPONENT_MODELS = {
    'Statistical Trend': STATISTICAL_TREND_PATH,
    'V31 Solar Gated': V31_SOLAR_PATH,
    'Hybrid XGB': HYBRID_XGB_PATH,
    'Robust Linear': ROBUST_LINEAR_PATH
}

MODEL_COLUMN_MAP = {
    'Statistical Trend': 'final_corrected_forecast',
    'V31 Solar Gated': 'final_pred',
    'Native Ensemble': 'Ensemble_Pred',
    'Hybrid XGB': 'predicted_yield_median',
    'Robust Linear': 'stage2_pred'
}


def load_and_merge_components():
    logging.info("--- Loading and Merging Super-Ensemble Components ---")

    if not HYBRID_XGB_PATH.exists(): return pd.DataFrame()
    df = pd.read_csv(HYBRID_XGB_PATH)[['year', 'district_no', 'kreisYield']]
    df['district_no'] = df['district_no'].astype(int)

    for model_name, path in COMPONENT_MODELS.items():
        p = Path(path)
        if p.exists():
            df_m = pd.read_csv(p)
            if 'district_no' in df_m.columns:
                df_m['district_no'] = df_m['district_no'].astype(int)
            pred_col = MODEL_COLUMN_MAP[model_name]
            if pred_col in df_m.columns:
                target_col = f'{model_name.replace(" ", "_")}_pred'
                df_m = df_m[['year', 'district_no', pred_col]].rename(columns={pred_col: target_col})
                df = pd.merge(df, df_m, on=['year', 'district_no'], how='left')

    df_clean = df[(df['year'] >= ANALYSIS_START_YEAR) & (df['year'] <= ANALYSIS_END_YEAR)].copy()
    anchor_col = f'{ANCHOR_MODEL.replace(" ", "_")}_pred'

    if anchor_col in df_clean.columns:
        for col in [c for c in df_clean.columns if c.endswith('_pred')]:
            df_clean[col] = df_clean[col].fillna(df_clean[anchor_col])

    df_clean.dropna(subset=[anchor_col, 'kreisYield'], inplace=True)

    # --- MERGE CONTEXT (TELECONNECTIONS) ---
    if STAGE1_FEATURES_PATH.exists():
        logging.info("Merging Context Features (Teleconnections)...")
        df_ctx = pd.read_csv(STAGE1_FEATURES_PATH)
        df_ctx['district_no'] = df_ctx['district_no'].astype(int)

        context_cols = [
            'year', 'district_no',
            'enso_mei_winter_avg', 'nao_winter_avg', 'sca_winter_avg',
            'effective_winter_water', 'trafficability_stress'
        ]
        context_cols = [c for c in context_cols if c in df_ctx.columns]

        df_clean = pd.merge(df_clean, df_ctx[context_cols], on=['year', 'district_no'], how='left')
        for c in context_cols:
            if c not in ['year', 'district_no']: df_clean[c] = df_clean[c].fillna(0)

    # --- CREATE "BEST MODEL" & ORACLE TARGETS ---
    model_cols = [f'{m.replace(" ", "_")}_pred' for m in COMPONENT_MODELS.keys()]
    model_cols = [c for c in model_cols if c in df_clean.columns]

    # Calculate absolute error for each model
    for col in model_cols:
        err_col = f'Error_{col}'
        df_clean[err_col] = (df_clean['kreisYield'] - df_clean[col]).abs()

    err_cols = [f'Error_{c}' for c in model_cols]
    df_clean['Best_Model'] = df_clean[err_cols].idxmin(axis=1).apply(lambda x: x.replace('Error_', ''))

    # RESTORED FORENSIC COLUMNS:
    df_clean['Oracle_Error'] = df_clean[err_cols].min(axis=1)
    df_clean['Median_Error'] = df_clean[err_cols].median(axis=1)
    df_clean['Regret_Weight'] = (df_clean['Median_Error'] - df_clean['Oracle_Error']).clip(lower=1.0)

    return df_clean


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_merge_components()
    if not df.empty:
        output_path = OUTPUT_DIR / OUTPUT_FILENAME
        df.to_csv(output_path, index=False)
        logging.info(f"--- Training Data Saved: {output_path} ---")
        logging.info(f"Generated 'Best_Model' Classification Target and Oracle Forensics.")


if __name__ == '__main__':
    main()