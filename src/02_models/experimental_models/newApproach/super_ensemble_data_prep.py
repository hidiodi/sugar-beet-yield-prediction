import pandas as pd
import logging
from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])
OUTPUT_FILENAME = 'super_ensemble_training_data.csv'

# --- File Paths ---
NATIVE_ENSEMBLE_PATH = project_root / 'src/models/native_ensemble_champion/native_ensemble_forecasts.csv'
STATISTICAL_TREND_PATH = Path(CONFIG['STATISTICAL_TREND_FILE'])
HYBRID_XGB_PATH = Path(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
V31_SOLAR_PATH = config.DATA_DIR / '06_model_output' / 'v31_solar_gated_forecast.csv'
ROBUST_LINEAR_PATH = config.DATA_DIR / '06_model_output' / 'recovery_models' / 'stage2_forecasts.csv'
# NEW: Stage 1 Features for Context
STAGE1_FEATURES_PATH = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']

ANALYSIS_START_YEAR = 2000
ANALYSIS_END_YEAR = 2024
ANCHOR_MODEL = 'Statistical Trend'

COMPONENT_MODELS = {
    'Statistical Trend': STATISTICAL_TREND_PATH,
    'V31 Solar Gated': V31_SOLAR_PATH,
    'Native Ensemble': NATIVE_ENSEMBLE_PATH,
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

    # 1. Establish Backbone with Ground Truth
    # FIX: Use Hybrid XGB because it contains 'kreisYield' (Target)
    if not HYBRID_XGB_PATH.exists():
        logging.error("Hybrid XGB Path not found. Cannot load Ground Truth.")
        return pd.DataFrame()

    df = pd.read_csv(HYBRID_XGB_PATH)[['year', 'district_no', 'kreisYield']]
    df['district_no'] = df['district_no'].astype(int)

    # 2. Merge Components
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
                logging.info(f"✓ Merged {model_name}")

    # 3. Clean & Fill Data
    df_clean = df[(df['year'] >= ANALYSIS_START_YEAR) & (df['year'] <= ANALYSIS_END_YEAR)].copy()

    anchor_col = f'{ANCHOR_MODEL.replace(" ", "_")}_pred'
    pred_cols = [f'{m.replace(" ", "_")}_pred' for m in COMPONENT_MODELS.keys()]

    # Fill missing with Anchor
    if anchor_col in df_clean.columns:
        for col in pred_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[anchor_col])

    df_clean.dropna(subset=[anchor_col, 'kreisYield'], inplace=True)

    # 4. NEW: Merge Context Features (The "Why")
    if STAGE1_FEATURES_PATH.exists():
        logging.info("Merging Context Features (Soil, Water, Region)...")
        df_ctx = pd.read_csv(STAGE1_FEATURES_PATH)
        df_ctx['district_no'] = df_ctx['district_no'].astype(int)

        # Select critical context features
        context_cols = [
            'year', 'district_no',
            'is_gdr',  # Regional Logic
            'avg_clay_0_30cm',  # Soil Logic
            'summer_water_balance_anomaly',  # Drought Logic
            'summer_days_tmax_gt_30c',  # Heat Logic
            'sowing_doy_anomaly'  # Ag Logic
        ]
        # Only keep available columns
        context_cols = [c for c in context_cols if c in df_ctx.columns]

        df_clean = pd.merge(df_clean, df_ctx[context_cols], on=['year', 'district_no'], how='left')

        # Fill context NaNs with 0 to prevent dropping
        for c in context_cols:
            if c not in ['year', 'district_no']:
                df_clean[c] = df_clean[c].fillna(0)

    logging.info(f"Data ready: {len(df_clean)} records.")
    return df_clean


def create_classification_features(df):
    logging.info("\n--- Creating Classification Targets & Features ---")
    df_new = df.copy()
    anchor_col = f'{ANCHOR_MODEL.replace(" ", "_")}_pred'

    # 1. Target: Best Model
    error_cols = []
    for model in COMPONENT_MODELS.keys():
        col_name = f'{model.replace(" ", "_")}_pred'
        if col_name not in df_new.columns: continue
        err_col = f'Error_{model.replace(" ", "_")}'
        df_new[err_col] = (df_new['kreisYield'] - df_new[col_name]).abs()
        error_cols.append(err_col)

    df_new['Best_Model'] = df_new[error_cols].idxmin(axis=1).apply(lambda x: x.replace('Error_', ''))

    # Oracle Stats
    df_new['Oracle_Error'] = df_new[error_cols].min(axis=1)

    # 2. Signals
    signal_cols = []
    for model in COMPONENT_MODELS.keys():
        if model == ANCHOR_MODEL: continue
        col_name = f'{model.replace(" ", "_")}_pred'
        if col_name not in df_new.columns: continue
        signal_name = f'Signal_{model.replace(" ", "_")}'
        df_new[signal_name] = df_new[col_name] - df_new[anchor_col]
        signal_cols.append(signal_name)

    # 3. Save
    exclude = error_cols
    final_cols = [c for c in df_new.columns if c not in exclude]

    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    df_new[final_cols].to_csv(output_path, index=False)
    logging.info(f"--- Training Data Saved: {output_path} ---")
    logging.info(f"Context Features Included: {[c for c in df_new.columns if c in ['is_gdr', 'avg_clay_0_30cm']]}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_merge_components()
    if not df.empty:
        create_classification_features(df)


if __name__ == '__main__':
    main()