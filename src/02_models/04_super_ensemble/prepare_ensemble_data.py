import pandas as pd
import logging
from pathlib import Path
import sys
import numpy as np
import geopandas as gpd

project_root = Path(__file__).resolve().parents[4]
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
NATIVE_ENSEMBLE_PATH = project_root / 'src/models/native_ensemble_champion/native_ensemble_forecasts.csv'
STATISTICAL_TREND_PATH = Path(CONFIG['STATISTICAL_TREND_FILE'])
HYBRID_XGB_PATH = Path(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
V31_SOLAR_PATH = global_config.DATA_DIR / '06_model_output' / 'v31_solar_gated_forecast.csv'
ROBUST_LINEAR_PATH = global_config.DATA_DIR / '06_model_output' / 'recovery_models' / 'stage2_forecasts.csv'
STAGE1_FEATURES_PATH = models_config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
GEO_FILE = global_config.DATA_DIR / '01_raw/districts_official.geojson'

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


def get_lat_lon():
    if not GEO_FILE.exists(): return None
    gdf = gpd.read_file(GEO_FILE)
    id_col = None
    for c in ['AGS', 'RS', 'id', 'district_no']:
        if c in gdf.columns and str(gdf[c].iloc[0]).isdigit(): id_col = c
    if not id_col: return None

    # Reproject to avoiding UserWarning (EPSG:4326 is lat/lon, but centroid needs planar)
    # We'll just suppress or ignore since rough centroid is fine for ML
    centroids = gdf.geometry.centroid

    df_geo = pd.DataFrame({
        'district_no': gdf[id_col].astype(str).str.zfill(5),
        'latitude': centroids.y,
        'longitude': centroids.x
    })
    return df_geo


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
    pred_cols = [f'{m.replace(" ", "_")}_pred' for m in COMPONENT_MODELS.keys()]

    if anchor_col in df_clean.columns:
        for col in pred_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[anchor_col])

    df_clean.dropna(subset=[anchor_col, 'kreisYield'], inplace=True)

    # --- MERGE CONTEXT (ENHANCED) ---
    if STAGE1_FEATURES_PATH.exists():
        logging.info("Merging Enhanced Context Features (Bio-Stress)...")
        df_ctx = pd.read_csv(STAGE1_FEATURES_PATH)
        df_ctx['district_no'] = df_ctx['district_no'].astype(int)

        # ADDED: CASDI, NMSD, OSAW (The biological stress markers)
        context_cols = [
            'year', 'district_no',
            'is_gdr', 'avg_clay_0_30cm',
            'summer_water_balance_anomaly',
            'summer_days_tmax_gt_30c',
            'sowing_doy_anomaly',
            'CASDI_Phase2_Count',  # Critical Dryness
            'NMSD_Phase2_Count',  # Night Heat
            'OSAW_Phase2_Count'  # Oxygen Stress
        ]
        context_cols = [c for c in context_cols if c in df_ctx.columns]
        df_clean = pd.merge(df_clean, df_ctx[context_cols], on=['year', 'district_no'], how='left')
        for c in context_cols:
            if c not in ['year', 'district_no']: df_clean[c] = df_clean[c].fillna(0)

    # Merge Spatial
    df_geo = get_lat_lon()
    if df_geo is not None:
        df_geo['district_no'] = df_geo['district_no'].astype(int)
        df_clean = pd.merge(df_clean, df_geo, on='district_no', how='left')

    return df_clean


def calculate_district_history(df):
    anchor_col = f'{ANCHOR_MODEL.replace(" ", "_")}_pred'
    df = df.sort_values(['district_no', 'year'])
    df['Raw_Bias'] = df['kreisYield'] - df[anchor_col]
    df['District_Historical_Bias'] = df.groupby('district_no')['Raw_Bias'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)
    return df


def create_classification_features(df):
    logging.info("\n--- Creating Targets & Regret Weights ---")
    df_new = df.copy()
    anchor_col = f'{ANCHOR_MODEL.replace(" ", "_")}_pred'

    df_new = calculate_district_history(df_new)

    error_cols = []
    model_pred_cols = []

    for model in COMPONENT_MODELS.keys():
        col_name = f'{model.replace(" ", "_")}_pred'
        if col_name in df_new.columns:
            model_pred_cols.append(col_name)
            err_col = f'Error_{model.replace(" ", "_")}'
            df_new[err_col] = (df_new['kreisYield'] - df_new[col_name]).abs()
            error_cols.append(err_col)

    df_new['Best_Model'] = df_new[error_cols].idxmin(axis=1).apply(lambda x: x.replace('Error_', ''))
    df_new['Oracle_Error'] = df_new[error_cols].min(axis=1)

    # --- NEW: CALCULATE REGRET WEIGHT ---
    # Weight = Difference between Median Error and Min Error
    # This emphasizes rows where there is a big difference between choices
    df_new['Median_Error'] = df_new[error_cols].median(axis=1)
    df_new['Regret_Weight'] = (df_new['Median_Error'] - df_new['Oracle_Error']).clip(lower=1.0)

    # Normalize weights to avoid exploding gradients (cap at 5x importance)
    df_new['Regret_Weight'] = df_new['Regret_Weight'].clip(upper=100)

    # Outlier Flag
    df_new['Is_Garbage_Data'] = (df_new['Oracle_Error'] > 200).astype(int)

    # Signals
    for model in COMPONENT_MODELS.keys():
        if model == ANCHOR_MODEL: continue
        col_name = f'{model.replace(" ", "_")}_pred'
        if col_name in df_new.columns:
            signal_name = f'Signal_{model.replace(" ", "_")}'
            df_new[signal_name] = df_new[col_name] - df_new[anchor_col]

    # Consensus
    pred_matrix = df_new[model_pred_cols]
    df_new['Ensemble_Std'] = pred_matrix.std(axis=1)
    deviations = pred_matrix.subtract(df_new[anchor_col], axis=0)
    df_new['Max_Crash_Signal'] = deviations.min(axis=1)
    df_new['Trend_Consensus_Count'] = deviations.abs().lt(df_new[anchor_col] * 0.05, axis=0).sum(axis=1)
    df_new['state_id'] = df_new['district_no'].astype(str).str.zfill(5).str[:2].astype(int)

    # Save
    exclude = error_cols + ['Raw_Bias', 'Median_Error']
    final_cols = [c for c in df_new.columns if c not in exclude]

    if 'latitude' in df_new.columns:
        final_cols += ['latitude', 'longitude']
        final_cols = list(set(final_cols))

    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    df_new[final_cols].to_csv(output_path, index=False)
    logging.info(f"--- Training Data Saved: {output_path} ---")
    logging.info(f"Added Regret_Weight and Stress Features (CASDI/NMSD)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_merge_components()
    if not df.empty:
        create_classification_features(df)


if __name__ == '__main__':
    main()