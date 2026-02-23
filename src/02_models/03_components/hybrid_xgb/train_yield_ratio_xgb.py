import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import sys
from pathlib import Path
import logging

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib

models_config = importlib.import_module("src.02_models.config")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TRAIN_CONFIG = models_config.STANDALONE_XGB_CONFIG


def train_standalone_model(train_config):
    logging.info("--- Starting Standalone Training (Target: Log Ratio) ---")
    df = pd.read_csv(train_config['DATA_PATH'])

    # CRITICAL FIX: Ensure base district_no is a formatted string before merging
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)

    # Merge Stage 2 features to give it the full arsenal
    stage2_path = global_config.DATA_DIR / '05_model_input/stage2_refined_features.csv'
    if stage2_path.exists():
        df2 = pd.read_csv(stage2_path)
        df2['district_no'] = df2['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, df2, on=['year', 'district_no'], how='left')

    target_col = train_config['TARGET_COL']
    df.dropna(subset=[target_col, 'stage1_forecast'], inplace=True)
    df = df[df['year'] >= (df['year'].min() + 7)].copy()
    df = df[df['stage1_forecast'] > 0.1]

    # Widened clip to allow crashes
    df['yield_ratio'] = np.log(df[target_col] / df['stage1_forecast']).clip(-0.6, 0.6)

    # DYNAMIC FEATURE SELECTION: Ingest everything except strict exclusions
    exclude_cols = [
        'district_no', 'year', 'kreisYield', 'yield', 'stage1_forecast',
        'yield_ratio', 'has_wofost_data', 'state_encoded', 'year_trend'
    ]
    valid_features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]

    X_train, y_train = df[valid_features], df['yield_ratio']
    logging.info(f"Features Unlocked: {len(valid_features)}. Rows: {len(X_train)}")

    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} model...")

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=quantile,
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            reg_lambda=2.0,
            gamma=1.0,
            subsample=0.8,
            colsample_bytree=0.5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        out_path = train_config[f'{name.upper()}_MODEL_PATH']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_path)


if __name__ == "__main__":
    train_standalone_model(TRAIN_CONFIG)