import pandas as pd
from xgboost import XGBRegressor
import joblib
import sys
from pathlib import Path
import logging

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TRAIN_CONFIG = config.XGBOOST_TRAINING_CONFIG


def train_and_save_models(train_config):
    logging.info("--- Starting Champion Training (Target: Yield Ratio) ---")

    try:
        df = pd.read_csv(train_config['DATA_PATH'])
    except Exception as e:
        logging.error(f"Failed to load data: {e}");
        sys.exit(1)

    target_col = 'kreisYield'
    df.dropna(subset=[target_col, 'stage1_forecast'], inplace=True)

    # --- STRUCTURAL PIVOT: PREDICT YIELD RATIO ---
    # Target = Actual / Trend
    # Using Ratio allows the 0-1 V14 Indices to work effectively
    df['yield_ratio'] = df['kreisYield'] / df['stage1_forecast']

    # Cap extreme outliers
    df['yield_ratio'] = df['yield_ratio'].clip(0.5, 1.5)

    y_train = df['yield_ratio']

    # --- 1. FEATURE SELECTION ---
    feature_cols = train_config['FEATURE_COLS']

    # Remove year features (Ratio is detrended)
    for col in ['year', 'year_trend']:
        if col in feature_cols:
            feature_cols.remove(col)

    valid_features = [c for c in feature_cols if c in df.columns]
    X_train = df[valid_features]

    logging.info(f"Training on {len(X_train)} samples. Target: Yield Ratio")

    # --- 2. Training Loop ---
    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} Model...")

        params = train_config[f'BEST_PARAMS_{name.upper()}']

        config_constraints = train_config.get('MONOTONE_CONSTRAINTS', {})
        monotone_constraints = []
        for feature in valid_features:
            c = config_constraints.get(feature, 0)
            monotone_constraints.append(c)

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=quantile,
            monotone_constraints=tuple(monotone_constraints),
            **params
        )

        model.fit(X_train, y_train)

        out_path = train_config[f'{name.upper()}_MODEL_PATH']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_path)

    logging.info("--- Champion Training Complete ---")


if __name__ == "__main__":
    train_and_save_models(TRAIN_CONFIG)