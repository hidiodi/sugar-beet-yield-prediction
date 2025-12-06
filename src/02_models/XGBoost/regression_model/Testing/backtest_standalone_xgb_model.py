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
TRAIN_CONFIG = config.STANDALONE_XGB_CONFIG


def train_standalone_model(train_config):
    logging.info("--- Starting Standalone Training (Target: Yield Ratio) ---")

    try:
        df = pd.read_csv(train_config['DATA_PATH'])
    except FileNotFoundError:
        logging.error("Data file not found.")
        sys.exit(1)

    target_col = train_config['TARGET_COL']
    df.dropna(subset=[target_col, 'stage1_forecast'], inplace=True)

    # --- STRUCTURAL PIVOT: PREDICT RATIO ---
    df['yield_ratio'] = df[target_col] / df['stage1_forecast']
    df['yield_ratio'] = df['yield_ratio'].clip(0.5, 1.5)

    feature_cols = train_config['FEATURE_COLS']
    # Remove year features from Standalone too (Physics Focus)
    valid_features = [c for c in feature_cols if c in df.columns and c not in ['year', 'year_trend']]

    X_train = df[valid_features]
    y_train = df['yield_ratio']  # Target is Ratio

    logging.info(f"Target: Yield Ratio. Features: {len(valid_features)}")

    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} model...")
        params = train_config[f'BEST_PARAMS_{name.upper()}']

        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile, **params)
        model.fit(X_train, y_train)

        out_path = train_config[f'{name.upper()}_MODEL_PATH']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_path)

    logging.info("--- Standalone Training Complete ---")


if __name__ == "__main__":
    train_standalone_model(TRAIN_CONFIG)