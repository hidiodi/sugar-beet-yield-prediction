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
    logging.info("--- Starting Champion Training (Target: Raw Yield) ---")
    # REVERTED: Hybrid works best on Raw Yield because the baseline is noisy.

    try:
        df = pd.read_csv(train_config['DATA_PATH'])
    except Exception as e:
        logging.error(f"Failed to load data: {e}");
        sys.exit(1)

    target_col = 'kreisYield'
    df.dropna(subset=[target_col], inplace=True)

    # --- 1. FEATURE SELECTION ---
    feature_cols = train_config['FEATURE_COLS']

    # Ensure year/trend are PRESENT for Raw Yield model
    # (They might have been removed by the Ratio logic previously)
    valid_features = [c for c in feature_cols if c in df.columns]

    # We must ensure 'year' or 'year_trend' is available if we predict raw yield
    if 'year' not in valid_features and 'year' in df.columns:
        valid_features.append('year')

    X_train = df[valid_features]
    y_train = df[target_col]  # Raw Yield

    logging.info(f"Training on {len(X_train)} samples. Target Mean: {y_train.mean():.2f}")

    # --- 2. Training Loop ---
    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} Model...")

        params = train_config[f'BEST_PARAMS_{name.upper()}']

        # Load constraints
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