# File: src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py
# REFACTORED (v22.0): Unshackled Model
# - REMOVED conflicting Monotone Constraints (Let data speak).
# - REMOVED 'year_trend' (Forces model to use Weather + Stage1).

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
    logging.info("--- Starting Champion Training (Unshackled) ---")

    try:
        df = pd.read_csv(train_config['DATA_PATH'])
    except Exception as e:
        logging.error(f"Failed to load data: {e}");
        sys.exit(1)

    target_col = 'kreisYield'
    df.dropna(subset=[target_col], inplace=True)

    # --- 1. FEATURE SELECTION (Pruned) ---
    feature_cols = train_config['FEATURE_COLS']

    # CRITICAL FIX: Remove 'year_trend' if present.
    # It allows the model to ignore weather and just count years.
    if 'year_trend' in feature_cols:
        feature_cols.remove('year_trend')

    valid_features = [c for c in feature_cols if c in df.columns]

    X_train = df[valid_features]
    y_train = df[target_col]

    logging.info(f"Training on {len(X_train)} samples using {len(valid_features)} features.")

    # --- 2. Training Loop ---
    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} Model...")

        params = train_config[f'BEST_PARAMS_{name.upper()}']

        # --- REVISED CONSTRAINTS ---
        # Only constrain features that are STRICTLY linear physically.
        # Remove constraints on complex bio-signals (Anoxia, Heat) that might have non-linear responses.

        safe_constraints = {
            'wofost_yield_water_limited': 1,  # Always positive correlation
            'stage1_forecast': 1,  # Always positive correlation
            'effective_winter_water': 1,  # More water tank = Good
        }

        monotone_constraints = []
        active_c = 0
        for feature in valid_features:
            c = safe_constraints.get(feature, 0)
            monotone_constraints.append(c)
            if c != 0: active_c += 1

        logging.info(f" -> Active Constraints: {active_c} (Restricted to safe features)")

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