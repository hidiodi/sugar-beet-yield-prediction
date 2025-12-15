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
    df['yield_ratio'] = df['kreisYield'] / df['stage1_forecast']
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

    # --- 2. HARDCODED PHYSICS CONSTRAINTS (The Fix) ---
    # This forces the model to respect reality, restoring the 0.54 R2
    manual_constraints = {
        'Index_Failure': -1,  # Failure Index MUST lower yield
        'Index_Bumper': 1,  # Bumper Index MUST raise yield
        'z_heat': -1,  # Heat MUST lower yield
        'z_bal': 1,  # Water Balance MUST raise yield
        'trend_x_failure': -1,  # Trend * Failure MUST lower yield
        'trend_x_bumper': 1,  # Trend * Bumper MUST raise yield
        'overwinter_stress': -1,  # If you have this feature
        'summer_heat_x_water_balance': 1  # Positive interaction
    }

    monotone_constraints = []
    for feature in valid_features:
        # Default to 0 (no constraint) if not in our manual list
        c = manual_constraints.get(feature, 0)
        monotone_constraints.append(c)

    monotone_constraints = tuple(monotone_constraints)
    logging.info(f"Applied Monotone Constraints to {sum(1 for x in monotone_constraints if x != 0)} features.")

    # --- 3. Training Loop ---
    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} Model...")

        params = train_config[f'BEST_PARAMS_{name.upper()}']

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=quantile,
            monotone_constraints=monotone_constraints,  # INJECTED HERE
            **params
        )

        model.fit(X_train, y_train)

        out_path = train_config[f'{name.upper()}_MODEL_PATH']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_path)

    logging.info("--- Champion Training Complete ---")


if __name__ == "__main__":
    train_and_save_models(TRAIN_CONFIG)