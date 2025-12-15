import pandas as pd
from xgboost import XGBRegressor
import joblib
import sys
from pathlib import Path
import logging

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib
models_config = importlib.import_module("src.02_models.config")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TRAIN_CONFIG = models_config.STANDALONE_XGB_CONFIG

# NEW: Define a Burn-In threshold to ignore statistically unstable early years
BURN_IN_YEARS = 7


def train_standalone_model(train_config):
    logging.info("--- Starting Standalone Training (Target: Yield Ratio) ---")

    try:
        df = pd.read_csv(train_config['DATA_PATH'])
    except FileNotFoundError:
        logging.error("Data file not found.")
        sys.exit(1)

    target_col = train_config['TARGET_COL']

    # 1. CLEANING: Ensure we have valid targets and inputs
    df.dropna(subset=[target_col, 'stage1_forecast'], inplace=True)

    # 2. IMPROVEMENT: Apply Burn-In Filter
    # With 'expanding' statistics, the first few years have high variance/low reliability.
    # We drop them to prevent the model from learning noise.
    start_year = df['year'].min()
    valid_start_year = start_year + BURN_IN_YEARS

    logging.info(f"Applying Burn-In: Dropping years < {valid_start_year} (Unstable Stats)")
    df = df[df['year'] >= valid_start_year].copy()

    # --- STRUCTURAL PIVOT: PREDICT RATIO ---
    # 3. IMPROVEMENT: Robust Ratio Calculation
    # Prevent division by zero or extreme outliers from 'honest' (noisy) forecasts
    df = df[df['stage1_forecast'] > 0.1]  # Safety check

    df['yield_ratio'] = df[target_col] / df['stage1_forecast']

    # Clip allows the model to focus on realistic deviations, not data errors
    df['yield_ratio'] = df['yield_ratio'].clip(0.5, 1.5)

    feature_cols = train_config['FEATURE_COLS']
    # Remove year features from Standalone too (Physics Focus)
    valid_features = [c for c in feature_cols if c in df.columns and c not in ['year', 'year_trend']]

    X_train = df[valid_features]
    y_train = df['yield_ratio']  # Target is Ratio

    logging.info(f"Target: Yield Ratio. Features: {len(valid_features)}. Rows: {len(X_train)}")

    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} model...")
        params = train_config[f'BEST_PARAMS_{name.upper()}']

        # 4. RECOMMENDATION: Increase Regularization if not already tuned
        # 'Honest' data is noisier. If you haven't re-tuned, consider forcing higher gamma here manually
        # params['gamma'] = max(params.get('gamma', 0), 1.0)

        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile, **params)
        model.fit(X_train, y_train)

        out_path = train_config[f'{name.upper()}_MODEL_PATH']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_path)

    logging.info("--- Standalone Training Complete ---")


if __name__ == "__main__":
    train_standalone_model(TRAIN_CONFIG)