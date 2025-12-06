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
    logging.info("--- Starting Champion Training (Target: Yield Residual) ---")

    try:
        df = pd.read_csv(train_config['DATA_PATH'])
    except Exception as e:
        logging.error(f"Failed to load data: {e}");
        sys.exit(1)

    # --- 1. PREPROCESSING: CALCULATE RESIDUALS ---
    # We want to predict how much the weather pushes yield ABOVE or BELOW the trend.
    target_col = 'yield_residual'

    # We need both the actual yield and the trend to calculate the target
    df.dropna(subset=['kreisYield', 'stage1_forecast'], inplace=True)

    # Target = Actual - Trend
    # Example: Yield 800, Trend 750 -> Target +50
    df[target_col] = df['kreisYield'] - df['stage1_forecast']

    # --- 2. FEATURE SELECTION ---
    feature_cols = train_config['FEATURE_COLS']
    valid_features = [c for c in feature_cols if c in df.columns]

    # CRITICAL: Drop Trend and Year from X to force Physics Learning
    # If we leave 'year' in, the model just learns "later years are better", which is the trend's job.
    cols_to_drop = ['stage1_forecast', 'year', 'year_trend', 'kreisYield', 'yield_residual']
    valid_features = [c for c in valid_features if c not in cols_to_drop]

    X_train = df[valid_features]
    y_train = df[target_col]

    logging.info(f"Training on {len(X_train)} samples. Target Mean (Residual): {y_train.mean():.2f}")
    logging.info(f"Physics Features Used: {len(valid_features)}")

    # --- 3. Training Loop ---
    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()} Model...")

        params = train_config[f'BEST_PARAMS_{name.upper()}']

        # Load constraints
        config_constraints = train_config.get('MONOTONE_CONSTRAINTS', {})
        monotone_constraints = []

        # Only apply constraints to features that actually exist in our filtered X_train
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