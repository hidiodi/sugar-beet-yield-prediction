# File: src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py
# REFACTORED (v5.0): "The Risk & Cone Model" Training
# Description:
#   Trains the residual model to predict deviations from the 5-year rolling trend.
#   Uses strict regularization to prevent overfitting to weather noise.
#   Focuses on "Gap" features (Physical vs Statistical) and Risk (std/p10).

import pandas as pd
from xgboost import XGBRegressor
import joblib
import sys
from pathlib import Path
import logging

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TRAIN_CONFIG = config.XGBOOST_TRAINING_CONFIG


def train_and_save_models(train_config):
    """
    Trains 3 quantile models (Lower, Median, Upper) on the Residuals.
    """
    logging.info("--- Starting ROBUST Quantile Model Training (Risk-Based) ---")

    # 1. Load Data
    data_path = train_config['DATA_PATH']
    logging.info(f"Loading data from {data_path}...")
    if not data_path.exists():
        logging.error(f"FATAL: Data not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # 2. Define Target: The Residual from a simple Rolling Trend
    # We predict: Actual Yield - 5yr Rolling Avg
    # The model explains *why* this year deviates from the recent norm.
    baseline_feature = 'yield_rolling_trend'
    target_col = 'trend_residual'

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # Calculate 5-year trailing average (shift 1 to avoid leakage)
    df[baseline_feature] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=3).mean().shift(1)
    )

    df[target_col] = df['kreisYield'] - df[baseline_feature]

    # Drop rows where we can't calculate the target or the anchor
    df.dropna(subset=[baseline_feature, target_col, 'stat_trend_forecast'], inplace=True)

    # 3. Define Features (The "Knowns" in March)
    # We interpret 'FEATURE_COLS' from config as the definitive list.
    # But we hardcode the critical ones here to ensure they are included.

    # CRITICAL: 'stat_trend_forecast' is NOT a leak here. It is a pre-calculated baseline
    # available in March. It is the "Anchor".

    robust_feature_cols = [
        # The Anchors
        'stat_trend_forecast',  # The complex statistical expectation
        'national_avg_yield_lag1',  # Macro trend

        # The Physics (The Cone of Uncertainty)
        'trend_vs_phys_gap',  # SIGNAL: WOFOST Mean - Stat Trend
        'wofost_esp_std',  # RISK: How volatile is the biology?
        'wofost_esp_p10',  # DOWNSIDE: What's the worst case?
        'wofost_skew',  # ASYMMETRY: Is upside or downside more likely?

        # The Antecedent State (Observed)
        'antecedent_precip_sum',  # Soil recharge
        'antecedent_gdd_sum_anomaly',  # Early heat
        'winter_cropland_ndvi_anomaly',  # Crop state entering spring

        # The Economics (Incentives)
        'fertilizer_price_index_lag1',
        'producer_price_index_lag1',

        # Static
        'avg_clay_0_30cm',
        'avg_sand_0_30cm',
        'avg_elevation'
    ]

    # Intersect with available columns
    valid_features = [c for c in robust_feature_cols if c in df.columns]

    missing = set(robust_feature_cols) - set(valid_features)
    if missing:
        logging.warning(f"The following robust features are missing from input: {missing}")

    X_train = df[valid_features]
    y_train = df[target_col]

    logging.info(f"Training on {len(X_train)} samples with {len(valid_features)} features.")
    logging.info(f"Features: {valid_features}")

    # 4. Model Hyperparameters (Regularized)
    # We use shallower trees and higher gamma to force the model to find
    # universal risks rather than memorizing specific year weather patterns.

    xgb_params = {
        'n_estimators': 800,
        'max_depth': 4,  # Shallow trees = Less overfitting
        'learning_rate': 0.02,  # Slow learning
        'subsample': 0.75,
        'colsample_bytree': 0.8,
        'min_child_weight': 15,  # Needs many samples to make a split decision
        'gamma': 2.0,  # High regularization threshold
        'n_jobs': -1,
        'random_state': 42
    }

    quantiles = train_config['QUANTILES']
    model_configs = {
        'lower': {'alpha': quantiles['lower'], 'path': train_config['LOWER_MODEL_PATH']},
        'median': {'alpha': quantiles['median'], 'path': train_config['MEDIAN_MODEL_PATH']},
        'upper': {'alpha': quantiles['upper'], 'path': train_config['UPPER_MODEL_PATH']}
    }

    # 5. Train and Save
    for name, cfg in model_configs.items():
        logging.info(f"Training {name.upper()} model (alpha={cfg['alpha']})...")

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=cfg['alpha'],
            **xgb_params
        )

        model.fit(X_train, y_train)

        # Save
        output_path = cfg['path']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        logging.info(f"✓ {name.upper()} model saved.")

    logging.info("--- Training Complete. Models Ready. ---")


if __name__ == "__main__":
    train_and_save_models(TRAIN_CONFIG)