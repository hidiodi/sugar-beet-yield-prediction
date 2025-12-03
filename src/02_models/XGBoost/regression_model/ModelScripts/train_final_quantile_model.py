# File: src/02_models/XGBoost/regression_model/ModelScripts/train_final_quantile_model.py
# REFACTORED (v10.0): Anomaly Hunter (Residuals + No Lag)
# Strategy:
#   - Target: Residual (Actual - Trend).
#   - Features: PURELY Physical/Weather/Economic. NO LAGS allowed.
#   - Goal: Force model to explain "Why is this year different?" using only causality.

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
    logging.info("--- Starting Hybrid Training (Anomaly Hunter Profile) ---")

    data_path = train_config['DATA_PATH']
    if not data_path.exists(): sys.exit(1)
    df = pd.read_csv(data_path)

    # 1. Target: Residual
    baseline_col = 'stat_trend_forecast'
    target_col = 'trend_residual'
    df[target_col] = df['kreisYield'] - df[baseline_col]
    df.dropna(subset=[baseline_col, target_col, 'trend_vs_phys_gap'], inplace=True)

    # 2. Features: STRICTLY CAUSAL (No Inertia/Lag)
    feature_cols = [
        # --- The Physics Signal (Gap between Biology and Statistics) ---
        'trend_vs_phys_gap',
        'wofost_skew',
        'wofost_esp_std',

        # --- Explicit Risks (Relative to Trend) ---
        # "Is the WOFOST worst-case scenario significantly below the Trend?"
        'wofost_esp_p10',
        'wofost_water_stress_mean',

        # --- Mechanisms ---
        'nitrogen_leaching_index',
        'toxic_carryover_index',
        'vector_pressure_local',
        'winter_pest_kill_days',

        # --- Antecedent Context ---
        'antecedent_precip_sum',
        'sowing_potential_days',
        'winter_cropland_ndvi_anomaly',

        # --- Static Context ---
        'avg_clay_0_30cm',
        'avg_sand_0_30cm',

        # --- Economics (Price Signals) ---
        'fertilizer_price_index_lag1',  # High Price -> Less N -> Lower Yield

        # --- Weather Patterns ---
        'nao_winter_avg',

        # --- Forecast Probabilities (Risk Detectors) ---
        'spring_temp_prob_warm_forecast',
        'spring_precip_prob_wet_forecast',
        'summer_precip_prob_wet_forecast',
        'summer_temp_prob_warm_forecast',

        # --- Interactions ---
        'spring_temp_x_antecedent_rain'
    ]

    valid_features = [c for c in feature_cols if c in df.columns]
    X_train = df[valid_features]
    y_train = df[target_col]

    logging.info(f"Training on {len(valid_features)} features (NO LAGS).")

    # 3. Hyperparameters (Balanced for Signal Detection)
    xgb_params = {
        'n_estimators': 1000,
        'max_depth': 5,
        'learning_rate': 0.015,
        'subsample': 0.75,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'gamma': 2.0,  # Filter noise, but allow strong physical signals
        'n_jobs': -1,
        'random_state': 42
    }

    for name, cfg in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()}...")
        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=cfg, **xgb_params)
        model.fit(X_train, y_train)

        out_path = config.XGBOOST_TRAINING_CONFIG[f'{name.upper()}_MODEL_PATH']
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_path)

    logging.info("--- Training Complete ---")


if __name__ == "__main__":
    train_and_save_models(TRAIN_CONFIG)