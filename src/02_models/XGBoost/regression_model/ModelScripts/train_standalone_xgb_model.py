# File: src/02_models/XGBoost/regression_model/ModelScripts/train_standalone_xgb_model.py
# REFACTORED (v10.0): Standalone No-Lag / Residuals
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
    logging.info("--- Starting Standalone Training (No Lag / Residuals) ---")

    try:
        df = pd.read_csv(train_config['DATA_PATH'])
    except FileNotFoundError: sys.exit(1)

    baseline_col = 'stat_trend_forecast'
    target_col = 'trend_residual'
    df[target_col] = df['kreisYield'] - df[baseline_col]

    # Feature List (Must match Hybrid exactly)
    feature_cols = [
        'trend_vs_phys_gap',
        'wofost_skew', 'wofost_esp_std', 'wofost_esp_p10', 'wofost_water_stress_mean',

        'nitrogen_leaching_index', 'toxic_carryover_index', 'vector_pressure_local', 'winter_pest_kill_days',
        'antecedent_precip_sum', 'sowing_potential_days', 'winter_cropland_ndvi_anomaly',
        'avg_clay_0_30cm', 'avg_sand_0_30cm',
        'fertilizer_price_index_lag1', 'nao_winter_avg',
        'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast',
        'summer_precip_prob_wet_forecast', 'summer_temp_prob_warm_forecast',
        'spring_temp_x_antecedent_rain'
    ]

    valid_features = [c for c in feature_cols if c in df.columns]
    df.dropna(subset=[target_col, baseline_col] + valid_features, inplace=True)
    X_train = df[valid_features]
    y_train = df[target_col]

    xgb_params = {
        'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.015,
        'subsample': 0.75, 'colsample_bytree': 0.7, 'min_child_weight': 10, 'gamma': 2.0,
        'n_jobs': -1, 'random_state': 42
    }

    for name, quantile in train_config['QUANTILES'].items():
        logging.info(f"Training {name.upper()}...")
        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile, **xgb_params)
        model.fit(X_train, y_train)
        joblib.dump(model, train_config[f'{name.upper()}_MODEL_PATH'])

    logging.info("--- Standalone Training Complete ---")

if __name__ == "__main__":
    train_standalone_model(TRAIN_CONFIG)