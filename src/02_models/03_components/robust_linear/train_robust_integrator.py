import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging
import sys
from tqdm import tqdm
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib

models_config = importlib.import_module("src.02_models.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
OUTPUT_DIR = global_config.DATA_DIR / '06_model_output' / 'recovery_models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE1_PATH = models_config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
MIN_HISTORY_YEARS = 10


def main():
    logging.info("--- Training Model C: Robust Ridge Regressor (OOS) ---")
    if not STAGE1_PATH.exists(): return

    df = pd.read_csv(STAGE1_PATH)
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)

    df.dropna(subset=['kreisYield', 'stage1_forecast'], inplace=True)
    df = df[df['stage1_forecast'] > 0.1]

    # Bounded Log Ratio Target
    df['log_ratio'] = np.log(df['kreisYield'] / df['stage1_forecast']).clip(-0.5, 0.5)

    # STRICT FEATURE SELECTION: Macro + New WOFOST Variables
    stable_features = [
        'enso_mei_winter_avg', 'nao_winter_avg', 'sca_winter_avg',
        'winter_precip_sum', 'feb_frost_days', 'effective_winter_water',
        'wofost_yield_water_limited', 'prob_sowing_failure', 'anoxia_events',  # <-- The new WOFOST physical priors
        'latitude', 'longitude', 'avg_elevation', 'avg_slope',
        'avg_sand_0_100cm', 'avg_clay_0_100cm', 'avg_bdod_0_100cm'
    ]
    features = [c for c in stable_features if c in df.columns]
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    data_start_year = df['year'].min()
    years = sorted(df['year'].unique())
    all_predictions = []

    for year in tqdm(years, desc="Walk-Forward Ridge CV"):
        train_df = df[(df['year'] < year) & (df['year'] >= data_start_year + MIN_HISTORY_YEARS)].copy()
        test_df = df[df['year'] == year].copy()
        if test_df.empty or len(train_df) < 50: continue

        model = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=[10.0, 50.0, 100.0, 500.0]))])
        model.fit(train_df[features], train_df['log_ratio'])

        abs_pred = np.exp(model.predict(test_df[features])) * test_df['stage1_forecast']
        current_preds = test_df[['district_no', 'year', 'kreisYield', 'stage1_forecast']].copy()
        current_preds['linear_pred'] = abs_pred.clip(lower=0)
        all_predictions.append(current_preds)

    if all_predictions:
        results = pd.concat(all_predictions, ignore_index=True)
        results.to_csv(OUTPUT_DIR / 'model_c_linear_forecasts.csv', index=False)


if __name__ == "__main__":
    main()