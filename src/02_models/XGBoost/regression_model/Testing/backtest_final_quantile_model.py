# File: src/02_models/XGBoost/regression_model/Testing/backtest_final_quantile_model.py
# REFACTORED (v16.0): Backtest with Causal Baseline Fill

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from pathlib import Path
import warnings
import sys

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

warnings.filterwarnings("ignore")
XGB_CONFIG = config.XGBOOST_TRAINING_CONFIG
BACKTEST_CONFIG = config.BACKTESTING_CONFIG


def fill_missing_trend_causally(df, trend_col, yield_col):
    """Fills missing trends using Expanding Mean (Causal)."""
    df_clean = df.copy()
    missing_mask = df_clean[trend_col].isna()
    if missing_mask.sum() == 0: return df_clean

    naive_trend = df_clean.sort_values('year').groupby('district_no')[yield_col] \
        .expanding().mean().shift(1).reset_index(level=0, drop=True)

    df_clean.loc[missing_mask, trend_col] = naive_trend.loc[missing_mask]
    return df_clean


def run_backtest(df, models, feature_cols):
    print(f"\n--- Starting Hybrid Backtest (Causal Fill) ---")
    baseline_col = 'stat_trend_forecast'
    target_col = 'trend_residual'

    # 1. Fill Gaps Causally (Globally first, as Expanding Mean is historical)
    df = fill_missing_trend_causally(df, baseline_col, 'kreisYield')

    df[target_col] = df['kreisYield'] - df[baseline_col]

    valid_features = [c for c in feature_cols if c in df.columns]

    # Valid mask for Training: Must have Target
    valid_train_mask = df[target_col].notna()

    all_predictions = []

    for year in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1)):
        # Train on EVERYTHING before this year
        train_df = df[(df['year'] < year) & valid_train_mask].copy()

        # Test on this year (Must have Baseline)
        test_df = df[(df['year'] == year) & df[baseline_col].notna()].copy()

        if test_df.empty or len(train_df) < 50: continue

        X_train = train_df[valid_features]
        y_train = train_df[target_col]
        X_test = test_df[valid_features]

        current_preds = test_df[['district_no', 'year', 'kreisYield', baseline_col]].copy()

        for name in ['lower', 'median', 'upper']:
            model = clone(models[name])
            model.fit(X_train, y_train)
            current_preds[f'predicted_yield_{name}'] = current_preds[baseline_col] + model.predict(X_test)
            current_preds[f'predicted_yield_{name}'] = current_preds[f'predicted_yield_{name}'].clip(lower=0)

        all_predictions.append(current_preds)

    if not all_predictions: return pd.DataFrame()
    return pd.concat(all_predictions, ignore_index=True)


def main():
    report_dir = Path(BACKTEST_CONFIG['REPORT_DIR'])
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        models = {name: joblib.load(XGB_CONFIG[f'{name.upper()}_MODEL_PATH']) for name in ['lower', 'median', 'upper']}
        df = pd.read_csv(XGB_CONFIG['DATA_PATH'])
    except Exception as e:
        print(f"Error: {e}");
        return

    features = XGB_CONFIG['FEATURE_COLS']

    results = run_backtest(df, models, features)
    if not results.empty:
        results.to_csv(report_dir / 'full_backtest_predictions.csv', index=False)
        mae = mean_absolute_error(results['kreisYield'], results['predicted_yield_median'])
        print(f"\nMAE : {mae:.2f} dt/ha")


if __name__ == "__main__":
    main()