# File: src/02_models/XGBoost/regression_model/Testing/backtest_standalone_xgb_model.py
# REFACTORED (v19.1): Standalone Backtest (Raw Yield) with Robust Input Handling

import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from pathlib import Path
import warnings
import sys
import numpy as np

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib
models_config = importlib.import_module("src.02_models.config")
analysis_config = importlib.import_module("src.03_analysis.config")

warnings.filterwarnings("ignore")
XGB_CONFIG = models_config.STANDALONE_XGB_CONFIG
BACKTEST_CONFIG = analysis_config.STANDALONE_BACKTESTING_CONFIG

# Burn-in for backtesting stability (ensure training set has settled stats)
MIN_HISTORY_YEARS = 10


def run_backtest(df, models, feature_cols):
    print(f"\n--- Starting Standalone Backtest (Raw Yield) ---")
    target_col = XGB_CONFIG['TARGET_COL']  # 'kreisYield'

    valid_features = [c for c in feature_cols if c in df.columns]

    # Training requires Target to exist
    valid_mask = df[target_col].notna()

    # IMPROVEMENT: Pre-filter Infinite/NaN values in features that might occur from
    # dividing by zero in expanding standard deviations
    df[valid_features] = df[valid_features].replace([np.inf, -np.inf], np.nan)

    all_predictions = []

    # Determine the absolute start of data to enforce burn-in
    data_start_year = df['year'].min()

    for year in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1)):

        # IMPROVEMENT: Enforce Burn-In on the Training Set
        # We only train on years where stats were stable (e.g., > 1987)
        train_start_threshold = data_start_year + MIN_HISTORY_YEARS

        train_df = df[(df['year'] < year) & (df['year'] >= train_start_threshold) & valid_mask].copy()

        # Test on current year
        test_df = df[(df['year'] == year)].copy()

        # Fill NaNs in Test/Train with 0 (safe fallback for Z-scores) or mean
        # This handles cases where expanding window produced NaNs
        train_df[valid_features] = train_df[valid_features].fillna(0)
        test_df[valid_features] = test_df[valid_features].fillna(0)

        if test_df.empty or len(train_df) < 50: continue

        X_train = train_df[valid_features]
        y_train = train_df[target_col]
        X_test = test_df[valid_features]

        # Calculate Ratio for Training (Just like in the main training script)
        # We model the Ratio, then multiply back
        train_forecast = train_df['stage1_forecast'].clip(lower=0.1)
        y_train_ratio = y_train / train_forecast
        y_train_ratio = y_train_ratio.clip(0.5, 1.5)

        current_preds = test_df[['district_no', 'year', 'kreisYield', 'stage1_forecast']].copy()

        for name in ['lower', 'median', 'upper']:
            model = clone(models[name])
            model.fit(X_train, y_train_ratio)  # TRAIN ON RATIO

            # Predict Ratio
            pred_ratio = model.predict(X_test)

            # Convert back to Absolute Yield
            # Yield = Ratio * Forecast
            abs_pred = pred_ratio * test_df['stage1_forecast']

            current_preds[f'predicted_yield_{name}'] = abs_pred.clip(lower=0)

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