# File: src/02_models/XGBoost/regression_model/Testing/backtest_final_quantile_model.py
# REFACTORED (v23.0): Raw Yield Backtest (Physics-First)
# Matches the training logic of v22.0.

import pandas as pd
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


def run_backtest(df, models, feature_cols):
    print(f"\n--- Starting Champion Backtest (Raw Yield Strategy) ---")

    # Target: Raw Yield (Same as Training)
    target_col = 'kreisYield'

    # Ensure Trend is treated as a Feature
    valid_features = [c for c in feature_cols if c in df.columns]
    if 'stage1_forecast' not in valid_features:
        print("⚠️ Warning: stage1_forecast missing from features!")

    # Training needs Target
    valid_train_mask = df[target_col].notna()

    all_predictions = []

    for year in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1)):
        # Train on past
        train_df = df[(df['year'] < year) & valid_train_mask].copy()

        # Test on current
        test_df = df[(df['year'] == year)].copy()

        if test_df.empty or len(train_df) < 50: continue

        X_train = train_df[valid_features]
        y_train = train_df[target_col]
        X_test = test_df[valid_features]

        current_preds = test_df[['district_no', 'year', 'kreisYield', 'stage1_forecast']].copy()

        for name in ['lower', 'median', 'upper']:
            model = clone(models[name])
            model.fit(X_train, y_train)

            # Direct Prediction (No Baseline Addition)
            current_preds[f'predicted_yield_{name}'] = model.predict(X_test)
            # Clip negative yields (Physics constraint)
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

    # Remove 'year_trend' if it was removed in training config (Consistency Check)
    if 'year_trend' in features and 'year_trend' not in models['median'].feature_names_in_:
        features.remove('year_trend')

    results = run_backtest(df, models, features)
    if not results.empty:
        results.to_csv(report_dir / 'full_backtest_predictions.csv', index=False)
        mae = mean_absolute_error(results['kreisYield'], results['predicted_yield_median'])
        print(f"\nMAE : {mae:.2f} dt/ha")


if __name__ == "__main__":
    main()