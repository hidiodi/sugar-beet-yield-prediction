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
    print(f"\n--- Starting Champion Backtest (Strategy: Yield Ratio) ---")

    target_col = 'kreisYield'

    # Remove year features
    valid_features = [c for c in feature_cols if c in df.columns and c not in ['year', 'year_trend']]

    if 'stage1_forecast' not in df.columns:
        print("Error: stage1_forecast missing!")
        return pd.DataFrame()

    valid_mask = df[target_col].notna() & df['stage1_forecast'].notna()

    all_predictions = []

    for year in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1)):
        train_df = df[(df['year'] < year) & valid_mask].copy()
        test_df = df[(df['year'] == year)].copy()

        if test_df.empty or len(train_df) < 50: continue

        X_train = train_df[valid_features]
        # Train on Ratio
        y_train = train_df[target_col] / train_df['stage1_forecast']
        y_train = y_train.clip(0.5, 1.5)

        X_test = test_df[valid_features]
        trend_baseline = test_df['stage1_forecast']

        current_preds = test_df[['district_no', 'year', 'kreisYield']].copy()

        for name in ['lower', 'median', 'upper']:
            model = clone(models[name])
            model.fit(X_train, y_train)

            # Predict Ratio, then multiply by Trend
            ratio_pred = model.predict(X_test)
            current_preds[f'predicted_yield_{name}'] = ratio_pred * trend_baseline
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