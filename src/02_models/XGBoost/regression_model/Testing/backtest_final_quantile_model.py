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
    print(f"\n--- Starting Champion Backtest (Strategy: Residual + Trend) ---")

    target_col = 'yield_residual'

    # Ensure we have the necessary columns
    df = df.dropna(subset=['kreisYield', 'stage1_forecast']).copy()

    # Calculate True Residual for training y
    df[target_col] = df['kreisYield'] - df['stage1_forecast']

    # Filter features (Remove Trend/Year/Yield from X)
    valid_features = [c for c in feature_cols if c in df.columns]
    cols_to_drop = ['stage1_forecast', 'year', 'year_trend', 'kreisYield', 'yield_residual']
    valid_features = [c for c in valid_features if c not in cols_to_drop]

    all_predictions = []

    for year in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1)):
        # Train on past
        train_df = df[df['year'] < year].copy()
        test_df = df[df['year'] == year].copy()

        if test_df.empty or len(train_df) < 50: continue

        X_train = train_df[valid_features]
        y_train = train_df[target_col]  # Predict Residual

        X_test = test_df[valid_features]

        # Base DataFrame for results
        current_preds = test_df[['district_no', 'year', 'kreisYield', 'stage1_forecast']].copy()

        for name in ['lower', 'median', 'upper']:
            model = clone(models[name])
            model.fit(X_train, y_train)

            # 1. Predict Residual (e.g., -50)
            pred_residual = model.predict(X_test)

            # 2. Add back Trend (e.g., 750 + -50 = 700)
            # We use the trend from the test year
            base_trend = test_df['stage1_forecast'].values
            final_yield = base_trend + pred_residual

            current_preds[f'predicted_yield_{name}'] = final_yield
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

    # Use feature cols from config
    features = XGB_CONFIG['FEATURE_COLS']
    results = run_backtest(df, models, features)

    if not results.empty:
        results.to_csv(report_dir / 'full_backtest_predictions.csv', index=False)
        mae = mean_absolute_error(results['kreisYield'], results['predicted_yield_median'])
        print(f"\nMAE : {mae:.2f} dt/ha")


if __name__ == "__main__":
    main()