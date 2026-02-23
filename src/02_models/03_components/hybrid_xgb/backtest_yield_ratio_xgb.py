import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from pathlib import Path
import warnings
import sys

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib

models_config = importlib.import_module("src.02_models.config")
analysis_config = importlib.import_module("src.03_analysis.config")

warnings.filterwarnings("ignore")
XGB_CONFIG = models_config.STANDALONE_XGB_CONFIG
BACKTEST_CONFIG = analysis_config.STANDALONE_BACKTESTING_CONFIG


def run_backtest(df):
    print(f"\n--- Starting Standalone Backtest (Fully Unlocked OOS) ---")
    target_col = XGB_CONFIG['TARGET_COL']

    df['district_no'] = df['district_no'].astype(str).str.zfill(5)

    # Merge Stage 2 features
    stage2_path = global_config.DATA_DIR / '05_model_input/stage2_refined_features.csv'
    if stage2_path.exists():
        df2 = pd.read_csv(stage2_path)
        df2['district_no'] = df2['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, df2, on=['year', 'district_no'], how='left')

    # CRITICAL FIX: Drop rows where target OR baseline forecast is NaN
    # This prevents np.log() from creating NaNs in the XGB target
    df.dropna(subset=[target_col, 'stage1_forecast'], inplace=True)

    exclude_cols = [
        'district_no', 'year', 'kreisYield', 'yield', 'stage1_forecast',
        'yield_ratio', 'has_wofost_data', 'state_encoded', 'year_trend'
    ]
    valid_features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]

    df[valid_features] = df[valid_features].replace([np.inf, -np.inf], np.nan)

    all_predictions = []
    data_start_year = df['year'].min()

    for year in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1)):
        # Train strictly on data up to the year before, enforcing a 10-year burn-in
        train_df = df[(df['year'] < year) & (df['year'] >= data_start_year + 10)].copy()
        test_df = df[(df['year'] == year)].copy()

        if test_df.empty or len(train_df) < 50: continue

        # Safe NaN filling for features
        train_df[valid_features] = train_df[valid_features].fillna(0)
        test_df[valid_features] = test_df[valid_features].fillna(0)

        X_train = train_df[valid_features]
        # Safety clip to prevent negative/zero yields from blowing up the log function
        y_train = train_df[target_col].clip(lower=0.1)
        X_test = test_df[valid_features]

        train_forecast = train_df['stage1_forecast'].clip(lower=0.1)

        # Calculate Log Ratio and ensure absolutely no NaNs leak through
        y_train_ratio = np.log(y_train / train_forecast).clip(-0.6, 0.6)
        y_train_ratio = y_train_ratio.fillna(0)

        current_preds = test_df[['district_no', 'year', 'kreisYield', 'stage1_forecast']].copy()

        for name in ['lower', 'median', 'upper']:
            model = XGBRegressor(
                objective='reg:quantileerror', quantile_alpha=XGB_CONFIG['QUANTILES'][name],
                n_estimators=150, max_depth=5, learning_rate=0.05,
                reg_lambda=2.0, gamma=1.0, subsample=0.8, colsample_bytree=0.5,
                random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train_ratio)

            abs_pred = np.exp(model.predict(X_test)) * test_df['stage1_forecast']
            current_preds[f'predicted_yield_{name}'] = abs_pred.clip(lower=0)

        all_predictions.append(current_preds)

    return pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()


def main():
    report_dir = Path(BACKTEST_CONFIG['REPORT_DIR'])
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(XGB_CONFIG['DATA_PATH'])

    results = run_backtest(df)
    if not results.empty:
        results.to_csv(report_dir / 'full_backtest_predictions.csv', index=False)
        mae = mean_absolute_error(results['kreisYield'], results['predicted_yield_median'])
        print(f"\nMAE : {mae:.2f} dt/ha")


if __name__ == "__main__":
    main()