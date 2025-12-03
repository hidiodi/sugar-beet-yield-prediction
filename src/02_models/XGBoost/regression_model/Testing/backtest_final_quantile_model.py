# File: src/02_models/XGBoost/regression_model/Testing/backtest_final_quantile_model.py
# REFACTORED (v10.0): Backtest (Residuals)
import pandas as pd
import joblib
from xgboost import XGBRegressor
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from tqdm import tqdm
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config

warnings.filterwarnings("ignore")
XGB_CONFIG = config.XGBOOST_TRAINING_CONFIG
BACKTEST_CONFIG = config.BACKTESTING_CONFIG


def run_backtest(df, models, feature_cols):
    print(f"\n--- Starting Hybrid Backtest (Residual Mode) ---")
    baseline_col = 'stat_trend_forecast'
    target_col = 'trend_residual'
    df[target_col] = df['kreisYield'] - df[baseline_col]

    valid_mask = df[baseline_col].notna() & df[target_col].notna()

    all_predictions = []
    for year in tqdm(range(BACKTEST_CONFIG['BACKTEST_START_YEAR'], BACKTEST_CONFIG['BACKTEST_END_YEAR'] + 1)):
        train_df = df[(df['year'] < year) & valid_mask].copy()
        test_df = df[(df['year'] == year) & df[baseline_col].notna()].copy()

        if test_df.empty or len(train_df) < 50: continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]

        current_preds = test_df[['district_no', 'year', 'kreisYield', baseline_col]].copy()
        for name in ['lower', 'median', 'upper']:
            model = clone(models[name])
            model.fit(X_train, y_train)

            # Reconstruction: Baseline + Predicted Residual
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
        print(f"Error: {e}"); return

    # No Lag Features
    features = [
        'trend_vs_phys_gap', 'wofost_skew', 'wofost_esp_std', 'wofost_esp_p10', 'wofost_water_stress_mean',
        'nitrogen_leaching_index', 'toxic_carryover_index', 'vector_pressure_local', 'winter_pest_kill_days',
        'antecedent_precip_sum', 'sowing_potential_days', 'winter_cropland_ndvi_anomaly',
        'avg_clay_0_30cm', 'avg_sand_0_30cm',
        'fertilizer_price_index_lag1', 'nao_winter_avg',
        'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast',
        'summer_precip_prob_wet_forecast', 'summer_temp_prob_warm_forecast',
        'spring_temp_x_antecedent_rain'
    ]
    features = [c for c in features if c in df.columns]

    results = run_backtest(df, models, features)
    if not results.empty:
        results.to_csv(report_dir / 'full_backtest_predictions.csv', index=False)
        mae = mean_absolute_error(results['kreisYield'], results['predicted_yield_median'])
        print(f"\nMAE : {mae:.2f} dt/ha")


if __name__ == "__main__":
    main()