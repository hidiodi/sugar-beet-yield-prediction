import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import sys
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
LOG_LEVEL = logging.INFO
OUTPUT_DIR = config.DATA_DIR / '06_model_output'
MODEL_DIR = Path('src/models/native_ensemble_champion')
DATA_PATH = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']

# --- PARAMS (Optimized for each Role) ---
PARAMS_V2 = {  # The Safety Officer (Anchored)
    'n_estimators': 1500, 'learning_rate': 0.015, 'max_depth': 6,
    'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 10,
    'objective': 'reg:absoluteerror', 'n_jobs': -1, 'random_state': 42
}

PARAMS_V8 = {  # The Opportunity Hunter (Unanchored)
    'n_estimators': 2000, 'learning_rate': 0.01, 'max_depth': 5,
    'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 15,
    'objective': 'reg:absoluteerror', 'reg_alpha': 10.0,
    'n_jobs': -1, 'random_state': 42
}


def load_data():
    logging.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 1. Trend Anchor
    if 'final_corrected_forecast' not in df.columns:
        trend_path = config.MODEL_COMPARISON_CONFIG['STATISTICAL_TREND_FILE']
        trend_df = pd.read_csv(trend_path)[['year', 'district_no', 'final_corrected_forecast']]
        df = pd.merge(df, trend_df, on=['year', 'district_no'], how='left')

    # 2. Residual Target
    df['residual_target'] = df['kreisYield'] - df['final_corrected_forecast']

    # 3. Global Context
    df['Global_Water_Balance'] = df.groupby('year')['summer_water_balance_anomaly'].transform('mean')
    df['Global_Heat'] = df.groupby('year')['summer_days_tmax_gt_30c'].transform('mean')

    if 'summer_solar_rad_anomaly_forecast' in df.columns:
        df['summer_solar_rad_anomaly_forecast'] = df['summer_solar_rad_anomaly_forecast'].fillna(0)
        df['Global_Solar'] = df.groupby('year')['summer_solar_rad_anomaly_forecast'].transform('mean')
    else:
        df['Global_Solar'] = 0.0

    df['Global_Water'] = df.groupby('year')['summer_water_balance_anomaly'].transform('mean')

    return df.dropna(subset=['kreisYield', 'final_corrected_forecast', 'residual_target'])


def train_ensemble_component(df, features, target_col, params, name):
    print(f"\n--- Training {name} ---")
    available_feats = [f for f in features if f in df.columns]

    # Constraints (Simplified)
    constraints_map = {
        'summer_water_balance_anomaly': 1, 'summer_days_tmax_gt_30c': -1,
        'summer_solar_rad_anomaly_forecast': 1, 'sowing_doy_anomaly': -1,
        'Global_Solar': 1, 'Global_Water': 1,
        'final_corrected_forecast': 1
    }
    constraints = tuple([constraints_map.get(f, 0) for f in available_feats])
    run_params = params.copy()
    run_params['monotone_constraints'] = constraints

    preds = []
    models = []

    for year in sorted(df['year'].unique()):
        if year < 2005: continue

        train = df[df['year'] != year]
        test = df[df['year'] == year]

        model = xgb.XGBRegressor(**run_params)
        model.fit(train[available_feats], train[target_col])
        p = model.predict(test[available_feats])

        res = test[['year', 'district_no']].copy()
        res[f'Pred_{name}'] = p
        preds.append(res)

        # Save final model (last fold acts as production model)
        if year == 2024:
            models.append(model)

    return pd.concat(preds), models[-1]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=LOG_LEVEL, format='%(message)s')

    df = load_data()

    # --- FEATURES ---

    # V2 Features (Anchored Safety)
    feats_v2 = [
        'final_corrected_forecast', 'summer_water_balance_anomaly', 'summer_days_tmax_gt_30c',
        'effective_winter_water', 'avg_sand_0_30cm', 'avg_clay_0_30cm',
        'sandy_soil_x_drought', 'clay_soil_x_drought', 'winter_buffer_x_summer_heat',
        'Global_Water_Balance', 'Global_Heat'
    ]

    # V8 Features (Unanchored Opportunity)
    feats_v8 = [
        'summer_water_balance_anomaly', 'summer_days_tmax_gt_30c', 'effective_winter_water',
        'summer_solar_rad_anomaly_forecast', 'sowing_doy_anomaly',
        'avg_sand_0_30cm', 'avg_clay_0_30cm', 'sandy_soil_x_drought', 'clay_soil_x_drought',
        'Global_Solar', 'Global_Water'
    ]

    # --- TRAIN ---

    # 1. Train V2 (Target: Raw Yield)
    res_v2, model_v2 = train_ensemble_component(df, feats_v2, 'kreisYield', PARAMS_V2, 'V2')

    # 2. Train V8 (Target: Residual)
    res_v8, model_v8 = train_ensemble_component(df, feats_v8, 'residual_target', PARAMS_V8, 'V8')
    # Add Trend back to V8 residual
    # We need to merge trend first because res_v8 only has IDs
    res_v8 = pd.merge(res_v8, df[['year', 'district_no', 'final_corrected_forecast']], on=['year', 'district_no'])
    res_v8['Pred_V8'] = res_v8['final_corrected_forecast'] + res_v8['Pred_V8']

    # --- ENSEMBLE ---

    final = df[['year', 'district_no', 'kreisYield', 'final_corrected_forecast']].copy()
    final = pd.merge(final, res_v2[['year', 'district_no', 'Pred_V2']], on=['year', 'district_no'])
    final = pd.merge(final, res_v8[['year', 'district_no', 'Pred_V8']], on=['year', 'district_no'])

    # Simple Average
    final['Ensemble_Pred'] = (final['Pred_V2'] + final['Pred_V8']) / 2

    # --- EVALUATION ---

    mae_t = mean_absolute_error(final['kreisYield'], final['final_corrected_forecast'])
    mae_e = mean_absolute_error(final['kreisYield'], final['Ensemble_Pred'])
    r2_e = r2_score(final['kreisYield'], final['Ensemble_Pred'])

    print("\n" + "=" * 50)
    print(" NATIVE ENSEMBLE CHAMPION (V2 + V8)")
    print("=" * 50)
    print(f"Trend MAE:    {mae_t:.2f}")
    print(f"Ensemble MAE: {mae_e:.2f}")
    print(f"Improvement:  {mae_t - mae_e:.2f}")
    print(f"Ensemble R²:  {r2_e:.3f}")

    print("\nCRITICAL YEARS FORENSICS:")
    for y in [2007, 2014, 2018]:
        sub = final[final['year'] == y]
        err_t = (sub['kreisYield'] - sub['final_corrected_forecast']).abs().mean()
        err_v2 = (sub['kreisYield'] - sub['Pred_V2']).abs().mean()
        err_v8 = (sub['kreisYield'] - sub['Pred_V8']).abs().mean()
        err_e = (sub['kreisYield'] - sub['Ensemble_Pred']).abs().mean()

        print(f"YEAR {y}:")
        print(f"  Trend:    {err_t:.1f}")
        print(f"  V2:       {err_v2:.1f} (Safety)")
        print(f"  V8:       {err_v8:.1f} (Opportunity)")
        print(f"  Ensemble: {err_e:.1f}  <-- FINAL")
        print("-" * 30)

    # Save
    final.to_csv(MODEL_DIR / 'native_ensemble_forecasts.csv', index=False)

    # Save Models
    joblib.dump(model_v2, MODEL_DIR / 'model_v2_safety.joblib')
    joblib.dump(model_v8, MODEL_DIR / 'model_v8_opportunity.joblib')
    print(f"\nModels saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()