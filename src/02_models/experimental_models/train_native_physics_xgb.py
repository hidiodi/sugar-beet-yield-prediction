import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
LOG_LEVEL = logging.INFO
OUTPUT_DIR = config.DATA_DIR / '06_model_output'
MODEL_DIR = Path('src/models/native_physics_comparison')
DATA_PATH = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']

# Standard Params
XGB_PARAMS_V2 = {
    'n_estimators': 1500, 'learning_rate': 0.015, 'max_depth': 6,
    'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 10,
    'objective': 'reg:absoluteerror', 'n_jobs': -1, 'random_state': 42
}
XGB_PARAMS_V8 = {
    'n_estimators': 2000, 'learning_rate': 0.01, 'max_depth': 5,
    'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 15,
    'objective': 'reg:absoluteerror', 'reg_alpha': 10.0,
    'n_jobs': -1, 'random_state': 42
}


def load_data():
    logging.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 1. Load Trend Anchor
    if 'final_corrected_forecast' not in df.columns:
        trend_path = config.MODEL_COMPARISON_CONFIG['STATISTICAL_TREND_FILE']
        if trend_path.exists():
            trend_df = pd.read_csv(trend_path)[['year', 'district_no', 'final_corrected_forecast']]
            df = pd.merge(df, trend_df, on=['year', 'district_no'], how='left')

    df['residual_target'] = df['kreisYield'] - df['final_corrected_forecast']

    # 2. Global Context (SCENARIO MODE: Uses Observed Indices)
    # We explicitly use the OBSERVED summer days column here
    if 'summer_days_tmax_gt_30c' in df.columns:
        df['Global_Heat'] = df.groupby('year')['summer_days_tmax_gt_30c'].transform('mean')
    else:
        df['Global_Heat'] = 0.0

    df['Global_Water'] = df.groupby('year')['summer_water_balance_anomaly'].transform('mean')

    if 'summer_solar_rad_anomaly_forecast' in df.columns:
        df['Global_Solar'] = df.groupby('year')['summer_solar_rad_anomaly_forecast'].transform('mean')
    else:
        df['Global_Solar'] = 0.0

    return df.dropna(subset=['kreisYield', 'final_corrected_forecast', 'residual_target'])


def train_walk_forward(df, model_name, features, target_col, params, start_year=2005):
    print(f"\n--- Training {model_name} (Walk-Forward: SCENARIO MODE) ---")
    feats = [f for f in features if f in df.columns]

    # Constraints (Scenario Mode - Uses Observed Physics)
    constraints_map = {
        'summer_days_tmax_gt_30c': -1,  # Observed Heat (The Scenario Driver)
        'heat_stress_sq': -1,  # Derived from Observed
        'Index_Failure': -1,  # Derived from Observed
        'summer_water_balance_anomaly': 1,
        'effective_winter_water': 1,
        'optimal_growth_index': 1,
        'spring_warmth_x_summer_rain': 1,
        'final_corrected_forecast': 1
    }
    constraints = tuple([constraints_map.get(f, 0) for f in feats])
    run_params = params.copy()
    run_params['monotone_constraints'] = constraints

    preds = []
    for year in sorted(df['year'].unique()):
        if year < start_year: continue
        train = df[df['year'] < year]
        test = df[df['year'] == year]
        if len(train) < 50: continue

        model = xgb.XGBRegressor(**run_params)
        model.fit(train[feats], train[target_col])
        p = model.predict(test[feats])

        res = test[['year', 'district_no']].copy()
        if model_name == 'Native_V8':
            res[model_name] = test['final_corrected_forecast'] + p
        else:
            res[model_name] = p
        preds.append(res)
        print(f"  Processed {year} (Train size: {len(train)})")

    return pd.concat(preds)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=LOG_LEVEL, format='%(message)s')
    df = load_data()

    # V2 (Crisis - Using Scenario Features)
    feats_v2 = [
        'final_corrected_forecast',
        'summer_water_balance_anomaly',
        'summer_days_tmax_gt_30c',  # OBSERVED SCENARIO INPUT
        'heat_stress_sq',
        'Index_Failure',
        'effective_winter_water', 'avg_sand_0_30cm', 'avg_clay_0_30cm',
        'sandy_soil_x_drought', 'clay_soil_x_drought',
        'winter_buffer_x_summer_heat',
        'Global_Water_Balance', 'Global_Heat'
    ]

    # V8 (Normal - Using Scenario Features)
    feats_v8 = [
        'summer_water_balance_anomaly',
        'summer_days_tmax_gt_30c',  # OBSERVED SCENARIO INPUT
        'effective_winter_water', 'optimal_growth_index',
        'spring_warmth_x_summer_rain', 'summer_solar_rad_anomaly_forecast',
        'sowing_doy_anomaly', 'avg_sand_0_30cm', 'avg_clay_0_30cm',
        'sandy_soil_x_drought', 'clay_soil_x_drought',
        'Global_Water', 'Global_Solar'
    ]

    preds_v2 = train_walk_forward(df, 'Native_V2', feats_v2, 'kreisYield', XGB_PARAMS_V2)
    preds_v8 = train_walk_forward(df, 'Native_V8', feats_v8, 'residual_target', XGB_PARAMS_V8)

    final = df[['year', 'district_no', 'kreisYield', 'final_corrected_forecast']].copy()
    final = pd.merge(final, preds_v2, on=['year', 'district_no'], how='inner')
    final = pd.merge(final, preds_v8, on=['year', 'district_no'], how='inner')

    output_csv = MODEL_DIR / 'native_model_comparison_v2_v8.csv'
    final.to_csv(output_csv, index=False)

    # Create Ensemble
    final['Ensemble_Pred'] = (final['Native_V2'] + final['Native_V8']) / 2
    ens_path = project_root / 'src/models/native_ensemble_champion/native_ensemble_forecasts.csv'
    ens_path.parent.mkdir(parents=True, exist_ok=True)
    final[['year', 'district_no', 'Ensemble_Pred']].to_csv(ens_path, index=False)

    print("\nSCENARIO ANALYSIS RESULTS (V2+V8):")
    print(f"Trend MAE: {mean_absolute_error(final['kreisYield'], final['final_corrected_forecast']):.2f}")
    print(f"V2 MAE:    {mean_absolute_error(final['kreisYield'], final['Native_V2']):.2f}")
    print(f"V8 MAE:    {mean_absolute_error(final['kreisYield'], final['Native_V8']):.2f}")


if __name__ == "__main__":
    main()