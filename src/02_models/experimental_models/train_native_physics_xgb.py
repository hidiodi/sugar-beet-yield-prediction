import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

LOG_LEVEL = logging.INFO
OUTPUT_DIR = config.DATA_DIR / '06_model_output'
MODEL_DIR = Path('src/models/native_physics_comparison')
DATA_PATH = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']

# --- HYPERPARAMETERS (Tuned for Hybrid Inputs) ---
# We use deeper trees because WOFOST features are high-quality signals
XGB_PARAMS_V2 = {
    'n_estimators': 1500, 'learning_rate': 0.01, 'max_depth': 5,
    'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 10,
    'objective': 'reg:absoluteerror', 'n_jobs': -1, 'random_state': 42
}
XGB_PARAMS_V8 = {
    'n_estimators': 2000, 'learning_rate': 0.005, 'max_depth': 4,
    'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 15,
    'objective': 'reg:absoluteerror', 'reg_alpha': 10.0, 'n_jobs': -1, 'random_state': 42
}


def load_data():
    logging.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 1. Trend Anchor
    if 'final_corrected_forecast' not in df.columns:
        trend_path = config.MODEL_COMPARISON_CONFIG['STATISTICAL_TREND_FILE']
        trend_df = pd.read_csv(trend_path)[['year', 'district_no', 'final_corrected_forecast']]
        df = pd.merge(df, trend_df, on=['year', 'district_no'], how='left')

    df['residual_target'] = df['kreisYield'] - df['final_corrected_forecast']

    # 2. Check for WOFOST Features (The "Brain")
    # If build_stage1_features.py ran correctly, these should be here.
    if 'wofost_yield_water_limited' not in df.columns:
        logging.warning("WOFOST features missing! Model will be weak.")
        df['wofost_yield_water_limited'] = df['final_corrected_forecast']  # Fallback
        df['anoxia_events'] = 0.0

    # 3. Global Context (Using Forecasts + WOFOST)
    if 'summer_temp_prob_warm_forecast' in df.columns:
        df['Global_Heat_Forecast'] = df.groupby('year')['summer_temp_prob_warm_forecast'].transform('mean')
    else:
        df['Global_Heat_Forecast'] = 0.0

    df['Global_WOFOST'] = df.groupby('year')['wofost_yield_water_limited'].transform('mean')

    return df.dropna(subset=['kreisYield', 'final_corrected_forecast', 'residual_target'])


def train_strict_walk_forward(df, model_name, features, target_col, params, start_year=2005):
    """
    STRICT WALK-FORWARD (HONEST).
    No future data. No LOYO.
    """
    print(f"\n--- Training {model_name} (Strict Walk-Forward + WOFOST) ---")
    feats = [f for f in features if f in df.columns]

    # Constraints: Ensure the model respects the Physics Engine
    constraints_map = {
        'wofost_yield_water_limited': 1,  # If WOFOST says high yield -> Predict High
        'anoxia_events': -1,  # If WOFOST sees oxygen stress -> Predict Low
        'summer_temp_prob_warm_forecast': -1,
        'effective_winter_water': 1,
        'final_corrected_forecast': 1
    }
    constraints = tuple([constraints_map.get(f, 0) for f in feats])
    run_params = params.copy()
    run_params['monotone_constraints'] = constraints

    preds = []
    years = sorted(df['year'].unique())

    for year in years:
        if year < start_year: continue

        # STRICT PAST ONLY
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

    # --- FEATURE SETS (HYBRID PHYSICS) ---

    # V2 (Crisis): Needs to detect failure
    feats_v2 = [
        'final_corrected_forecast',  # Trend
        'wofost_yield_water_limited',  # BIOPHYSICAL OPINION (Crucial)
        'anoxia_events',  # WOFOST Stress Metric
        'effective_winter_water',  # Soil Memory
        'summer_temp_prob_warm_forecast',  # Forecast
        'avg_sand_0_30cm',
        'Global_WOFOST'  # Global Context
    ]

    # V8 (Opportunity): Needs to detect potential
    feats_v8 = [
        'wofost_yield_water_limited',  # BIOPHYSICAL OPINION
        'effective_winter_water',
        'optimal_growth_index',
        'summer_precip_anomaly_forecast',
        'sowing_doy_anomaly',
        'avg_clay_0_30cm',
        'Global_WOFOST'
    ]

    preds_v2 = train_strict_walk_forward(df, 'Native_V2', feats_v2, 'kreisYield', XGB_PARAMS_V2)
    preds_v8 = train_strict_walk_forward(df, 'Native_V8', feats_v8, 'residual_target', XGB_PARAMS_V8)

    final = df[['year', 'district_no', 'kreisYield', 'final_corrected_forecast']].copy()
    final = pd.merge(final, preds_v2, on=['year', 'district_no'], how='inner')
    final = pd.merge(final, preds_v8, on=['year', 'district_no'], how='inner')

    output_csv = MODEL_DIR / 'native_model_comparison_v2_v8.csv'
    final.to_csv(output_csv, index=False)

    final['Ensemble_Pred'] = (final['Native_V2'] + final['Native_V8']) / 2
    ens_path = config.DATA_DIR / '06_model_output/native_ensemble_champion/native_ensemble_forecasts.csv'
    ens_path.parent.mkdir(parents=True, exist_ok=True)
    final[['year', 'district_no', 'Ensemble_Pred']].to_csv(ens_path, index=False)

    print("\nHYBRID WALK-FORWARD RESULTS:")
    print(f"Trend MAE: {mean_absolute_error(final['kreisYield'], final['final_corrected_forecast']):.2f}")
    print(f"V2 MAE:    {mean_absolute_error(final['kreisYield'], final['Native_V2']):.2f}")
    print(f"Ensemble:  {mean_absolute_error(final['kreisYield'], final['Ensemble_Pred']):.2f}")


if __name__ == "__main__":
    main()