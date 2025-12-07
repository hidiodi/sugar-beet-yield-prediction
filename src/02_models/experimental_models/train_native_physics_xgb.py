import pandas as pd
import numpy as np
import xgboost as xgb
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
MODEL_DIR = Path('src/models/native_physics_comparison')
DATA_PATH = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
INDICES_PATH = config.DATA_DIR / '02_intermediate/climateIndices/long_range_climate_features.csv'

# --- HYPERPARAMETERS (Conservative for Walk-Forward) ---
# We use slightly deeper trees to allow interactions between Econ and Weather
XGB_PARAMS_V2 = {
    'n_estimators': 1500,
    'learning_rate': 0.01,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'objective': 'reg:absoluteerror',
    'n_jobs': -1,
    'random_state': 42
}

XGB_PARAMS_V8 = {
    'n_estimators': 2000,
    'learning_rate': 0.005,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 15,
    'objective': 'reg:absoluteerror',
    'reg_alpha': 15.0,
    'n_jobs': -1,
    'random_state': 42
}


def load_data():
    logging.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 1. Merge Trend Anchor
    if 'final_corrected_forecast' not in df.columns:
        trend_path = config.MODEL_COMPARISON_CONFIG['STATISTICAL_TREND_FILE']
        if trend_path.exists():
            trend_df = pd.read_csv(trend_path)[['year', 'district_no', 'final_corrected_forecast']]
            df = pd.merge(df, trend_df, on=['year', 'district_no'], how='left')

    df['residual_target'] = df['kreisYield'] - df['final_corrected_forecast']

    # 2. Merge Climate Indices (The Missing Link)
    if INDICES_PATH.exists():
        logging.info(f"Merging Climate Indices from {INDICES_PATH}...")
        indices = pd.read_csv(INDICES_PATH)
        indices.columns = indices.columns.str.strip()  # Safety clean
        indices['year'] = indices['year'].astype(int)
        df = pd.merge(df, indices, on='year', how='left')

        # Fill missing with neutral
        for c in ['nao_winter_avg', 'sca_winter_avg', 'enso_mei_winter_avg']:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)
            else:
                df[c] = 0.0
    else:
        logging.warning("Climate Indices not found! Model will be weaker.")

    # 3. Global Context (Forecast Based)
    if 'summer_precip_anomaly_forecast' in df.columns:
        df['Global_Water_Forecast'] = df.groupby('year')['summer_precip_anomaly_forecast'].transform('mean')
    else:
        df['Global_Water_Forecast'] = 0.0

    if 'summer_temp_prob_warm_forecast' in df.columns:
        df['Global_Heat_Forecast'] = df.groupby('year')['summer_temp_prob_warm_forecast'].transform('mean')
    else:
        df['Global_Heat_Forecast'] = 0.0

    return df.dropna(subset=['kreisYield', 'final_corrected_forecast', 'residual_target'])


def train_strict_walk_forward(df, model_name, features, target_col, params, start_year=2005):
    """Strict Walk-Forward (No Future Leaks)."""
    print(f"\n--- Training {model_name} (Strict Walk-Forward + Econ/Climate) ---")
    feats = [f for f in features if f in df.columns]

    # Constraints (Logic: Econ+, Weather+, Climate?)
    constraints_map = {
        # Weather (Forecast)
        'summer_temp_prob_warm_forecast': -1,
        'summer_precip_anomaly_forecast': 1,
        'effective_winter_water': 1,

        # Economics (Strong Signal)
        'profit_margin_proxy_lag1': 1,  # High Margin -> Better Management -> High Yield
        'cost_of_inputs_lag1': -1,  # High Cost -> Lower Inputs -> Lower Yield

        # Climate Indices (Complex, usually unconstrained or tested)
        # We leave them unconstrained as their impact varies by region

        # Globals
        'Global_Water_Forecast': 1,
        'final_corrected_forecast': 1
    }

    # Map constraints, default to 0 (Unconstrained) if not listed
    constraints = tuple([constraints_map.get(f, 0) for f in feats])
    run_params = params.copy()
    run_params['monotone_constraints'] = constraints

    preds = []

    for year in sorted(df['year'].unique()):
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

    # --- FEATURE SETS (MAXIMUM INFORMATION) ---

    # V2 (Anchored): Needs stability
    feats_v2 = [
        'final_corrected_forecast',
        # Weather (Observed/Forecast)
        'effective_winter_water',
        'summer_temp_prob_warm_forecast',
        'summer_precip_anomaly_forecast',

        # Economics (The "Human Factor")
        'profit_margin_proxy_lag1',
        'cost_of_inputs_lag1',

        # Climate (The "Macro Factor")
        'nao_winter_avg',
        'sca_winter_avg',

        # Context
        'avg_sand_0_30cm', 'avg_clay_0_30cm',
        'Global_Water_Forecast'
    ]

    # V8 (Residual): Needs drivers of deviation
    feats_v8 = [
        # Weather
        'effective_winter_water',
        'summer_temp_prob_warm_forecast',
        'summer_precip_anomaly_forecast',
        'spring_warmth_x_summer_rain',
        'optimal_growth_index',

        # Economics
        'profit_margin_proxy_lag1',

        # Climate
        'nao_winter_avg',
        'sca_winter_avg',
        'enso_mei_winter_avg',

        # Context
        'avg_sand_0_30cm',
        'Global_Water_Forecast'
    ]

    # Train
    preds_v2 = train_strict_walk_forward(df, 'Native_V2', feats_v2, 'kreisYield', XGB_PARAMS_V2)
    preds_v8 = train_strict_walk_forward(df, 'Native_V8', feats_v8, 'residual_target', XGB_PARAMS_V8)

    # Merge & Save
    final = df[['year', 'district_no', 'kreisYield', 'final_corrected_forecast']].copy()
    final = pd.merge(final, preds_v2, on=['year', 'district_no'], how='inner')
    final = pd.merge(final, preds_v8, on=['year', 'district_no'], how='inner')

    output_csv = MODEL_DIR / 'native_model_comparison_v2_v8.csv'
    final.to_csv(output_csv, index=False)

    # Update Ensemble
    final['Ensemble_Pred'] = (final['Native_V2'] + final['Native_V8']) / 2
    ens_path = project_root / 'src/models/native_ensemble_champion/native_ensemble_forecasts.csv'
    ens_path.parent.mkdir(parents=True, exist_ok=True)
    final[['year', 'district_no', 'Ensemble_Pred']].to_csv(ens_path, index=False)

    # Results
    mae_t = mean_absolute_error(final['kreisYield'], final['final_corrected_forecast'])
    mae_e = mean_absolute_error(final['kreisYield'], final['Ensemble_Pred'])
    r2_e = r2_score(final['kreisYield'], final['Ensemble_Pred'])

    print("\n" + "=" * 50)
    print(" STRICT WALK-FORWARD RESULTS (ECON + CLIMATE)")
    print("=" * 50)
    print(f"Trend MAE:    {mae_t:.2f}")
    print(f"Ensemble MAE: {mae_e:.2f}")
    print(f"Ensemble R²:  {r2_e:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()