import pandas as pd
import numpy as np
import xgboost as xgb
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

# --- PARAMS ---
# V2: Safety (Deep trees to find the specific crash conditions)
PARAMS_V2 = {
    'n_estimators': 2000, 'learning_rate': 0.01, 'max_depth': 7,
    'subsample': 0.6, 'colsample_bytree': 0.6, 'min_child_weight': 10,
    'objective': 'reg:absoluteerror', 'n_jobs': -1, 'random_state': 42,
    'gamma': 0.5
}

# V8: Opportunity (Shallow trees to find general linear improvements)
PARAMS_V8 = {
    'n_estimators': 1500, 'learning_rate': 0.015, 'max_depth': 4,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 20,
    'objective': 'reg:absoluteerror', 'reg_alpha': 10.0,
    'n_jobs': -1, 'random_state': 42
}


def load_data_with_smart_risk():
    logging.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 1. Trend Anchor
    if 'final_corrected_forecast' not in df.columns:
        trend_path = config.MODEL_COMPARISON_CONFIG['STATISTICAL_TREND_FILE']
        trend_df = pd.read_csv(trend_path)[['year', 'district_no', 'final_corrected_forecast']]
        df = pd.merge(df, trend_df, on=['year', 'district_no'], how='left')

    # 2. Residual Target
    df['residual_target'] = df['kreisYield'] - df['final_corrected_forecast']

    # 3. --- SMART RISK CALCULATION (Fix V3) ---
    # Heat is only bad if it's dry.

    # A. Drought Intensity (0 if wet, Positive if dry)
    # Anomaly < 0 means deficit. We flip it so positive = stress.
    if 'summer_water_balance_anomaly' in df.columns:
        # -150mm is the start of real stress. -300mm is catastrophic.
        # We normalize so -150mm = 1.0 (Moderate Risk base)
        df['drought_stress'] = (df['summer_water_balance_anomaly'] * -1 / 150.0).clip(lower=0)
    else:
        df['drought_stress'] = 0

    # B. Heat Multiplier (1.0 = Neutral, 1.5 = Hot)
    if 'summer_days_tmax_gt_30c' in df.columns:
        # If hot, we multiply the drought stress.
        # 10 days = 1.2x, 20 days = 1.4x
        df['heat_multiplier'] = 1.0 + (df['summer_days_tmax_gt_30c'] / 50.0)
    else:
        df['heat_multiplier'] = 1.0

    # C. Combined Risk
    # Logic: If Wet (drought_stress=0), Risk is 0 (regardless of heat).
    # If Dry (drought_stress=1.0), and Hot (mult=1.4), Risk = 1.4.
    df['Smart_Risk_Index'] = df['drought_stress'] * df['heat_multiplier']

    # 4. Fill NaNs
    fill_cols = ['drought_stress', 'heat_multiplier', 'Smart_Risk_Index']
    for c in fill_cols:
        df[c] = df[c].fillna(0)

    return df.dropna(subset=['kreisYield', 'final_corrected_forecast', 'residual_target'])


def train_ensemble_component(df, features, target_col, params, name):
    print(f"\n--- Training {name} ---")
    available_feats = [f for f in features if f in df.columns]

    constraints_map = {
        'summer_water_balance_anomaly': 1,  # More water -> High Yield
        'summer_days_tmax_gt_30c': -1,  # More heat -> Low Yield (generally)
        'Smart_Risk_Index': -1,  # High Risk -> Low Yield
        'drought_stress': -1,
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

        if year == 2024:
            models.append(model)

    return pd.concat(preds), models[-1]


def apply_smart_ensemble(df):
    """
    Switching Logic:
    - Normal (Risk < 0.8): Trust Trend (80%) + Opportunity (20%)
    - Crisis (Risk > 1.2): Trust Safety (100%)
    - Transition: Linear blend
    """

    # 1. Low Risk (Normal/Good Years)
    # 2014 should be here (Risk ~0 because wet)
    mask_normal = df['Smart_Risk_Index'] < 0.8

    # 2. High Risk (Crash Years)
    # 2018 should be here (Risk > 1.5)
    mask_crisis = df['Smart_Risk_Index'] > 1.2

    trend = df['final_corrected_forecast']
    v2 = df['Pred_V2']  # Safety
    v8 = df['Pred_V8']  # Opportunity (Trend + Residual)

    # Vectorized Selection
    # Default (Transition Zone): 50/50 Blend
    df['Ensemble_Pred'] = (v2 * 0.5) + (v8 * 0.5)
    df['Ensemble_Mode'] = 'Transition'

    # Apply Normal Logic
    df.loc[mask_normal, 'Ensemble_Pred'] = (trend[mask_normal] * 0.85) + (v8[mask_normal] * 0.15)
    df.loc[mask_normal, 'Ensemble_Mode'] = 'Normal (Trend_Guard)'

    # Apply Crisis Logic
    df.loc[mask_crisis, 'Ensemble_Pred'] = v2[mask_crisis]
    df.loc[mask_crisis, 'Ensemble_Mode'] = 'CRISIS (Safety_First)'

    return df


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=LOG_LEVEL, format='%(message)s')

    df = load_data_with_smart_risk()

    # --- FEATURES ---
    # V2 (Safety): Needs to see the Risk Index to know *how bad* the crash is
    feats_v2 = [
        'final_corrected_forecast',
        'Smart_Risk_Index',  # <--- The corrected Signal
        'drought_stress',
        'summer_water_balance_anomaly',
        'effective_winter_water'
    ]

    # V8 (Opportunity): Focuses on good conditions
    feats_v8 = [
        'summer_water_balance_anomaly',
        'summer_days_tmax_gt_30c',
        'Smart_Risk_Index',
        'sowing_doy_anomaly',
        'avg_clay_0_30cm'
    ]

    # --- TRAIN ---
    res_v2, model_v2 = train_ensemble_component(df, feats_v2, 'kreisYield', PARAMS_V2, 'V2')
    res_v8, model_v8 = train_ensemble_component(df, feats_v8, 'residual_target', PARAMS_V8, 'V8')

    # Reconstruct V8
    res_v8 = pd.merge(res_v8, df[['year', 'district_no', 'final_corrected_forecast']], on=['year', 'district_no'])
    res_v8['Pred_V8'] = res_v8['final_corrected_forecast'] + res_v8['Pred_V8']

    # --- ENSEMBLE ---
    final = df[['year', 'district_no', 'kreisYield', 'final_corrected_forecast', 'Smart_Risk_Index']].copy()
    final = pd.merge(final, res_v2[['year', 'district_no', 'Pred_V2']], on=['year', 'district_no'])
    final = pd.merge(final, res_v8[['year', 'district_no', 'Pred_V8']], on=['year', 'district_no'])

    final = apply_smart_ensemble(final)

    # --- EVALUATION ---
    mae_t = mean_absolute_error(final['kreisYield'], final['final_corrected_forecast'])
    mae_e = mean_absolute_error(final['kreisYield'], final['Ensemble_Pred'])
    r2_e = r2_score(final['kreisYield'], final['Ensemble_Pred'])

    print("\n" + "=" * 50)
    print(" SMART ENSEMBLE (Conditional Risk Logic)")
    print("=" * 50)
    print(f"Trend MAE:    {mae_t:.2f}")
    print(f"Ensemble MAE: {mae_e:.2f}")
    print(f"Improvement:  {mae_t - mae_e:.2f}")
    print(f"Ensemble R²:  {r2_e:.3f}")

    print("\nCRITICAL YEARS FORENSICS:")
    for y in [2007, 2014, 2018]:
        sub = final[final['year'] == y]
        if sub.empty: continue

        mode_counts = sub['Ensemble_Mode'].value_counts()
        avg_risk = sub['Smart_Risk_Index'].mean()
        err_t = (sub['kreisYield'] - sub['final_corrected_forecast']).abs().mean()
        err_e = (sub['kreisYield'] - sub['Ensemble_Pred']).abs().mean()

        print(f"YEAR {y}:")
        print(f"  Avg Risk:     {avg_risk:.2f}")
        print(f"  Dominant Mode: {mode_counts.idxmax()} ({mode_counts.max()} districts)")
        print(f"  Trend Err:    {err_t:.1f}")
        print(f"  Ensemble Err: {err_e:.1f}")
        print("-" * 30)

    final.to_csv(MODEL_DIR / 'native_ensemble_forecasts.csv', index=False)
    joblib.dump(model_v2, MODEL_DIR / 'model_v2_safety.joblib')
    joblib.dump(model_v8, MODEL_DIR / 'model_v8_opportunity.joblib')
    print(f"\nModels saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()