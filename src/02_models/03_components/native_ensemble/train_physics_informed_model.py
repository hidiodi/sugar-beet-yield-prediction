import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib
models_config = importlib.import_module("src.02_models.config")
analysis_config = importlib.import_module("src.03_analysis.config")

LOG_LEVEL = logging.INFO
OUTPUT_DIR = global_config.DATA_DIR / '06_model_output'
MODEL_DIR = Path('src/models/native_physics_comparison')
DATA_PATH = models_config.XGBOOST_TRAINING_CONFIG['DATA_PATH']

# --- QUANTILE HYPERPARAMETERS ---
# We use the same parameters for both but change the Quantile Alpha.
# This aligns with the "Standalone" success.

COMMON_PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 4,  # Moderate depth to prevent overfitting
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 20,  # Robustness constraint
    'objective': 'reg:quantileerror',
    'n_jobs': -1,
    'random_state': 42
}

# V2: PREDICT THE FLOOR (Conservative / Risk)
PARAMS_V2 = COMMON_PARAMS.copy()
PARAMS_V2['quantile_alpha'] = 0.20  # Predict the 20th percentile (Bad Year Outcome)

# V8: PREDICT THE CEILING (Optimistic / Potential)
PARAMS_V8 = COMMON_PARAMS.copy()
PARAMS_V8['quantile_alpha'] = 0.80  # Predict the 80th percentile (Good Year Outcome)


def load_and_engineer_data():
    logging.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    if 'final_corrected_forecast' not in df.columns:
        trend_path = analysis_config.MODEL_COMPARISON_CONFIG['STATISTICAL_TREND_FILE']
        trend_df = pd.read_csv(trend_path)[['year', 'district_no', 'final_corrected_forecast']]
        df = pd.merge(df, trend_df, on=['year', 'district_no'], how='left')

    # --- TARGET: RATIO ---
    # We predict the multiplier relative to the trend (e.g., 0.9, 1.1)
    df['target_ratio'] = df['kreisYield'] / df['final_corrected_forecast']

    # Clip extreme outliers (e.g. crop failures < 0.5 or data errors > 1.5)
    # to prevent them from skewing the quantile learning.
    df['target_ratio'] = df['target_ratio'].clip(0.5, 1.5)

    if 'wofost_yield_water_limited' not in df.columns:
        df['wofost_yield_water_limited'] = df['final_corrected_forecast']

    # --- PHYSICS SIGNALS ---
    df['Wofost_Ratio'] = df['wofost_yield_water_limited'] / (df['final_corrected_forecast'] + 1)

    # Interaction: Heat Stress per unit of Water
    if 'summer_temp_prob_warm_forecast' in df.columns and 'effective_winter_water' in df.columns:
        df['Stress_Index'] = df['summer_temp_prob_warm_forecast'] / (df['effective_winter_water'] + 10)
    else:
        df['Stress_Index'] = 0

    return df.dropna(subset=['kreisYield', 'final_corrected_forecast', 'target_ratio'])


def train_quantile_walk_forward(df, model_name, features, params, start_year=2005):
    print(f"\n--- Training {model_name} (Quantile: {params['quantile_alpha']}) ---")
    feats = [f for f in features if f in df.columns]

    preds = []
    years = sorted(df['year'].unique())

    for year in years:
        if year < start_year: continue

        # STRICT WALK-FORWARD: Train on ALL past data (Bad & Good years mixed)
        train = df[df['year'] < year]
        test = df[df['year'] == year]

        if len(train) < 50: continue

        model = xgb.XGBRegressor(**params)
        model.fit(train[feats], train['target_ratio'])

        # Predict Ratio
        p_ratio = model.predict(test[feats])

        res = test[['year', 'district_no']].copy()
        res[model_name] = p_ratio
        preds.append(res)
        print(f"  Processed {year} (Train size: {len(train)})")

    return pd.concat(preds)


def safe_ensemble(row):
    """
    The Safety-First Ensemble.
    Default: Trust the Trend (1.0).
    Trigger: Only if WOFOST signal is strong, Blend Trend with Quantile Model.
    """
    trend_val = row['final_corrected_forecast']

    # The Signals
    wofost_signal = row['Wofost_Ratio']  # e.g., 0.90 means Physics says -10%

    # The Specialists
    v2_floor = row['Native_V2']  # e.g., 0.85
    v8_ceiling = row['Native_V8']  # e.g., 1.15

    # --- LOGIC ---

    # Case 1: Physics indicates Significant Stress (< 0.95)
    if wofost_signal < 0.95:
        # We suspect a bad year.
        # Blend Trend (Conservative) with V2 (Pessimistic Floor)
        # Weight: 60% Trend, 40% V2. (Don't go all the way to the floor)
        ratio_pred = (0.6 * 1.0) + (0.4 * v2_floor)
        return trend_val * ratio_pred

    # Case 2: Physics indicates Significant Growth (> 1.05)
    elif wofost_signal > 1.05:
        # We suspect a bumper year.
        # Blend Trend (Conservative) with V8 (Optimistic Ceiling)
        # Weight: 60% Trend, 40% V8.
        ratio_pred = (0.6 * 1.0) + (0.4 * v8_ceiling)
        return trend_val * ratio_pred

    # Case 3: Physics is Neutral (0.95 - 1.05)
    else:
        # Trust the Trend completely.
        return trend_val


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=LOG_LEVEL, format='%(message)s')
    df = load_and_engineer_data()

    # --- FEATURE SETS ---
    # V2 (Floor): Needs features that signal disaster
    feats_v2 = [
        'Wofost_Ratio',
        'Stress_Index',
        'anoxia_events',
        'summer_temp_prob_warm_forecast'
    ]

    # V8 (Ceiling): Needs features that signal abundance
    feats_v8 = [
        'Wofost_Ratio',
        'effective_winter_water',
        'optimal_growth_index',
        'avg_clay_0_30cm'
    ]

    # --- TRAIN ON ALL DATA (Quantile Regression) ---
    preds_v2 = train_quantile_walk_forward(df, 'Native_V2', feats_v2, PARAMS_V2)
    preds_v8 = train_quantile_walk_forward(df, 'Native_V8', feats_v8, PARAMS_V8)

    # Merge
    final = df[['year', 'district_no', 'kreisYield', 'final_corrected_forecast', 'Wofost_Ratio']].copy()
    final = pd.merge(final, preds_v2, on=['year', 'district_no'], how='inner')
    final = pd.merge(final, preds_v8, on=['year', 'district_no'], how='inner')

    # Apply Ensemble
    final['Ensemble_Pred'] = final.apply(safe_ensemble, axis=1)

    output_csv = MODEL_DIR / 'native_model_comparison_quantile.csv'
    final.to_csv(output_csv, index=False)

    ens_path = global_config.DATA_DIR / '06_model_output/native_ensemble_champion/native_ensemble_forecasts.csv'
    ens_path.parent.mkdir(parents=True, exist_ok=True)
    final[['year', 'district_no', 'Ensemble_Pred']].to_csv(ens_path, index=False)

    print("\n--- QUANTILE ENSEMBLE RESULTS ---")
    print(f"Trend MAE:     {mean_absolute_error(final['kreisYield'], final['final_corrected_forecast']):.2f}")
    print(f"Ensemble MAE:  {mean_absolute_error(final['kreisYield'], final['Ensemble_Pred']):.2f}")


if __name__ == "__main__":
    main()