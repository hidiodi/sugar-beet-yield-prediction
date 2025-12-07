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

# --- HYPERPARAMETERS (Conservative for Noisy Forecasts) ---
# We reduce depth and learning rate because the March signal is weak.
# Aggressive fitting would just memorize noise.
XGB_PARAMS_V2 = {  # Anchored Model
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 4,  # Shallow trees to avoid overfitting
    'subsample': 0.6,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'objective': 'reg:absoluteerror',
    'n_jobs': -1,
    'random_state': 42
}

XGB_PARAMS_V8 = {  # Residual Model
    'n_estimators': 1500,
    'learning_rate': 0.005,  # Very slow learning
    'max_depth': 3,  # Very shallow
    'subsample': 0.6,
    'colsample_bytree': 0.7,
    'min_child_weight': 15,
    'objective': 'reg:absoluteerror',
    'reg_alpha': 20.0,  # High regularization
    'n_jobs': -1,
    'random_state': 42
}


def load_data():
    logging.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 1. Merge with the Honest Trend Model (create_trendModel.py output)
    if 'final_corrected_forecast' not in df.columns:
        trend_path = config.MODEL_COMPARISON_CONFIG['STATISTICAL_TREND_FILE']
        if trend_path.exists():
            logging.info(f"Merging Trend Forecasts from {trend_path}...")
            trend_df = pd.read_csv(trend_path)[['year', 'district_no', 'final_corrected_forecast']]
            df = pd.merge(df, trend_df, on=['year', 'district_no'], how='left')
        else:
            logging.error("CRITICAL: Trend model file not found! Run create_trendModel.py first.")
            sys.exit(1)

    # 2. Define Residual Target (What we want V8 to predict)
    df['residual_target'] = df['kreisYield'] - df['final_corrected_forecast']

    # 3. Global Context Features (Safe March Aggregates)
    # We use the forecast probabilities for the global signal
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
    """
    Executes Strict Walk-Forward Validation.
    Training Data: All years < Target Year.
    Test Data: Target Year.
    NO FUTURE LEAKAGE.
    """
    print(f"\n--- Training {model_name} (Strict Walk-Forward) ---")
    feats = [f for f in features if f in df.columns]

    # Define Physics Constraints (Monotonicity)
    # Ensures the model respects basic biology even if the signal is noisy
    constraints_map = {
        # Bad Weather (Forecast)
        'summer_temp_prob_warm_forecast': -1,  # Higher Prob -> Lower Yield
        'heat_stress_sq': -1,  # (If available from honest swap)

        # Good Weather (Forecast)
        'summer_precip_anomaly_forecast': 1,  # More Rain -> Higher Yield
        'summer_water_balance_anomaly': 1,
        'summer_solar_rad_anomaly_forecast': 1,

        # Observed Antecedents (Safe)
        'effective_winter_water': 1,  # Wet Winter -> Higher Yield

        # Derived/Interactions
        'optimal_growth_index': 1,
        'spring_warmth_x_summer_rain': 1,

        # Anchors
        'final_corrected_forecast': 1
    }

    constraints = tuple([constraints_map.get(f, 0) for f in feats])
    run_params = params.copy()
    run_params['monotone_constraints'] = constraints

    preds = []

    # Iterate through each validation year
    years = sorted(df['year'].unique())
    for year in years:
        if year < start_year:
            continue

        # STRICT SPLIT: Train only on the past
        train = df[df['year'] < year]
        test = df[df['year'] == year]

        # Require at least 20 years of history for stability
        if len(train['year'].unique()) < 20:
            print(f"Skipping {year}: Insufficient history ({len(train['year'].unique())} years)")
            continue

        model = xgb.XGBRegressor(**run_params)
        model.fit(train[feats], train[target_col])
        p = model.predict(test[feats])

        res = test[['year', 'district_no']].copy()

        # For V8 (Residual Model), we add the prediction back to the trend
        if model_name == 'Native_V8':
            # Prediction = Trend + Predicted_Residual
            res[model_name] = test['final_corrected_forecast'] + p
        else:
            # Prediction = Raw Output
            res[model_name] = p

        preds.append(res)
        print(f"  Processed {year} (Train samples: {len(train)})")

    return pd.concat(preds)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=LOG_LEVEL, format='%(message)s')

    df = load_data()

    # --- FEATURE SETS (HONEST MARCH) ---

    # V2: Anchored Model (Safety)
    # Tries to adjust the Trend based on Winter/Forecast signals
    feats_v2 = [
        'final_corrected_forecast',  # The Anchor
        'effective_winter_water',  # The strongest observed signal
        'summer_temp_prob_warm_forecast',  # The weak forecast signal
        'summer_precip_anomaly_forecast',
        'summer_water_balance_anomaly',  # Derived Forecast
        'avg_sand_0_30cm', 'avg_clay_0_30cm',  # Soil context
        'Global_Water_Forecast',  # Global context
        'Global_Heat_Forecast'
    ]

    # V8: Residual Model (Opportunity)
    # Tries to find deviations from the trend
    feats_v8 = [
        'effective_winter_water',  # Winter Water
        'winter_buffer_x_summer_heat',  # Interaction (Winter * Forecast)
        'summer_temp_prob_warm_forecast',
        'summer_precip_anomaly_forecast',
        'optimal_growth_index',  # Forecast derived
        'spring_warmth_x_summer_rain',  # Forecast derived
        'sowing_doy_anomaly',  # Observed Sowing
        'avg_sand_0_30cm', 'avg_clay_0_30cm',
        'Global_Water_Forecast'
    ]

    # 1. Train V2 (Target: Raw Yield)
    preds_v2 = train_strict_walk_forward(df, 'Native_V2', feats_v2, 'kreisYield', XGB_PARAMS_V2)

    # 2. Train V8 (Target: Residuals)
    preds_v8 = train_strict_walk_forward(df, 'Native_V8', feats_v8, 'residual_target', XGB_PARAMS_V8)

    # 3. Merge & Save
    final = df[['year', 'district_no', 'kreisYield', 'final_corrected_forecast']].copy()
    final = pd.merge(final, preds_v2, on=['year', 'district_no'], how='inner')
    final = pd.merge(final, preds_v8, on=['year', 'district_no'], how='inner')

    output_csv = MODEL_DIR / 'native_model_comparison_v2_v8.csv'
    final.to_csv(output_csv, index=False)

    # 4. Update Ensemble (Simple Average) for the Switcher
    final['Ensemble_Pred'] = (final['Native_V2'] + final['Native_V8']) / 2
    ens_path = project_root / 'src/models/native_ensemble_champion/native_ensemble_forecasts.csv'
    ens_path.parent.mkdir(parents=True, exist_ok=True)
    final[['year', 'district_no', 'Ensemble_Pred']].to_csv(ens_path, index=False)

    # 5. Report
    mae_t = mean_absolute_error(final['kreisYield'], final['final_corrected_forecast'])
    mae_2 = mean_absolute_error(final['kreisYield'], final['Native_V2'])
    mae_8 = mean_absolute_error(final['kreisYield'], final['Native_V8'])
    mae_e = mean_absolute_error(final['kreisYield'], final['Ensemble_Pred'])

    r2_e = r2_score(final['kreisYield'], final['Ensemble_Pred'])

    print("\n" + "=" * 50)
    print(" HONEST MARCH FORECAST RESULTS (2005-2024)")
    print("=" * 50)
    print(f"Trend Model MAE: {mae_t:.2f}")
    print(f"V2 (Anchored):   {mae_2:.2f}")
    print(f"V8 (Residual):   {mae_8:.2f}")
    print(f"Ensemble:        {mae_e:.2f}")
    print(f"Ensemble R²:     {r2_e:.4f}")
    print("-" * 50)
    print("Interpretation:")
    if mae_e < mae_t:
        print(">> SUCCESS: The physics models are improving on the Trend.")
    else:
        print(">> WARNING: The Trend Model is beating the Forecast (Signal too weak).")


if __name__ == "__main__":
    main()