import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV
from pygam import LinearGAM, s, l, te
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import argparse
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore")

# Define the fixed, non-overlapping periods for our two-stage process
CALIBRATION_START_YEAR = 1990
CALIBRATION_END_YEAR = 2000
EVALUATION_START_YEAR = 2001
EVALUATION_END_YEAR = 2024

# Paths and constants
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
BASE_OUTPUT_DIR = 'reports/hybrid_stable_calibration_results'

# Feature set and model parameters from previous scripts
FEATURE_COLS = [
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
    'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
    'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
    'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast', 'spring_temp_prob_warm_forecast',
    'spring_precip_prob_wet_forecast', 'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',
    'lat', 'lon', 'avg_elevation', 'avg_slope', 'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm',
    'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'fertilizer_price_index_lag1_anomaly_capped',
    'is_fertilizer_price_extreme', 'is_summer_forecast_dry', 'gdd_x_fertilizer_price',
    'spring_temp_x_spring_precip', 'antecedent_gdd_sum_anomaly_sq', 'summer_heat_x_profit_margin',
    'summer_precip_x_input_costs'
]
XGB_PARAMS = {
    'n_estimators': 914, 'learning_rate': 0.026114, 'max_depth': 5, 'subsample': 0.922850,
    'colsample_bytree': 0.811573, 'gamma': 1.830853, 'min_child_weight': 2,
    'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1
}
TARGET_COL = 'kreisYield'
DETRENDED_TARGET_COL = 'kreisYield_detrended'


def get_gam_terms():
    """Constructs the formula for the LinearGAM model using integer indices."""
    # This function remains the same, ensuring GAM works correctly.
    smooth_features = ['antecedent_gdd_sum_anomaly', 'spring_temp_anomaly_forecast', 'summer_temp_anomaly_forecast',
                       'avg_elevation', 'avg_clay_0_30cm', 'avg_sand_0_30cm', 'gdd_x_fertilizer_price']
    linear_features = [f for f in FEATURE_COLS if f not in smooth_features + ['lat', 'lon']]

    lat_idx, lon_idx = FEATURE_COLS.index('lat'), FEATURE_COLS.index('lon')
    terms = te(lat_idx, lon_idx)
    for feat in smooth_features:
        terms += s(FEATURE_COLS.index(feat))
    for feat in linear_features:
        terms += l(FEATURE_COLS.index(feat))
    return terms


def run_calibration_phase(df: pd.DataFrame, primary_model_type: str):
    """
    Stage 1: Run a mini-backtest on the calibration period to find stable q-multipliers.
    """
    print(f"\n--- STAGE 1: CALIBRATION PHASE ({CALIBRATION_START_YEAR}-{CALIBRATION_END_YEAR}) ---")

    all_normalized_residuals = []

    for cal_year in tqdm(range(CALIBRATION_START_YEAR, CALIBRATION_END_YEAR + 1), desc="Calibrating"):
        # 1. Split data: train on everything BEFORE the calibration year
        train_df = df[df['year'] < cal_year].copy()
        cal_holdout_df = df[df['year'] == cal_year].copy()

        if cal_holdout_df.empty or train_df.empty:
            continue

        X_train, y_train = train_df[FEATURE_COLS], train_df[DETRENDED_TARGET_COL]
        X_cal_holdout, y_cal_holdout = cal_holdout_df[FEATURE_COLS], cal_holdout_df[DETRENDED_TARGET_COL]

        # 2. Train Primary Model
        scaler = StandardScaler().fit(X_train)
        X_train_scaled, X_cal_holdout_scaled = scaler.transform(X_train), scaler.transform(X_cal_holdout)

        if primary_model_type == 'lasso':
            primary_model = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)
        else:  # gam
            primary_model = LinearGAM(get_gam_terms(), max_iter=500).fit(X_train_scaled, y_train)

        # 3. Train Secondary (Error) Model
        train_errors = y_train - primary_model.predict(X_train_scaled)
        secondary_model = XGBRegressor(**XGB_PARAMS).fit(X_train, np.abs(train_errors))

        # 4. Calculate Normalized Residuals on the holdout year
        preds_primary = primary_model.predict(X_cal_holdout_scaled)
        error_magnitudes = secondary_model.predict(X_cal_holdout)
        holdout_residuals = y_cal_holdout - preds_primary

        normalized_residuals = holdout_residuals / (error_magnitudes + 1e-6)
        all_normalized_residuals.extend(normalized_residuals)

    # 5. Aggregate and find the final, stable q-multipliers
    q_lower, q_upper = np.percentile(all_normalized_residuals, [2.5, 97.5])

    print("\n--- Calibration Complete ---")
    print(f"Found stable multipliers based on {len(all_normalized_residuals)} out-of-sample points:")
    print(f"  q_lower (2.5th percentile): {q_lower:.4f}")
    print(f"  q_upper (97.5th percentile): {q_upper:.4f}")

    return q_lower, q_upper


def run_evaluation_phase(df: pd.DataFrame, q_lower: float, q_upper: float, primary_model_type: str):
    """
    Stage 2: Run the real backtest on the evaluation period using the fixed q-multipliers.
    """
    print(f"\n--- STAGE 2: EVALUATION PHASE ({EVALUATION_START_YEAR}-{EVALUATION_END_YEAR}) ---")

    all_predictions = []

    for eval_year in tqdm(range(EVALUATION_START_YEAR, EVALUATION_END_YEAR + 1), desc="Evaluating"):
        # 1. Split data: train on everything BEFORE the evaluation year
        train_df = df[df['year'] < eval_year].copy()
        test_df = df[df['year'] == eval_year].copy()

        if test_df.empty or train_df.empty:
            continue

        X_train, y_train = train_df[FEATURE_COLS], train_df[DETRENDED_TARGET_COL]
        X_test = test_df[FEATURE_COLS]

        # 2. Train Primary and Secondary models (same as calibration)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

        if primary_model_type == 'lasso':
            primary_model = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)
        else:  # gam
            primary_model = LinearGAM(get_gam_terms(), max_iter=500).fit(X_train_scaled, y_train)

        train_errors = y_train - primary_model.predict(X_train_scaled)
        secondary_model = XGBRegressor(**XGB_PARAMS).fit(X_train, np.abs(train_errors))

        # 3. Construct Final Interval using the FIXED q-multipliers
        y_pred_detrended = primary_model.predict(X_test_scaled)
        y_pred_final = y_pred_detrended + test_df['yield_trend'].values

        error_pred_final = secondary_model.predict(X_test)

        lower_bound = y_pred_final + q_lower * error_pred_final
        upper_bound = y_pred_final + q_upper * error_pred_final

        # 4. Store results
        fold_results = test_df[['district_no', 'year', TARGET_COL]].copy()
        fold_results['predicted_yield'] = y_pred_final
        fold_results['lower_bound'] = lower_bound
        fold_results['upper_bound'] = upper_bound
        all_predictions.append(fold_results)

    return pd.concat(all_predictions, ignore_index=True)


def main(primary_model_type: str):
    """
    Orchestrates the two-stage calibration and evaluation pipeline.
    """
    print(f"--- Starting Stable Calibration Backtest for Primary Model: {primary_model_type.upper()} ---")

    # --- 1. Load and Prepare Data ---
    df = pd.read_csv(DATA_PATH)
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing features: {missing_cols}");
        return

    print("\n--- Applying Causal Detrending ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')[TARGET_COL].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(method='ffill').fillna(x.iloc[0]))
    df.dropna(subset=['yield_trend'], inplace=True)
    df[DETRENDED_TARGET_COL] = df[TARGET_COL] - df['yield_trend']
    print(" -> Detrending complete.")

    # --- 2. Run the Two-Stage Process ---
    q_lower, q_upper = run_calibration_phase(df, primary_model_type)
    final_results_df = run_evaluation_phase(df, q_lower, q_upper, primary_model_type)

    # --- 3. Save Final Results ---
    if final_results_df.empty:
        print("❌ CRITICAL: No predictions were generated in the evaluation phase.")
        return

    output_dir = os.path.join(BASE_OUTPUT_DIR, f'hybrid_{primary_model_type}_run')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'hybrid_{primary_model_type}_results.csv')
    final_results_df.to_csv(output_path, index=False)

    print(f"\n\n--- Backtest Complete ---")
    print(f"✅ Final evaluation results for '{primary_model_type}' saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a two-stage, stable-calibration backtest for hybrid models.")
    parser.add_argument("--primary_model", type=str, required=True, choices=["lasso", "gam"],
                        help="The primary model to use: 'lasso' or 'gam'.")
    args = parser.parse_args()
    main(args.primary_model)