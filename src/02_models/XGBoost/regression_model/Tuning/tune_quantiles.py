import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import os
import warnings
import optuna

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
N_TRIALS = 50
STUDY_NAME_PREFIX = "xgb_yield_quantile_hybrid_v5"
VALIDATION_START_YEAR = 2007
VALIDATION_END_YEAR = 2014
QUANTILES_TO_TUNE = [0.025, 0.5, 0.975]


def pinball_loss(y_true, y_pred, alpha):
    """Calculates the pinball loss, the correct metric for quantile regression."""
    delta = y_true - y_pred
    loss = np.maximum(alpha * delta, (alpha - 1) * delta)
    return np.mean(loss)


def load_and_prepare_data():
    """Loads data and applies causal detrending."""
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: Please run the updated feature engineering script first.")
        return None, None

    print("Applying Causal (Trailing Mean) Detrending...")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.ffill())
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.bfill())
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print("Detrending complete.")

    # --- DEFINITIVE V5 HYBRID FEATURE SET ---
    feature_cols = [
        'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
        'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
        'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast', 'summer_solar_rad_anomaly_forecast',
        'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast',
        'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',
        'lat', 'lon', 'avg_elevation', 'avg_slope',
        'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm',
        'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_anomaly',
        'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
        'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',
        'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip',
        'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
        'hot_dry_interaction', 'lat_x_summer_temp', 'sandy_soil_x_drought',
        'antecedent_gdd_sum_anomaly_sq', 'spring_temp_prob_warm_forecast_sq',
        'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq',
        'summer_precip_prob_wet_forecast_sq',
        'wofost_forecast_yield_fresh_dt',
        'wofost_forecast_x_profit_margin',
        'has_wofost_data',
        'state_encoded',
        'summer_precip_anomaly_forecast_sq',
        'is_drought_high_clay_in_state_11',
        'state6_precip_interaction',
        'summer_days_precip_gt_20mm',
        'summer_days_tmax_gt_30c'
    ]

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: The following feature columns are missing: {missing_cols}")
        print("Please ensure you have run the latest version of 'build_stage1_features.py'.")
        return None, None

    print("Data loaded and prepared successfully with the complete V5 feature set.")
    return df, feature_cols


def objective(trial, df, feature_cols, alpha):
    """Objective function for Optuna, optimizing for pinball loss using a robust walk-forward validation."""
    params = {
        'objective': 'reg:quantileerror',
        'quantile_alpha': alpha,
        'n_estimators': trial.suggest_int('n_estimators', 400, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': -1
    }

    # --- Walk-Forward / Rolling-Origin Validation ---
    all_predictions = []
    for year_to_predict in range(VALIDATION_START_YEAR, VALIDATION_END_YEAR + 1):
        train_df = df[df['year'] < year_to_predict]
        test_df = df[df['year'] == year_to_predict]
        if test_df.empty or train_df.empty:
            continue

        X_train, y_train = train_df[feature_cols], train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        final_predictions = model.predict(X_test) + test_df['yield_trend']

        fold_results = test_df[['kreisYield']].copy()

        # --- THIS IS THE CORRECTED LINE ---
        # Explanation: We create the 'predicted_yield' column directly instead of
        # trying to access it before it exists. This resolves the KeyError.
        fold_results['predicted_yield'] = final_predictions
        # --- END OF CORRECTION ---

        all_predictions.append(fold_results)

    if not all_predictions:
        return float('inf')

    results_df = pd.concat(all_predictions)
    loss = pinball_loss(results_df['kreisYield'], results_df['predicted_yield'], alpha)
    return loss


if __name__ == "__main__":
    data, feature_cols = load_and_prepare_data()

    if data is not None:
        all_best_params = {}
        for alpha in QUANTILES_TO_TUNE:
            study_name = f"{STUDY_NAME_PREFIX}_{str(alpha).replace('.', 'p')}"
            storage_name = f"sqlite:///{study_name}.db"
            study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name,
                                        load_if_exists=True)

            print(f"\n--- Starting hyperparameter tuning for QUANTILE ALPHA = {alpha} ---")
            print(f"Using study: {study_name}")

            study.optimize(
                lambda trial: objective(trial, data, feature_cols, alpha),
                n_trials=N_TRIALS,
                show_progress_bar=True
            )
            all_best_params[alpha] = study.best_params

        print("\n\n" + "=" * 50)
        print("      ALL TUNING FINISHED!")
        print("=" * 50)

        for alpha, params in all_best_params.items():
            print(f"\nBest Hyperparameters for Quantile {alpha}:")
            print(f"BEST_PARAMS_{str(alpha).upper().replace('.', 'P')} = {{")
            for key, value in params.items():
                if isinstance(value, str):
                    print(f"    '{key}': '{value}',")
                elif isinstance(value, float):
                    print(f"    '{key}': {value:.6f},")
                else:
                    print(f"    '{key}': {value},")
            print("}")
        print("\nCopy these dictionaries into your final training script.")