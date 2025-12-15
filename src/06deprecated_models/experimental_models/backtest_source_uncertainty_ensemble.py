# File: src/models/backtest_source_uncertainty_ensemble.py
# Description: The definitive backtest for a Deep Ensemble trained on diverse
#              climate futures from the SEAS5 ensemble. This propagates source
#              uncertainty through the modeling chain.

import pandas as pd
import numpy as np
import geopandas as gpd
from xgboost import XGBRegressor
import os
import warnings
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
# This script requires a more complex feature engineering pipeline.
# We first need to merge the member-specific forecast features with the static features.
MEMBER_FEATURES_PATH = 'data/02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv'
STATIC_FEATURES_PATH = 'data/05_model_input/stage1_preseason_features.csv'  # Contains non-forecast features
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = 'reports/figures/district_level_diagnostics/source_uncertainty_ensemble'

# --- Model Parameters ---
N_ENSEMBLE_MODELS = 10
BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2024
BASE_PARAMS = {  # Using slightly simpler params for speed, can be tuned later
    'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 5,
    'subsample': 0.9, 'colsample_bytree': 0.8, 'gamma': 1.5,
    'min_child_weight': 2, 'n_jobs': -1
}
BASE_MODEL = XGBRegressor(**BASE_PARAMS)


def load_and_merge_data():
    """Loads and merges member-specific and static features to create the full model input."""
    print("--- Loading and Merging All Feature Sources ---")
    try:
        member_df = pd.read_csv(MEMBER_FEATURES_PATH)
        static_df = pd.read_csv(STATIC_FEATURES_PATH)
    except FileNotFoundError as e:
        print(f"❌ Error: Missing required feature file. {e}")
        return None

    # Select only the non-forecast features from the static file
    static_cols_to_keep = [
        'district_no', 'year', 'kreisYield', 'lat', 'lon', 'avg_elevation',
        'avg_slope', 'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm',
        'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'avg_bdod_0_100cm',
        'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm',
        'avg_phh2o_0_100cm', 'winter_cropland_ndvi_mean',
        'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean',
        'winter_cropland_LST_anomaly', 'winter_cropland_snow_cover_days',
        'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',
        'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly',
        'antecedent_gdd_sum_anomaly', 'is_summer_forecast_dry', 'gdd_x_fertilizer_price',
        'spring_temp_x_spring_precip', 'antecedent_gdd_sum_anomaly_sq',
        'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
        # Drop squared probability features as we're not using probabilities
    ]
    static_df_subset = static_df[static_cols_to_keep]

    # Merge the two dataframes
    # This will expand the static features to match every SEAS5 member
    full_df = pd.merge(member_df, static_df_subset, on=['year', 'district_no'])

    # Apply detrending to the full, merged dataset
    print("-> Applying Causal Detrending...")
    full_df.sort_values(by=['district_no', 'year', 'seas5_member'], inplace=True)
    # Detrending is based on the actual yield, which is constant across members for a given year
    # We calculate the trend on the non-expanded data first for correctness
    trend = static_df_subset.sort_values(by=['district_no', 'year']).groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    trend_df = pd.concat([static_df_subset[['year', 'district_no']], trend.rename('yield_trend')], axis=1)

    full_df = pd.merge(full_df, trend_df, on=['year', 'district_no'])
    full_df.dropna(subset=['yield_trend'], inplace=True)
    full_df['kreisYield_detrended'] = full_df['kreisYield'] - full_df['yield_trend']

    print(f"✅ Data fully merged and preprocessed. Total rows: {len(full_df)}")
    return full_df


def run_source_uncertainty_backtest(df: pd.DataFrame, feature_cols: list):
    """Performs a rolling backtest using the diverse-future dataset."""
    print("\n--- Starting Source Uncertainty Ensemble Backtest ---")
    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Year"):
        train_df = df[df['year'] < year_to_predict]
        test_df = df[df['year'] == year_to_predict]
        if test_df.empty or train_df.empty: continue

        # --- Train Ensemble on Diverse Futures ---
        ensemble_models = {'lower': [], 'median': [], 'upper': []}
        for i in range(N_ENSEMBLE_MODELS):
            # Each bootstrap sample now contains a random mix of districts, years, AND SEAS5 members
            bootstrap_train_df = train_df.sample(frac=1.0, replace=True, random_state=i)
            X_train_boot = bootstrap_train_df[feature_cols]
            y_train_boot = bootstrap_train_df['kreisYield_detrended']
            for name, alpha in [('lower', 0.025), ('median', 0.5), ('upper', 0.975)]:
                model = clone(BASE_MODEL)
                model.set_params(objective='reg:quantileerror', quantile_alpha=alpha, random_state=i)
                model.fit(X_train_boot, y_train_boot)
                ensemble_models[name].append(model)

        # --- Predict on Test Data ---
        # For each district, we now have N_MEMBERS (e.g., 25) possible futures
        # We run our 10 models over all N_MEMBER futures to get a full distribution
        X_test = test_df[feature_cols]

        test_preds = {name: [] for name in ['lower', 'median', 'upper']}
        for name, models in ensemble_models.items():
            for model in models:
                test_preds[name].append(model.predict(X_test))

        # This gives us a (10 models x N_test_rows) array for each quantile

        # Aggregate the results per district
        # First, add predictions back to the test dataframe
        temp_results = test_df[['district_no', 'year', 'kreisYield', 'yield_trend']].copy()

        # For each district-year, we now have a distribution of predictions
        # (10 XGBoost models * N SEAS5 members)
        # Let's take the mean over all these possibilities
        temp_results['pred_lower'] = np.mean(test_preds['lower'], axis=0)
        temp_results['pred_median'] = np.mean(test_preds['median'], axis=0)
        temp_results['pred_upper'] = np.mean(test_preds['upper'], axis=0)

        # To calculate epistemic uncertainty, we first average over the SEAS5 members for each model
        # This gives us 10 different "consensus" median predictions for each district
        medians_by_model = [
            pd.DataFrame({'district_no': test_df['district_no'], 'pred': p}).groupby('district_no')['pred'].mean() for p
            in test_preds['median']]
        epistemic_df = pd.concat(medians_by_model, axis=1).std(axis=1).reset_index(name='epistemic_uncertainty')

        # Now, average the predictions for each district to get the final numbers
        final_fold_results = temp_results.groupby(['district_no', 'year', 'kreisYield']).mean().reset_index()
        final_fold_results = pd.merge(final_fold_results, epistemic_df, on='district_no')

        # Retrend the results
        final_fold_results['predicted_yield_lower'] = final_fold_results['pred_lower'] + final_fold_results[
            'yield_trend']
        final_fold_results['predicted_yield_median'] = final_fold_results['pred_median'] + final_fold_results[
            'yield_trend']
        final_fold_results['predicted_yield_upper'] = final_fold_results['pred_upper'] + final_fold_results[
            'yield_trend']

        all_predictions.append(final_fold_results)

    return pd.concat(all_predictions, ignore_index=True)


def main():
    """Main function to orchestrate the new ensemble backtest."""
    full_df = load_and_merge_data()
    if full_df is None: return

    # Dynamically determine feature columns from the final dataframe
    # Exclude identifiers, targets, and metadata
    excluded_cols = ['district_no', 'year', 'seas5_member', 'kreisYield', 'yield_trend', 'kreisYield_detrended', 'name']
    feature_cols = [col for col in full_df.columns if col not in excluded_cols]
    print(f"\n-> Using {len(feature_cols)} features for training.")

    results = run_source_uncertainty_backtest(full_df, feature_cols)

    # --- Final Analysis ---
    # Since this is a raw ensemble, we still need to run a final analysis script
    # This script's primary job is to produce the results CSV
    os.makedirs(REPORT_DIR, exist_ok=True)
    results_path = os.path.join(REPORT_DIR, 'source_uncertainty_ensemble_results.csv')
    results.to_csv(results_path, index=False, float_format='%.6f')
    print(f"\n--- Backtest Complete ---")
    print(f"Detailed results saved to: {results_path}")
    print("Run the analysis script to generate plots and final metrics.")


if __name__ == "__main__":
    main()