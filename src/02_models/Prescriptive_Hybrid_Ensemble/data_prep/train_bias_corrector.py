# File: src/models/train_bias_corrector.py
# v4: Adds extensive diagnostic logging to the data assembly function to pinpoint
#     exactly where data is being lost during merges or cleaning.

import pandas as pd
from pathlib import Path
import numpy as np
import logging
import sys
import os
import joblib
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# (CONFIG remains the same as v2)
CONFIG = {
    'YEARS': {'START': 1981, 'END': 2023},
    'FILE_PATHS': {
        'STATIC_FEATURES': Path('data/05_model_input/stage1_preseason_features.csv'),
        'HISTORICAL_YIELDS': Path('data/02_intermediate/sugarbeet_yield.csv'),
        'ESTIMATED_MANAGEMENT': Path('data/02_intermediate/historical_management_sensitivity.csv'),
        'RAW_SIMULATIONS_DIR': Path('data/03_processed/08_phe_output/raw_simulations'),
        'DAILY_WEATHER_DIR': Path('data/02_intermediate/daily_weather'),
        'OUTPUT_MODELS_DIR': Path('models/bias_correctors'),
        'OUTPUT_PREDICTIONS_CSV': Path('data/08_phe_output/historical_validation_predictions.csv'),
    },
    'MODEL_PARAMS': {
        'GDD_THRESHOLD_FOR_TRAINING': 175,
        'HISTORICAL_FERT_MULTIPLIER': 1.0,
        'XGB_PARAMS': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8,
                       'colsample_bytree': 0.8, 'random_state': 42}
    }
}

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')


def assemble_ground_truth_data():
    """
    Assembles the master training dataset with added diagnostics.
    """
    logging.info("--- Assembling Ground Truth Training Data ---")

    # 1. Load base data
    df_static = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_FEATURES'])
    df_yield = pd.read_csv(CONFIG['FILE_PATHS']['HISTORICAL_YIELDS'])
    df_static['district_no'] = df_static['district_no'].astype(str).str.zfill(5)
    df_yield['district_no'] = df_yield['district_no'].astype(str).str.zfill(5)
    df_base = pd.merge(df_static, df_yield, on=['year', 'district_no'])
    logging.info(f"Step 1: Loaded {len(df_base)} historical records with static features.")

    # 2. Load and merge estimated management
    df_mgmt = pd.read_csv(CONFIG['FILE_PATHS']['ESTIMATED_MANAGEMENT'])
    df_mgmt = df_mgmt[df_mgmt['gdd_threshold'] == CONFIG['MODEL_PARAMS']['GDD_THRESHOLD_FOR_TRAINING']]
    df_mgmt['district_no'] = df_mgmt['district_no'].astype(str).str.zfill(5)
    df_historical = pd.merge(df_base, df_mgmt[['year', 'district_no', 'est_planting_date']], on=['year', 'district_no'],
                             how='left')
    df_historical.dropna(subset=['est_planting_date'], inplace=True)
    df_historical['est_planting_date'] = pd.to_datetime(df_historical['est_planting_date'])
    logging.info(f"Step 2: Merged with estimated management. Shape: {df_historical.shape}")

    # 3. Load ALL relevant simulation results
    all_sim_results = []
    sim_dir = CONFIG['FILE_PATHS']['RAW_SIMULATIONS_DIR']
    hist_fert = CONFIG['MODEL_PARAMS']['HISTORICAL_FERT_MULTIPLIER']
    for year in tqdm(range(CONFIG['YEARS']['START'], CONFIG['YEARS']['END'] + 1), desc="Loading simulation data"):
        sim_file = sim_dir / f"{year}_raw.csv"
        if not sim_file.exists(): continue
        df_sim_year = pd.read_csv(sim_file)
        df_sim_year = df_sim_year[df_sim_year['fertilizer_multiplier'] == hist_fert]
        all_sim_results.append(df_sim_year)

    df_all_sims = pd.concat(all_sim_results, ignore_index=True)
    df_all_sims['district_no'] = df_all_sims['district_no'].astype(str).str.zfill(5)
    df_all_sims['planting_date'] = pd.to_datetime(df_all_sims['planting_date'])
    logging.info(f"Step 3: Loaded {len(df_all_sims)} relevant simulation results.")

    # 4. Perform a nearest-date match
    logging.info("Step 4: Finding best-matching simulation for each historical estimate...")
    df_merged = pd.merge(df_historical, df_all_sims, on=['year', 'district_no'], how='left')
    logging.info(f"  - After initial left merge, shape is: {df_merged.shape}")

    # Check how many historical records found no simulation data at all
    null_sim_count = df_merged['planting_date'].isnull().sum()
    if null_sim_count > 0:
        logging.warning(
            f"  - WARNING: {null_sim_count} historical records had no matching simulation data for their district-year and were dropped.")
        df_merged.dropna(subset=['planting_date'], inplace=True)

    df_merged['date_diff'] = (df_merged['planting_date'] - df_merged['est_planting_date']).abs()
    idx = df_merged.groupby(['year', 'district_no', 'est_planting_date'])['date_diff'].idxmin()
    df_train = df_merged.loc[idx].drop(columns=['date_diff'])
    logging.info(f"  - After finding nearest date, shape is: {df_train.shape}")

    # 5. Engineer Pre-Season Weather Features
    weather_features = []
    weather_dir = CONFIG['FILE_PATHS']['DAILY_WEATHER_DIR']
    for year in tqdm(df_train['year'].unique(), desc="Engineering weather features"):
        # (Optimized logic is the same)
        # ...
        weather_file = weather_dir / f"historical_daily_weather_era5_{year}.csv"
        if not weather_file.exists(): continue
        df_weather_year = pd.read_csv(weather_file, parse_dates=['date'])
        df_weather_year['district_no'] = df_weather_year['district_no'].astype(str).str.zfill(5)
        preseason_mask = (df_weather_year['date'].dt.month >= 1) & (df_weather_year['date'].dt.month <= 3)
        df_preseason = df_weather_year[preseason_mask].copy()
        df_preseason['tavg'] = (df_preseason['tmin'] + df_preseason['tmax']) / 2.0
        district_agg = df_preseason.groupby('district_no').agg(preseason_precip_sum=('precip', 'sum'),
                                                               preseason_tavg_mean=('tavg', 'mean'),
                                                               preseason_srad_sum=('srad', 'sum')).reset_index()
        district_agg['year'] = year
        weather_features.append(district_agg)

    df_weather_feats = pd.concat(weather_features, ignore_index=True)
    df_train = pd.merge(df_train, df_weather_feats, on=['year', 'district_no'], how='left')
    logging.info(f"Step 5: After merging weather features, shape is: {df_train.shape}")

    # 6. Define target and clean up
    df_train['simulated_yield_fw_dtha'] = (df_train['simulated_yield_twso'] * 0.01) / 0.25

    # Diagnostic check BEFORE dropping NaNs
    logging.info("Step 6: Final cleaning. Checking for NaNs before dropping:")
    logging.info(f"  - NaNs in 'simulated_yield_fw_dtha': {df_train['simulated_yield_fw_dtha'].isnull().sum()}")
    logging.info(f"  - NaNs in 'yield': {df_train['yield'].isnull().sum()}")
    logging.info(f"  - NaNs in 'preseason_tavg_mean': {df_train['preseason_tavg_mean'].isnull().sum()}")

    df_train.dropna(subset=['simulated_yield_fw_dtha', 'yield', 'preseason_tavg_mean'], inplace=True)
    logging.info(f"  - After dropping NaNs, final shape is: {df_train.shape}")

    if len(df_train) == 0:
        logging.error("CRITICAL: The final DataFrame is empty. Halting before training.")
        return pd.DataFrame()  # Return empty frame to avoid crash

    df_train['target_error'] = df_train['yield'] - df_train['simulated_yield_fw_dtha']
    df_train['planting_day_of_year'] = df_train['est_planting_date'].dt.dayofyear

    logging.info(f"Successfully assembled training data with {len(df_train)} district-year instances.")
    return df_train


def main():
    logging.info("--- Starting Phase 1, Task 1.2: Train Bias Corrector ---")

    CONFIG['FILE_PATHS']['OUTPUT_MODELS_DIR'].mkdir(exist_ok=True, parents=True)
    CONFIG['FILE_PATHS']['OUTPUT_PREDICTIONS_CSV'].parent.mkdir(exist_ok=True, parents=True)

    df_master = assemble_ground_truth_data()

    if df_master.empty:
        logging.error("Main training DataFrame is empty. Cannot proceed with training.")
        sys.exit(1)

    # (The rest of the main function is the same...)
    all_years = sorted(df_master['year'].unique())
    out_of_sample_predictions = []
    features_to_use = [
        'avg_sand_0_100cm', 'avg_clay_0_100cm', 'avg_som_0_100cm', 'avg_bdod_0_100cm',
        'avg_elevation', 'latitude', 'longitude', 'planting_day_of_year',
        'preseason_precip_sum', 'preseason_tavg_mean', 'preseason_srad_sum'
    ]
    for year_to_exclude in tqdm(all_years, desc="LOYO Cross-Validation"):
        train_df = df_master[df_master['year'] != year_to_exclude]
        test_df = df_master[df_master['year'] == year_to_exclude]
        X_train = train_df[features_to_use]
        y_train = train_df['target_error']
        X_test = test_df[features_to_use]
        model = xgb.XGBRegressor(**CONFIG['MODEL_PARAMS']['XGB_PARAMS'])
        model.fit(X_train, y_train)
        model_path = CONFIG['FILE_PATHS']['OUTPUT_MODELS_DIR'] / f"bias_corrector_{year_to_exclude}.joblib"
        joblib.dump(model, model_path)
        predictions = model.predict(X_test)
        test_predictions_df = test_df[
            ['year', 'district_no', 'yield', 'simulated_yield_fw_dtha', 'target_error']].copy()
        test_predictions_df['predicted_error'] = predictions
        out_of_sample_predictions.append(test_predictions_df)

    df_final_preds = pd.concat(out_of_sample_predictions, ignore_index=True)
    output_path = CONFIG['FILE_PATHS']['OUTPUT_PREDICTIONS_CSV']
    df_final_preds.to_csv(output_path, index=False)

    logging.info(f"✓ All models trained. Out-of-sample predictions saved to {output_path.name}")
    rmse = np.sqrt(mean_squared_error(df_final_preds['target_error'], df_final_preds['predicted_error']))
    logging.info(f"Overall RMSE of Bias Corrector on out-of-sample predictions: {rmse:.4f}")

    logging.info("\n" + "=" * 70 + "\n✓ BIAS CORRECTOR TRAINING COMPLETED SUCCESSFULLY!\n" + "=" * 70)


if __name__ == "__main__":
    main()