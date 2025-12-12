#import pandas as pd
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error
import sys
import numpy as np

# --- Project Setup (Assumes 'config' and path structure is correct) ---
# NOTE: Adjust project_root path if necessary for your environment
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config

# Ensure logging is configured for clear output
logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])
OUTPUT_FILENAME = 'super_ensemble_training_data.csv'

# Additional File Paths (using config constants where available, fallbacks otherwise)
# NOTE: These paths must match the file system structure used in the previous scripts.
NATIVE_ENSEMBLE_PATH = project_root / 'src/models/native_ensemble_champion/native_ensemble_forecasts.csv'
STATISTICAL_TREND_PATH = Path(CONFIG['STATISTICAL_TREND_FILE'])
HYBRID_XGB_PATH = Path(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
V31_SOLAR_PATH = config.DATA_DIR / '06_model_output' / 'v31_solar_gated_forecast.csv'

# --- Configuration for Analysis ---
ANALYSIS_START_YEAR = 2000
ANALYSIS_END_YEAR = 2024

# The four component models for the Super-Ensemble
COMPONENT_MODELS = {
    'Statistical Trend': STATISTICAL_TREND_PATH,
    'V31 Solar Gated': V31_SOLAR_PATH,
    'Native Ensemble': NATIVE_ENSEMBLE_PATH,
    'Hybrid XGB': HYBRID_XGB_PATH
}

# Mapping of file columns to our unified feature columns
MODEL_COLUMN_MAP = {
    'Statistical Trend': 'final_corrected_forecast',
    'V31 Solar Gated': 'final_pred',
    'Native Ensemble': 'Ensemble_Pred',
    'Hybrid XGB': 'predicted_yield_median'
}

def load_and_merge_components():
    """Loads and merges the four core component model predictions."""
    logging.info("--- Loading and Merging Super-Ensemble Components ---")

    # Start with the Hybrid XGB file to get the base district/year structure and true yield (kreisYield)
    df = pd.read_csv(HYBRID_XGB_PATH)[['year', 'district_no', 'kreisYield']]
    df['district_no'] = df['district_no'].astype(int)

    for model_name, path in COMPONENT_MODELS.items():
        p = Path(path)
        if p.exists():
            df_m = pd.read_csv(p)

            # Ensure district_no is int and get the prediction column name
            if 'district_no' in df_m.columns:
                df_m['district_no'] = df_m['district_no'].astype(int)

            # Get the prediction column
            pred_col = MODEL_COLUMN_MAP[model_name]

            cols_to_keep = ['year', 'district_no', pred_col]

            if pred_col in df_m.columns:
                df_m = df_m[cols_to_keep]

                # Rename the prediction column to a standard format (e.g., 'Trend_pred')
                target_col = f'{model_name.replace(" ", "_")}_pred'
                df_m.rename(columns={pred_col: target_col}, inplace=True)

                df = pd.merge(df, df_m, on=['year', 'district_no'], how='left')
                logging.info(f"✓ Merged {model_name} as {target_col}")
            else:
                 logging.warning(f"Skipping {model_name}: Missing required column '{pred_col}' in {p.name}")
        else:
            logging.warning(f"File not found for {model_name} at {p}")

    # Final cleanup: Filter for the analysis period and drop rows with any missing prediction
    df_clean = df[(df['year'] >= ANALYSIS_START_YEAR) & (df['year'] <= ANALYSIS_END_YEAR)].copy()

    pred_cols = [f'{m.replace(" ", "_")}_pred' for m in COMPONENT_MODELS.keys()]
    df_clean.dropna(subset=pred_cols + ['kreisYield'], inplace=True)

    logging.info(f"Data ready: {len(df_clean)} complete records for {ANALYSIS_START_YEAR}-{ANALYSIS_END_YEAR}.")
    return df_clean

def create_super_ensemble_features(df):
    """
    Creates the target variable (Best Model) and the predictor features (Differences/Residuals).
    """
    logging.info("\n--- Creating Super-Ensemble Training Features ---")

    df_new = df.copy()
    y_true = df_new['kreisYield']

    pred_cols = [f'{m.replace(" ", "_")}_pred' for m in COMPONENT_MODELS.keys()]

    # ----------------------------------------------------
    # PART A: CALCULATE ERRORS AND NEW PERFECT ORACLE
    # ----------------------------------------------------

    error_cols = []
    for col in pred_cols:
        error_col = col.replace('_pred', '_error')
        df_new[error_col] = (df_new[col] - y_true).abs()
        error_cols.append(error_col)

    # Calculate the New Perfect Oracle
    df_new['Min_Error'] = df_new[error_cols].min(axis=1)

    new_oracle_mae = df_new['Min_Error'].mean()
    logging.info(f"** NEW PERFECT ORACLE MAE: {new_oracle_mae:.4f} dt/ha **")
    logging.info(f"This is the theoretical best performance we can achieve.")

    # ----------------------------------------------------
    # PART B: CREATE CLASSIFICATION TARGET (BEST MODEL)
    # ----------------------------------------------------

    # Get the name of the column that contains the minimum error for each row
    df_new['Best_Model_Error_Col'] = df_new[error_cols].idxmin(axis=1)

    # Map the error column name back to the model name (e.g., 'Trend_error' -> 'Statistical Trend')
    df_new['Best_Model'] = df_new['Best_Model_Error_Col'].str.replace('_error', '')

    # Analyze the distribution of the target variable
    logging.info("\nDistribution of the Optimal Component Model (Target Variable):")
    logging.info(df_new['Best_Model'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

    # ----------------------------------------------------
    # PART C: CREATE PREDICTOR FEATURES (DIFFERENCES & RESIDUALS)
    # ----------------------------------------------------

    # 1. Prediction Differences (Meta-Learner needs to know how far apart the components are)
    features = []
    for i in range(len(pred_cols)):
        for j in range(i + 1, len(pred_cols)):
            col_i = pred_cols[i]
            col_j = pred_cols[j]
            feature_name = f'Diff_{col_i.split("_")[0]}_vs_{col_j.split("_")[0]}'
            df_new[feature_name] = df_new[col_i] - df_new[col_j]
            features.append(feature_name)

    # 2. Residuals (How far is each prediction from the simplest average)
    df_new['Mean_Pred'] = df_new[pred_cols].mean(axis=1)
    for col in pred_cols:
        feature_name = f'Residual_from_Mean_{col.split("_")[0]}'
        df_new[feature_name] = df_new[col] - df_new['Mean_Pred']
        features.append(feature_name)

    logging.info(f"\nGenerated {len(features)} new predictor features.")

    # Select final columns: IDs, Target, Predictions, and Features
    final_cols = ['year', 'district_no', 'kreisYield', 'Best_Model'] + pred_cols + features

    # Save the prepared data for the next step (Meta-Learner training)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    df_new[final_cols].to_csv(output_path, index=False)
    logging.info(f"\n--- Data preparation complete. Saved to: {output_path} ---")

    return df_new[final_cols]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and Clean Data
    df = load_and_merge_components()
    if df.empty:
        return

    # Step 2: Create Features and Target
    create_super_ensemble_features(df)

if __name__ == '__main__':
    main()