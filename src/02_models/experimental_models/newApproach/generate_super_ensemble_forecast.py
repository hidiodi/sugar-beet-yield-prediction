import pandas as pd
import logging
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score
import json
import sys
import numpy as np

# --- Project Setup (Re-import statements and setup remain the same) ---
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])
INPUT_FILENAME = 'super_ensemble_training_data.csv'
MODEL_FILENAME = 'super_ensemble_meta_learner_TSCV.json'
LABEL_MAP_FILENAME = 'meta_learner_label_map_TSCV.json'
FINAL_FORECAST_FILENAME = 'super_ensemble_final_forecast_TSCV.csv'


# Renamed to accept a dataframe and a title for reuse
def analyze_model_performance(df_analysis, models_to_analyze, title):
    """Calculates and logs MAE and R2 for a list of models on a given dataframe."""
    results = []
    logging.info("\n" + "=" * 90)
    logging.info(f"       {title}")
    logging.info("=" * 90)

    # 1. Calculate metrics using the column names with underscores (e.g., 'Super_Ensemble_pred')
    for m in models_to_analyze:
        # We need the column name from the dataframe (e.g., 'Super_Ensemble_pred')
        col = f'{m}_pred'

        # We use a clean name for the report (e.g., 'Super Ensemble')
        report_name = m.replace('_', ' ')

        # We must rename columns in the analysis dataframe temporarily for the report formatting
        if col in df_analysis.columns:
            clean = df_analysis.dropna(subset=[col, 'kreisYield'])
            if clean.empty: continue

            # Recalculate MAE and R2 on the *filtered* data
            mae = mean_absolute_error(clean['kreisYield'], clean[col])
            r2 = r2_score(clean['kreisYield'], clean[col])
            results.append({'Model': report_name, 'MAE': mae, 'R2': r2})
        else:
            logging.warning(f"Skipping model {report_name}: Column {col} not found.")

    res_df = pd.DataFrame(results).sort_values('MAE')
    logging.info(res_df.to_string(index=False, float_format="%.4f"))
    return res_df


def generate_forecast():
    """
    Generates the final Super-Ensemble forecast and performs time-frame comparisons.
    """
    input_path = OUTPUT_DIR / INPUT_FILENAME
    model_path = OUTPUT_DIR / MODEL_FILENAME
    label_map_path = OUTPUT_DIR / LABEL_MAP_FILENAME

    # Error checking for file existence... (omitted for brevity)

    logging.info("--- Loading Data and Meta-Learner ---")
    df = pd.read_csv(input_path)

    # 1. Load Model and Generate Predictions (Same logic as before)
    # ... (Loading model, creating feature set X_full, and getting y_pred_model_name)
    model = XGBClassifier()
    model.load_model(model_path)

    with open(label_map_path, 'r') as f:
        original_label_map = json.load(f)

    label_map = {v: k for k, v in original_label_map.items()}

    EXCLUDE_COLS = ['year', 'district_no', 'kreisYield', 'Best_Model']
    PRED_COLS = [col for col in df.columns if col.endswith('_pred')]
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS + PRED_COLS]
    X_full = df[feature_cols]

    logging.info("Predicting optimal component model using Meta-Learner...")
    y_pred_encoded = model.predict(X_full)
    y_pred_model_name = pd.Series(y_pred_encoded).map(label_map).values
    df['Predicted_Best_Model'] = y_pred_model_name

    # 2. Generate the Final Super-Ensemble Prediction
    logging.info("Generating Super-Ensemble Prediction via Hard Switch...")
    df['Prediction_Col_Name'] = df['Predicted_Best_Model'].apply(lambda x: f'{x}_pred')
    conditions = [df['Prediction_Col_Name'] == col for col in PRED_COLS]
    choices = [df[col] for col in PRED_COLS]
    df['Super_Ensemble_pred'] = np.select(conditions, choices, default=np.nan)

    # 3. Final Performance Analysis (Combined)

    # List of all models for final report (using underscores for column lookup)
    all_models_for_report = [
        'Super_Ensemble', 'Statistical_Trend', 'V31_Solar_Gated',
        'Native_Ensemble', 'Hybrid_XGB'
    ]

    # --- ANALYSIS 1: FULL PERIOD (2000-2024) ---
    df_2000_2024 = df.copy()  # Use the full dataframe

    # Re-calculate Oracle MAE for the final report section
    error_cols = [col.replace('_pred', '_error') for col in PRED_COLS]
    for i, col in enumerate(PRED_COLS):
        df_2000_2024[error_cols[i]] = (df_2000_2024[col] - df_2000_2024['kreisYield']).abs()
    df_2000_2024['Min_Error'] = df_2000_2024[error_cols].min(axis=1)
    new_oracle_mae = df_2000_2024['Min_Error'].mean()
    super_ensemble_mae = mean_absolute_error(df_2000_2024['kreisYield'], df_2000_2024['Super_Ensemble_pred'])

    logging.info("\n--- SUPER-ENSEMBLE PERFORMANCE COMPARISON (2000-2024) ---")
    logging.info("\n" + "#" * 90)
    logging.info(f"   IDEAL (ORACLE) MAE: {new_oracle_mae:.4f} dt/ha")
    logging.info(f"   ACTUAL SUPER-ENSEMBLE MAE: {super_ensemble_mae:.4f} dt/ha")
    logging.info(f"   ERROR CLOSED: {super_ensemble_mae - new_oracle_mae:+.4f} dt/ha above Oracle")
    logging.info("#" * 90)

    analyze_model_performance(df_2000_2024, all_models_for_report, title="MODEL PERFORMANCE (2000-2024)")

    # --- ANALYSIS 2: HIGH-VOLATILITY PERIOD (2014-2024) ---
    logging.info("\n" + "=" * 90)
    logging.info("       FOCUSED ANALYSIS: HIGH-VOLATILITY PERIOD (2014-2024)")
    logging.info("=" * 90)

    df_2014_2024 = df[df['year'] >= 2014].copy()

    if df_2014_2024.empty:
        logging.warning("No data available for 2014-2024 analysis.")
    else:
        # Re-calculate Oracle MAE for the 2014-2024 subset
        for i, col in enumerate(PRED_COLS):
            df_2014_2024[error_cols[i]] = (df_2014_2024[col] - df_2014_2024['kreisYield']).abs()
        df_2014_2024['Min_Error'] = df_2014_2024[error_cols].min(axis=1)

        oracle_mae_14_24 = df_2014_2024['Min_Error'].mean()
        super_ensemble_mae_14_24 = mean_absolute_error(df_2014_2024['kreisYield'], df_2014_2024['Super_Ensemble_pred'])

        logging.info("\n" + "#" * 90)
        logging.info(f"   IDEAL (ORACLE) MAE (2014-2024): {oracle_mae_14_24:.4f} dt/ha")
        logging.info(f"   ACTUAL SUPER-ENSEMBLE MAE (2014-2024): {super_ensemble_mae_14_24:.4f} dt/ha")
        logging.info(f"   ERROR CLOSED: {super_ensemble_mae_14_24 - oracle_mae_14_24:+.4f} dt/ha above Oracle")
        logging.info("#" * 90)

        analyze_model_performance(df_2014_2024, all_models_for_report, title="MODEL PERFORMANCE (2014-2024)")

    # 4. Save Final Forecast
    output_path = OUTPUT_DIR / FINAL_FORECAST_FILENAME
    # Use the full dataframe for saving
    df[['year', 'district_no', 'kreisYield', 'Super_Ensemble_pred', 'Predicted_Best_Model']].to_csv(output_path,
                                                                                                    index=False)
    logging.info(f"\n--- Final Super-Ensemble Forecast saved to: {output_path} ---")


if __name__ == '__main__':
    generate_forecast()