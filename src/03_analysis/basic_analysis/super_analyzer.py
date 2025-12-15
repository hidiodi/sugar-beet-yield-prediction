import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- DYNAMIC CONFIGURATION (Fixed Path Mismatch) ---
# We use the same config dictionary that the data prep scripts used
MODEL_CONFIG = config.MODEL_COMPARISON_CONFIG
INPUT_DIR = Path(MODEL_CONFIG['OUTPUT_DIR'])

TRAINING_DATA_PATH = INPUT_DIR / 'super_ensemble_training_data.csv'
FORECAST_PATH = INPUT_DIR / 'super_ensemble_final_forecast_TSCV.csv'
OUTPUT_REPORT_DIR = config.BASE_DIR / 'reports/super_analysis'
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Merges the Ground Truth Potential (Training Data) with Actual Decisions (Forecast)."""
    if not TRAINING_DATA_PATH.exists():
        logging.error(f"Missing Training Data: {TRAINING_DATA_PATH}")
        return None
    if not FORECAST_PATH.exists():
        logging.error(f"Missing Forecast Data: {FORECAST_PATH}")
        return None

    # Load Ground Truth (The "Oracle")
    df_truth = pd.read_csv(TRAINING_DATA_PATH)

    # Load Actual Decisions (The "Meta-Learner")
    df_pred = pd.read_csv(FORECAST_PATH)

    # Merge on keys
    # Note: df_pred has the 'Super_Ensemble_pred', df_truth has 'Oracle_Error' and 'Best_Model'
    # We need to be careful not to duplicate columns if they exist in both
    cols_to_use = ['year', 'district_no', 'Super_Ensemble_pred', 'Predicted_Best_Model']
    if 'Prob_Robust_Linear' in df_pred.columns:  # Add probabilities if available
        prob_cols = [c for c in df_pred.columns if c.startswith('Prob_')]
        cols_to_use.extend(prob_cols)

    df = pd.merge(df_truth, df_pred[cols_to_use], on=['year', 'district_no'], how='inner')

    return df


def analyze_regret(df):
    """
    Regret = Error of Selected Model - Error of Best Possible Model (Oracle).
    """
    logging.info("\n" + "=" * 80)
    logging.info(" 1. REGRET ANALYSIS (Selector Performance)")
    logging.info("=" * 80)

    # Calculate Actual Error
    df['Actual_Error'] = (df['kreisYield'] - df['Super_Ensemble_pred']).abs()

    # Calculate Regret (Actual - Ideal)
    # Ensure Oracle Error is non-negative
    df['Regret'] = (df['Actual_Error'] - df['Oracle_Error']).clip(lower=0)

    total_regret = df['Regret'].mean()
    oracle_mae = df['Oracle_Error'].mean()
    actual_mae = df['Actual_Error'].mean()

    logging.info(f"Current Ensemble MAE: {actual_mae:.2f}")
    logging.info(f"Oracle (Perfect) MAE: {oracle_mae:.2f}")
    logging.info(f"Average Regret:       {total_regret:.2f} (Loss due to imperfect selection)")

    # Conclusion
    share_of_loss = 0
    if actual_mae > 0:
        share_of_loss = (total_regret / actual_mae) * 100

    logging.info(f"Insight: {share_of_loss:.1f}% of the error comes from picking the wrong model.")

    if share_of_loss > 20:
        logging.info(">> RECOMMENDATION: Focus on improving the META-LEARNER (Classifier).")
    else:
        logging.info(">> RECOMMENDATION: Focus on improving the BASE MODELS (Physics/Signals).")

    return df


def analyze_systemic_failure(df):
    """
    Where does even the Oracle fail?
    """
    logging.info("\n" + "=" * 80)
    logging.info(" 2. SYSTEMIC FAILURE ANALYSIS (Blind Spots)")
    logging.info("=" * 80)

    # Define "Failure" as Oracle Error > 50 dt/ha
    hard_rows = df[df['Oracle_Error'] > 50].copy()

    logging.info(f"Number of 'Hard' predictions: {len(hard_rows)} ({len(hard_rows) / len(df):.1%})")

    # 1. Temporal Analysis
    worst_years = hard_rows.groupby('year')['Oracle_Error'].mean().sort_values(ascending=False).head(5)
    logging.info("\nTop 5 Years where ALL models fail:")
    logging.info(worst_years)

    # 2. Regional Analysis (East vs West)
    df['Region'] = df['district_no'].astype(str).str.zfill(5).str[:2].astype(int).apply(
        lambda x: 'East (GDR)' if x >= 11 else 'West'
    )

    region_perf = df.groupby('Region')['Oracle_Error'].mean()
    logging.info("\nUnsolvable Error by Region:")
    logging.info(region_perf)

    return hard_rows


def analyze_component_specialization(df):
    """
    When is each model the winner?
    """
    logging.info("\n" + "=" * 80)
    logging.info(" 3. COMPONENT SPECIALIZATION (Who wins when?)")
    logging.info("=" * 80)

    # Best Model Distribution
    dist = df['Best_Model'].value_counts(normalize=True) * 100
    logging.info("True 'Best Model' Frequency (The Oracle's Choice):")
    logging.info(dist)

    logging.info("\nWinning Conditions (Mean Deviation from Trend when winning):")
    for model in dist.index:
        signal_col = f'Signal_{model}'
        if signal_col in df.columns:
            wins = df[df['Best_Model'] == model]
            avg_signal = wins[signal_col].mean()
            logging.info(f"  {model:<20}: {avg_signal:+.2f} dt/ha vs Trend")


def generate_action_plan(df):
    logging.info("\n" + "=" * 80)
    logging.info(" 4. FINAL STRATEGIC VERDICT")
    logging.info("=" * 80)

    actual_mae = df['Actual_Error'].mean()
    oracle_mae = df['Oracle_Error'].mean()

    print(f"1. OPTIMIZATION SPACE: We can gain {actual_mae - oracle_mae:.2f} dt/ha just by fixing the Classifier.")

    hard_years = df.groupby('year')['Oracle_Error'].mean()
    # Safe check for keys
    is_2003_bad = hard_years.get(2003, 0) > 60
    is_2018_bad = hard_years.get(2018, 0) > 60

    print("\n2. MODEL GAPS:")
    if is_2003_bad:
        print(f"   -> 2003 is still unsolved (Oracle Error: {hard_years.get(2003):.1f}). Needs Heat Physics.")
    else:
        print("   -> 2003 is Solved/Manageable.")

    if is_2018_bad:
        print(f"   -> 2018 is still unsolved (Oracle Error: {hard_years.get(2018):.1f}). Needs Drought Physics.")
    else:
        print("   -> 2018 is Solved/Manageable.")


def main():
    df = load_data()
    if df is not None:
        df = analyze_regret(df)
        analyze_systemic_failure(df)
        analyze_component_specialization(df)
        generate_action_plan(df)


if __name__ == "__main__":
    main()