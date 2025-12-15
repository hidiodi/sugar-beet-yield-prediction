import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import numpy as np

# --- Project Setup (Assumes 'config' and path structure is correct) ---
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config

# Ensure logging is configured for clear output
logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])

# Additional File Paths from the original script
NATIVE_COMPARISON_PATH = project_root / 'src/models/native_physics_comparison/native_model_comparison_v2_v8.csv'
NATIVE_ENSEMBLE_PATH = project_root / 'src/models/native_ensemble_champion/native_ensemble_forecasts.csv'
SWITCHED_MODEL_PATH = config.DATA_DIR / '06_model_output' / 'final_switched_forecast.csv'
V31_SOLAR_PATH = config.DATA_DIR / '06_model_output' / 'v31_solar_gated_forecast.csv'

# --- Configuration for Analysis ---
# Set the time period for the analysis. Change this to [2014] to isolate the high-volatility years.
ANALYSIS_START_YEAR = 2000  # Change this to 2014 to replicate the high-volatility analysis
ANALYSIS_END_YEAR = 2024


# --- Utility Functions from Original Script (Modified/Included for self-containment) ---

def load_and_merge_models():
    """Loads and merges all relevant model predictions."""
    logging.info("--- Loading and Merging Model Data ---")

    # 1. Base: Hybrid Model (Used to get the base district/year structure)
    base_path = Path(CONFIG['HYBRID_XGB_PREDICTIONS_FILE'])
    if not base_path.exists():
        logging.error("Hybrid XGB predictions not found. Cannot proceed.")
        return pd.DataFrame()

    df = pd.read_csv(base_path)[['year', 'district_no', 'kreisYield']]

    # Define all model paths and columns
    models_to_merge = [
        (CONFIG['HYBRID_XGB_PREDICTIONS_FILE'], ['predicted_yield_median'], ['Hybrid XGB_pred']),
        (CONFIG['STANDALONE_XGB_PREDICTIONS_FILE'], ['predicted_yield_median'], ['Standalone XGB_pred']),
        (CONFIG['STATISTICAL_TREND_FILE'], ['final_corrected_forecast'], ['Statistical Trend_pred']),
        (NATIVE_COMPARISON_PATH, ['Native_V2', 'Native_V8'], ['Native V2_pred', 'Native V8_pred']),
        (NATIVE_ENSEMBLE_PATH, ['Ensemble_Pred'], ['Native Ensemble_pred']),
        (SWITCHED_MODEL_PATH, ['Final_Pred', 'Strategy_Mode'], ['Regime Switch V10_pred', 'Strategy_Mode']),
        (V31_SOLAR_PATH, ['final_pred'], ['V31 Solar Gated_pred']),
    ]

    for path, source_cols, target_cols in models_to_merge:
        p = Path(path)
        if p.exists():
            df_m = pd.read_csv(p)
            # Ensure district_no is int for merging
            if 'district_no' in df_m.columns:
                df_m['district_no'] = df_m['district_no'].astype(int)

            cols_to_keep = ['year', 'district_no'] + source_cols
            # Check if all required source columns exist before merging
            if all(col in df_m.columns for col in source_cols):
                df_m = df_m[cols_to_keep]
                merge_df = pd.merge(df, df_m, on=['year', 'district_no'], how='left')

                # Rename the columns after merge
                rename_map = dict(zip(source_cols, target_cols))
                df = merge_df.rename(columns=rename_map)
                logging.info(f"✓ Merged model predictions from: {p.name}")
            else:
                logging.warning(f"Skipping merge from {p.name}: Missing required columns {source_cols}")
        else:
            logging.warning(f"File not found: {p.name}")

    return df


# --- Core Analysis Functions ---

def analyze_model_performance(df, models_to_analyze, title="OVERALL PERFORMANCE"):
    """Calculates and logs MAE and R2 for a list of models."""
    results = []
    logging.info("\n" + "=" * 80)
    logging.info(f"      {title}")
    logging.info("=" * 80)

    for m in models_to_analyze:
        col = f'{m}_pred'
        if col in df.columns:
            clean = df.dropna(subset=[col, 'kreisYield'])
            if clean.empty: continue

            mae = mean_absolute_error(clean['kreisYield'], clean[col])
            r2 = r2_score(clean['kreisYield'], clean[col])
            count = len(clean)

            results.append({'Model': m, 'Count': count, 'MAE': mae, 'R2': r2})

    res_df = pd.DataFrame(results).sort_values('MAE')
    logging.info(f"Analysis Period: {ANALYSIS_START_YEAR}-{ANALYSIS_END_YEAR}")
    logging.info(res_df.to_string(index=False, float_format="%.4f"))
    return res_df


def trench_analysis(df):
    """
    Performs a deep diagnosis of the Regime Switch V10 model's switching logic.
    """
    logging.info("\n" + "#" * 90)
    logging.info("        TRENCH ANALYSIS: DIAGNOSING REGIME SWITCH V10 FAILURE POINTS")
    logging.info("#" * 90)

    required_cols = ['Regime Switch V10_pred', 'Native Ensemble_pred',
                     'Native V2_pred', 'Native V8_pred', 'kreisYield', 'Strategy_Mode']

    analysis_df = df.dropna(subset=required_cols).copy()

    if analysis_df.empty:
        logging.error("Trench Analysis skipped: Insufficient data with V10 predictions.")
        return

    y_true = analysis_df['kreisYield']

    # --- A. The Ideal Benchmark: Perfect Switch Model ---

    # 1. Calculate errors for the three native components
    analysis_df['V2_Error'] = (analysis_df['Native V2_pred'] - y_true).abs()
    analysis_df['V8_Error'] = (analysis_df['Native V8_pred'] - y_true).abs()
    analysis_df['Ensemble_Error'] = (analysis_df['Native Ensemble_pred'] - y_true).abs()

    # 2. Find the minimum possible error for each district/year among the components
    analysis_df['Min_Component_Error'] = analysis_df[
        ['V2_Error', 'V8_Error', 'Ensemble_Error']
    ].min(axis=1)

    mae_perfect_switch = analysis_df['Min_Component_Error'].mean()
    mae_v10 = mean_absolute_error(y_true, analysis_df['Regime Switch V10_pred'])
    mae_ensemble = analysis_df['Ensemble_Error'].mean()

    logging.info("\n" + "=" * 50)
    logging.info("        1. IDEAL VS ACTUAL PERFORMANCE")
    logging.info("=" * 50)
    logging.info(f"MAE (Actual V10 Switch):   {mae_v10:.4f}")
    logging.info(f"MAE (Native Ensemble Base):{mae_ensemble:.4f}")
    logging.info(f"MAE (Perfect Oracle Switch):{mae_perfect_switch:.4f}")
    logging.info("-" * 50)
    logging.info(f"Total V10 Over-Error (vs Ensemble): {mae_v10 - mae_ensemble:+.4f} dt/ha")
    logging.info(
        f"Theoretical Improvement Possible:   {mae_ensemble - mae_perfect_switch:.4f} dt/ha (If switch were perfect)")
    logging.info(f"Cost of Flawed Switch (vs Perfect): {mae_v10 - mae_perfect_switch:.4f} dt/ha")

    # --- B. Strategy Mode Breakdown (Quantifying Switching Cost) ---

    logging.info("\n" + "=" * 50)
    logging.info("        2. STRATEGY MODE FAILURE ANALYSIS")
    logging.info("=" * 50)

    # Calculate the V10 error for comparison
    analysis_df['V10_Error'] = (analysis_df['Regime Switch V10_pred'] - y_true).abs()

    strategy_results = []
    for mode, group in analysis_df.groupby('Strategy_Mode'):
        count = len(group)

        # 1. Base Error (What the model starts with)
        base_mae = group['Ensemble_Error'].mean()

        # 2. Final V10 Error (What the switch resulted in)
        switch_mae = group['V10_Error'].mean()

        # 3. Switching Cost (The error added by the V10 logic)
        switching_cost = switch_mae - base_mae

        strategy_results.append({
            'Strategy Mode': mode,
            'Count': count,
            'Base MAE': base_mae,
            'Switch MAE': switch_mae,
            'Switching Cost': switching_cost
        })

    res_df = pd.DataFrame(strategy_results).sort_values('Switching Cost', ascending=False)

    logging.info(f"{'Strategy Mode':<35} | {'Count':<6} | {'Base MAE':<9} | {'Switch MAE':<9} | {'Switching Cost':<15}")
    logging.info("-" * 80)

    for _, row in res_df.iterrows():
        # Highlight the strategy that is introducing the most error
        marker = ">>" if row['Switching Cost'] > 10 else "  "
        logging.info(
            f"{marker} {row['Strategy Mode']:<32} | {int(row['Count']):<6} | {row['Base MAE']:.2f} | {row['Switch MAE']:.2f} | {row['Switching Cost']:+.2f}"
        )

    # --- C. Geographical/Time Failure Map ---

    # Calculate the error difference: V10_Error - Ensemble_Error
    analysis_df['Error_Delta'] = analysis_df['V10_Error'] - analysis_df['Ensemble_Error']

    logging.info("\n" + "=" * 50)
    logging.info("        3. GEOGRAPHICAL/TIME FAILURE MAP")
    logging.info("=" * 50)

    # Top 5 Districts where the V10 switch *consistently* added the most error (averaged over all years)
    district_failure = analysis_df.groupby('district_no')['Error_Delta'].mean().sort_values(ascending=False)

    logging.info("\n--- Top 5 Districts Where V10 Consistently Fails (Adds Max Error) ---")
    for district, delta in district_failure.head(5).items():
        count = analysis_df[analysis_df['district_no'] == district].shape[0]
        logging.info(f"District {district:<3} ({count} pts): Added error of +{delta:.2f} dt/ha (on avg)")

    # Top 5 Years where the V10 switch *consistently* added the most error (averaged over all districts)
    year_failure = analysis_df.groupby('year')['Error_Delta'].mean().sort_values(ascending=False)

    logging.info("\n--- Top 5 Years Where V10 Consistently Fails (Adds Max Error) ---")
    for year, delta in year_failure.head(5).items():
        logging.info(f"Year {year}: Added error of +{delta:.2f} dt/ha (on avg)")

    logging.info("\n# End of Trench Analysis #")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_merge_models()
    if df.empty:
        logging.error("No data loaded. Exiting.")
        return

    # Filter the data based on the configured time period
    df_filtered = df[(df['year'] >= ANALYSIS_START_YEAR) & (df['year'] <= ANALYSIS_END_YEAR)].copy()
    if df_filtered.empty:
        logging.error(f"No data available for the period {ANALYSIS_START_YEAR}-{ANALYSIS_END_YEAR}. Exiting.")
        return

    logging.info(
        f"--- Analysis set to period: {ANALYSIS_START_YEAR} - {ANALYSIS_END_YEAR} ({len(df_filtered)} data points) ---")

    # 1. Overall Performance Check
    all_models = [
        'Statistical Trend', 'V31 Solar Gated', 'Native Ensemble', 'Regime Switch V10',
        'Hybrid XGB', 'Native V8', 'Native V2', 'Standalone XGB'
    ]
    analyze_model_performance(df_filtered, all_models, title="OVERALL MODEL ACCURACY")

    # 2. Deep Trench Analysis on Regime Switch V10
    trench_analysis(df_filtered)


if __name__ == '__main__':
    # Execute the analysis
    main()