# File: analyze_forecast_skill.py
# Description: Compares the processed forecast data against the high-resolution
#              ground truth to analyze forecast skill, errors, and biases.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import logging
from tqdm import tqdm

# --- Setup detailed logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- 1. CONFIGURATION ---
BASE_DIR = Path.cwd()

# --- INPUT FILES ---
ECMWF51_FEATURES_PATH = BASE_DIR / "data/02_intermediate/ecmwf51_forecast_features_FINAL.csv"
GROUND_TRUTH_PATH = BASE_DIR / "data/02_intermediate/agera5_ground_truth_FINAL.csv"

# --- OUTPUT DIRECTORY ---
ANALYSIS_OUTPUT_DIR = BASE_DIR / "reports/forecast_analysis"

# --- 2. MAIN WORKFLOW ---
def analyze_forecast_skill():
    """
    Analyzes the skill of both anomaly and probabilistic ECMWF51 forecasts
    against the AgERA5 ground truth.
    """
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("--- SCRIPT START: analyze_forecast_skill ---")

    try:
        forecast_df = pd.read_csv(ECMWF51_FEATURES_PATH)
        truth_df = pd.read_csv(GROUND_TRUTH_PATH)
    except FileNotFoundError as e:
        logging.error(f"FATAL: Input file not found. Ensure previous scripts have run. Missing: {e.filename}")
        return

    analysis_df = pd.merge(forecast_df, truth_df, on=['year', 'district_no'], how='inner')
    analysis_df = analysis_df[analysis_df['year'] >= 1981].copy()
    logging.info(f"Analysis will be performed on {len(analysis_df)} data points from 1981 onwards.")

    # --- PART 1: ANOMALY FORECAST SKILL (Magnitude) ---
    logging.info("\n--- PART 1: Analyzing Anomaly Forecast Skill (Magnitude) ---")
    forecast_anomaly_cols = [col for col in forecast_df.columns if 'anomaly_forecast' in col]
    anomaly_pairs = []
    for f_col in forecast_anomaly_cols:
        a_col = f_col.replace('_forecast', '_actual')
        if a_col in truth_df.columns:
            anomaly_pairs.append((f_col, a_col))

    logging.info(f"Found {len(anomaly_pairs)} anomaly pairs to analyze.")
    for forecast_col, actual_col in tqdm(anomaly_pairs, desc="Analyzing Anomaly Forecasts"):
        forecast_series = analysis_df[forecast_col]
        actual_series = analysis_df[actual_col]

        correlation = forecast_series.corr(actual_series)
        mae = mean_absolute_error(actual_series, forecast_series)
        logging.info(f"  Metrics for {forecast_col}: Correlation={correlation:.3f}, MAE={mae:.3f}")

        plt.figure(figsize=(8, 8))
        plt.scatter(actual_series, forecast_series, alpha=0.1, label='Forecast vs. Actual')
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Forecast")
        plt.title(f'Anomaly Forecast vs. Actual: {forecast_col.replace("_forecast", "")}\nCorrelation: {correlation:.2f}, MAE: {mae:.2f}')
        plt.xlabel("Actual Anomaly (AGERA5)")
        plt.ylabel("Forecast Anomaly (ECMWF51)")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plot_path = ANALYSIS_OUTPUT_DIR / f"{forecast_col}_vs_actual_skill_scatter.png"
        plt.savefig(plot_path)
        plt.close()

    # --- PART 2: PROBABILISTIC FORECAST SKILL (Confidence) ---
    logging.info("\n--- PART 2: Analyzing Probabilistic Forecast Skill (Confidence) ---")
    prob_pairs = []
    forecast_prob_cols = [col for col in forecast_df.columns if 'prob' in col]
    for p_col in forecast_prob_cols:
        a_col = None
        if p_col.endswith('_temp_prob_warm_forecast'):
            base_name = p_col.replace('_prob_warm_forecast', '')
            a_col = f"{base_name}_anomaly_actual"
            direction = 'Warmer'
        elif p_col.endswith('_precip_prob_wet_forecast'):
            base_name = p_col.replace('_prob_wet_forecast', '')
            a_col = f"{base_name}_anomaly_actual"
            direction = 'Wetter'

        if a_col and a_col in truth_df.columns:
            prob_pairs.append((p_col, a_col, direction))

    logging.info(f"Found {len(prob_pairs)} probability pairs to analyze.")
    if not prob_pairs:
        logging.warning("Could not find any matching probability vs. actual columns. Skipping this analysis part.")
    else:
        for forecast_col, actual_col, direction in tqdm(prob_pairs, desc="Analyzing Probability Forecasts"):
            forecast_series = analysis_df[forecast_col]
            actual_series = analysis_df[actual_col]
            correlation = forecast_series.corr(actual_series)
            logging.info(f"  Metrics for {forecast_col}: Correlation={correlation:.3f}")

            plt.figure(figsize=(8, 8))
            plt.scatter(actual_series, forecast_series, alpha=0.1)
            plt.axhline(y=0.5, color='r', linestyle='--', label='Climatological Forecast (50%)')
            plt.axvline(x=0, color='grey', linestyle=':', alpha=0.5)
            plt.title(f'Probabilistic Forecast vs. Actual Outcome\nCorrelation: {correlation:.2f}')
            plt.xlabel("Actual Anomaly (AGERA5)")
            plt.ylabel(f"Forecast Probability of a {direction} Season (ECMWF51)")
            plt.ylim(0, 1)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plot_path = ANALYSIS_OUTPUT_DIR / f"{forecast_col}_vs_actual_skill_scatter.png"
            plt.savefig(plot_path)
            plt.close()

    logging.info("\n--- SCRIPT END ---")


if __name__ == "__main__":
    analyze_forecast_skill()