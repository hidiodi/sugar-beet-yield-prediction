import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import logging

# --- Setup detailed logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- 1. CONFIGURATION ---
BASE_DIR = Path.cwd()

# --- INPUT FILES ---
# Input 1: The forecast data with anomaly AND probability features
SEAS5_FEATURES_PATH = BASE_DIR / "data/02_intermediate/seas5_forecast_features_1979_2021.csv"
# Input 2: The full "actuals" data for comparison
GROUND_TRUTH_PATH = BASE_DIR / "data/02_intermediate/agera5_ground_truth_1979_2021.csv"

# --- OUTPUT DIRECTORY ---
ANALYSIS_OUTPUT_DIR = BASE_DIR / "reports/forecast_analysis"

# --- 2. MAIN WORKFLOW ---
def analyze_forecast_skill():
    """
    Analyzes the skill of both anomaly and probabilistic SEAS5 forecasts
    against the AgERA5 ground truth.
    """
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("--- SCRIPT START: analyze_forecast_skill ---")

    try:
        forecast_df = pd.read_csv(SEAS5_FEATURES_PATH)
        truth_df = pd.read_csv(GROUND_TRUTH_PATH)
    except FileNotFoundError as e:
        logging.error(f"FATAL: Input file not found. Ensure previous scripts have run. Missing: {e.filename}")
        return

    # Merge forecast with ground truth. Use an inner join to only keep matching rows.
    # Filter for years where real SEAS5 data exists.
    analysis_df = pd.merge(forecast_df, truth_df, on=['year', 'district_no'], how='inner')
    analysis_df = analysis_df[analysis_df['year'] > 1980].copy()
    logging.info(f"Analysis will be performed on {len(analysis_df)} data points from 1981 onwards.")

    # --- PART 1: ANOMALY FORECAST SKILL (Magnitude) ---
    logging.info("\n--- PART 1: Analyzing Anomaly Forecast Skill (Magnitude) ---")
    anomaly_pairs = [
        ('spring_temp_anomaly_forecast', 'spring_temp_anomaly_actual'),
        ('spring_precip_anomaly_forecast', 'spring_precip_anomaly_actual'),
        ('summer_temp_anomaly_forecast', 'summer_temp_anomaly_actual'),
        ('summer_precip_anomaly_forecast', 'summer_precip_anomaly_actual'),
    ]

    for forecast_col, actual_col in anomaly_pairs:
        logging.info(f"\n--- Analyzing: {forecast_col} vs. {actual_col} ---")
        forecast_series = analysis_df[forecast_col]
        actual_series = analysis_df[actual_col]

        correlation = forecast_series.corr(actual_series)
        mae = mean_absolute_error(actual_series, forecast_series)
        logging.info(f"  Correlation: {correlation:.3f}, Mean Absolute Error (MAE): {mae:.3f}")

        plt.figure(figsize=(8, 8))
        plt.scatter(actual_series, forecast_series, alpha=0.1, label='Forecast vs. Actual')
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Forecast")
        plt.title(f'Anomaly Forecast vs. Actual: {forecast_col.replace("_forecast", "")}\nCorrelation: {correlation:.2f}, MAE: {mae:.2f}')
        plt.xlabel("Actual Anomaly (AGERA5)")
        plt.ylabel("Forecast Anomaly (SEAS5)")
        plt.grid(True); plt.legend(); plt.axis('equal'); plt.tight_layout()
        plot_path = ANALYSIS_OUTPUT_DIR / f"{forecast_col}_vs_actual_skill_scatter.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"  Scatter plot saved to: {plot_path}")

    # --- PART 2: PROBABILISTIC FORECAST SKILL (Confidence) ---
    logging.info("\n--- PART 2: Analyzing Probabilistic Forecast Skill (Confidence) ---")
    probability_pairs = [
        ('spring_temp_prob_warm_forecast', 'spring_temp_anomaly_actual', 'Warmer'),
        ('spring_precip_prob_wet_forecast', 'spring_precip_anomaly_actual', 'Wetter'),
        ('summer_temp_prob_warm_forecast', 'summer_temp_anomaly_actual', 'Warmer'),
        ('summer_precip_prob_wet_forecast', 'summer_precip_anomaly_actual', 'Wetter'),
    ]

    for forecast_col, actual_col, direction in probability_pairs:
        logging.info(f"\n--- Analyzing: {forecast_col} vs. {actual_col} ---")
        forecast_series = analysis_df[forecast_col]
        actual_series = analysis_df[actual_col]

        # A high probability forecast should correlate with a positive anomaly
        correlation = forecast_series.corr(actual_series)
        logging.info(f"  Correlation between probability and actual anomaly: {correlation:.3f}")

        plt.figure(figsize=(8, 8))
        plt.scatter(actual_series, forecast_series, alpha=0.1)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Climatological Forecast (50%)')
        plt.axvline(x=0, color='grey', linestyle=':', alpha=0.5)
        plt.title(f'Probabilistic Forecast vs. Actual Outcome\nCorrelation: {correlation:.2f}')
        plt.xlabel("Actual Anomaly (AGERA5)")
        plt.ylabel(f"Forecast Probability of a {direction} Season (SEAS5)")
        plt.ylim(0, 1) # Probability is always between 0 and 1
        plt.grid(True); plt.legend(); plt.tight_layout()
        plot_path = ANALYSIS_OUTPUT_DIR / f"{forecast_col}_vs_actual_skill_scatter.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"  Scatter plot saved to: {plot_path}")

    logging.info("\n--- SCRIPT END ---")


if __name__ == "__main__":
    analyze_forecast_skill()