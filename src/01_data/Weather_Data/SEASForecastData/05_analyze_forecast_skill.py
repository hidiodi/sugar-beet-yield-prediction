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
# Input 1: The raw forecast data from Script 01
SEAS5_FEATURES_PATH = BASE_DIR / "data/02_intermediate/seas5_forecast_features_1979_2021.csv"

# Input 2: The HYBRID file for our model, which is created by YOUR working Script 02
HYBRID_DATA_PATH = BASE_DIR / "data/03_processed/final_hybrid_features_only.csv"

# Input 3: The full "actuals" data, created by the new Script 02b for analysis
GROUND_TRUTH_PATH = BASE_DIR / "data/02_intermediate/agera5_ground_truth_1979_2021.csv"

# --- OUTPUT FILES ---
ANALYSIS_OUTPUT_DIR = BASE_DIR / "reports/forecast_analysis"
FINAL_FEATURES_PATH = BASE_DIR / "data/03_processed/final_hybrid_features_only.csv"


# --- 2. MAIN WORKFLOW ---
def analyze_and_finalize_features():
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("--- SCRIPT START: analyze_and_finalize_features ---")

    try:
        # Load the forecast data (for the analysis part)
        forecast_df = pd.read_csv(SEAS5_FEATURES_PATH)

        # Load the ground truth data (used as the "actual" values in the analysis)
        truth_df = pd.read_csv(GROUND_TRUTH_PATH)

        # Load the hybrid data (used for the final sanity check and is the script's final output)
        hybrid_df = pd.read_csv(HYBRID_DATA_PATH)

    except FileNotFoundError as e:
        logging.error(
            f"FATAL: Input file not found. Ensure ALL previous scripts (01, 02, 02b) have run. Missing: {e.filename}")
        return

    # --- PART 1: REAL FORECAST SKILL ANALYSIS ---
    # This part compares the FORECAST against the GROUND TRUTH.
    logging.info("--- STARTING MEANINGFUL FORECAST SKILL ANALYSIS (1981-2021) ---")
    analysis_df = pd.merge(forecast_df, truth_df, on=['year', 'district_no'])
    analysis_df = analysis_df[analysis_df['year'] > 1980].copy()

    feature_pairs = [
        ('spring_temp_anomaly_forecast', 'spring_temp_anomaly_actual'),
        ('spring_precip_anomaly_forecast', 'spring_precip_anomaly_actual'),
        ('summer_temp_anomaly_forecast', 'summer_temp_anomaly_actual'),
        ('summer_precip_anomaly_forecast', 'summer_precip_anomaly_actual'),
    ]

    for forecast_col, actual_col in feature_pairs:
        logging.info(f"\n--- Analyzing: {forecast_col} vs. {actual_col} ---")
        forecast_series = analysis_df[forecast_col]
        actual_series = analysis_df[actual_col]

        correlation = forecast_series.corr(actual_series)
        mae = mean_absolute_error(actual_series, forecast_series)
        logging.info(f"  Correlation: {correlation:.3f}, Mean Absolute Error (MAE): {mae:.3f}")

        plt.figure(figsize=(8, 8))
        plt.scatter(actual_series, forecast_series, alpha=0.1)
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Forecast")
        plt.title(f'Forecast vs. Actual: {forecast_col}\nCorrelation: {correlation:.2f}, MAE: {mae:.2f}')
        plt.xlabel("Actual Anomaly (AGERA5)")
        plt.ylabel("Forecast Anomaly (SEAS5)")
        plt.grid(True);
        plt.legend();
        plt.axis('equal');
        plt.tight_layout()
        plot_path = ANALYSIS_OUTPUT_DIR / f"{forecast_col}_vs_actual_skill_scatter.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"  Scatter plot saved to: {plot_path}")

    # --- PART 2: FINAL HYBRID DATASET SANITY CHECK AND SAVING ---
    # This part works ONLY with the HYBRID data, which is what you need for your model.
    logging.info("\n--- FINAL HYBRID DATASET SANITY CHECK (This is the data for your model) ---")

    logging.info(f"Final dataset shape: {hybrid_df.shape}")
    if hybrid_df.isnull().sum().sum() > 0:
        logging.error("FATAL: NaN values found in the final hybrid dataset.")
        return
    else:
        logging.info("OK: No NaN values found in the final hybrid dataset.")

    for col in [c for c in hybrid_df.columns if '_hybrid' in c]:
        logging.info(
            f"  Stats for '{col}': Mean={hybrid_df[col].mean():.2f}, Min={hybrid_df[col].min():.2f}, Max={hybrid_df[col].max():.2f}")

    FINAL_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    hybrid_df.to_csv(FINAL_FEATURES_PATH, index=False)
    logging.info(f"SUCCESS: Final hybrid feature set for modeling saved to {FINAL_FEATURES_PATH}")
    logging.info("--- SCRIPT END ---")


if __name__ == "__main__":
    analyze_and_finalize_features()