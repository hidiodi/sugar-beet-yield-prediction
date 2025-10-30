# File: src/data_prep/estimate_historical_management.py
# Description: Estimates historical planting dates, performs a sensitivity analysis,
#              and runs formal quality gates for plausibility and stability.
#
# This script fulfills all of Phase 0 of the Prescriptive Hybrid Ensemble Master Plan.

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# === G L O B A L   C O N F I G U R A T I O N ===
# ==============================================================================

CONFIG = {
    'FILE_PATHS': {
        'HISTORICAL_YIELDS': Path('data/02_intermediate/sugarbeet_yield.csv'),
        'HISTORICAL_DAILY_WEATHER_DIR': Path('data/02_intermediate/daily_weather/'),
        'OUTPUT_DIR': Path('data/02_intermediate/'),
        'OUTPUT_FILENAME': 'historical_management_sensitivity.csv'
    },
    'GDD_PARAMS': {
        'BASE_TEMPERATURE_C': 4.0,
        'THRESHOLDS_TO_TEST': [125, 150, 175, 200, 225],
        'PRIMARY_THRESHOLD': 175  # The main threshold for Quality Gate 0.1
    },
    'ANALYSIS': {
        'PLOT_OUTPUT_DIR': Path('reports/figures/'),
        'PLOT_FILENAME': 'gdd_threshold_sensitivity_plot.png'
    },
    # --- NEW: Configuration for Quality Gates ---
    'QUALITY_GATES': {
        'PLAUSIBILITY_WINDOW_START': '03-01',  # March 1st
        'PLAUSIBILITY_WINDOW_END': '05-15',  # May 15th
        'PLAUSIBILITY_PASS_PERCENT': 90.0,
        'STABILITY_SHIFT_THRESHOLD_DAYS': 7.0
    }
}

# (Logging setup remains the same)
logging.getLogger().handlers = []
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)


def estimate_planting_date_for_district(district_weather_df: pd.DataFrame, base_temp: float, threshold: float) -> str:
    # (This function is unchanged)
    cumulative_gdd = 0.0
    for _, row in district_weather_df.iterrows():
        tmean = row['tmean']
        daily_gdd = max(0, tmean - base_temp)
        cumulative_gdd += daily_gdd
        if cumulative_gdd > threshold:
            return row['date'].date().isoformat()
    return None


def create_sensitivity_plot(df: pd.DataFrame, output_path: Path):
    # (This function is largely unchanged)
    logging.info("Generating sensitivity analysis plot...")
    df_plot = df.dropna(subset=['est_planting_date']).copy()
    df_plot['day_of_year'] = pd.to_datetime(df_plot['est_planting_date']).dt.dayofyear

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df_plot, x='gdd_threshold', y='day_of_year', ax=ax)
    ax.set_title('Sensitivity of Estimated Planting Date to GDD Threshold', fontsize=16)
    ax.set_xlabel('Cumulative Growing Degree Day (GDD) Threshold', fontsize=12)
    ax.set_ylabel('Estimated Planting Date (Day of Year)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"✓ Sensitivity plot saved to: {output_path}")
    plt.close()


# --- NEW: Quality Gate 0.1 Function ---
def run_quality_gate_0_1_plausibility(df_all_results: pd.DataFrame):
    """Checks if the primary GDD proxy is agronomically plausible."""
    logging.info("--- [QUALITY GATE 0.1: Plausibility Check] ---")
    cfg = CONFIG['QUALITY_GATES']
    primary_threshold = CONFIG['GDD_PARAMS']['PRIMARY_THRESHOLD']

    df_primary = df_all_results[df_all_results['gdd_threshold'] == primary_threshold].dropna()
    df_primary['date'] = pd.to_datetime(df_primary['est_planting_date'])

    # Get day-of-year for the agronomic window
    start_day = pd.to_datetime(f"2000-{cfg['PLAUSIBILITY_WINDOW_START']}").dayofyear
    end_day = pd.to_datetime(f"2000-{cfg['PLAUSIBILITY_WINDOW_END']}").dayofyear

    df_primary['in_window'] = df_primary['date'].dt.dayofyear.between(start_day, end_day)

    total_valid_estimates = len(df_primary)
    if total_valid_estimates == 0:
        logging.error("QG 0.1 FAILED: No valid planting dates were estimated at all.")
        sys.exit(1)

    percent_in_window = (df_primary['in_window'].sum() / total_valid_estimates) * 100.0
    pass_threshold = cfg['PLAUSIBILITY_PASS_PERCENT']

    logging.info(f"Agronomic Window: Day {start_day} (Mar 1) to {end_day} (May 15)")
    logging.info(f"Result for T={primary_threshold}: {percent_in_window:.2f}% of dates fall within the window.")

    if percent_in_window >= pass_threshold:
        logging.info(f"DECISION: PASS! ({percent_in_window:.2f}% >= {pass_threshold}%)")
    else:
        logging.error(f"DECISION: FAIL! ({percent_in_window:.2f}% < {pass_threshold}%)")
        logging.error("The GDD proxy is flawed. Project HALTED as per execution plan.")
        sys.exit(1)


# --- NEW: Quality Gate 0.2 Function ---
def run_quality_gate_0_2_stability(df_all_results: pd.DataFrame):
    """Checks if the proxy is overly sensitive to threshold changes."""
    logging.info("--- [QUALITY GATE 0.2: Stability Check] ---")
    cfg = CONFIG['QUALITY_GATES']
    primary_t = CONFIG['GDD_PARAMS']['PRIMARY_THRESHOLD']
    t_minus_25 = primary_t - 25
    t_plus_25 = primary_t + 25

    # Filter for the relevant thresholds
    df_stability = df_all_results[df_all_results['gdd_threshold'].isin([t_minus_25, primary_t, t_plus_25])].copy()
    df_stability['day_of_year'] = pd.to_datetime(df_stability['est_planting_date']).dt.dayofyear

    # Pivot the table to easily compare dates
    df_pivot = df_stability.pivot_table(index=['year', 'district_no'], columns='gdd_threshold',
                                        values='day_of_year').dropna()

    if len(df_pivot) < 10:
        logging.error("QG 0.2 FAILED: Not enough overlapping data points to perform stability check.")
        sys.exit(1)

    # Calculate absolute shifts
    shift_1 = (df_pivot[primary_t] - df_pivot[t_minus_25]).abs()
    shift_2 = (df_pivot[t_plus_25] - df_pivot[primary_t]).abs()

    mean_absolute_shift = np.mean([shift_1.mean(), shift_2.mean()])
    pass_threshold = cfg['STABILITY_SHIFT_THRESHOLD_DAYS']

    logging.info(f"Comparing shifts between GDD thresholds: {t_minus_25}, {primary_t}, and {t_plus_25}.")
    logging.info(f"Mean absolute shift in planting date for a +/- 25 GDD change: {mean_absolute_shift:.2f} days.")

    if mean_absolute_shift < pass_threshold:
        logging.info(f"DECISION: PASS! ({mean_absolute_shift:.2f} days < {pass_threshold} days)")
    else:
        logging.warning(f"DECISION: FAIL! ({mean_absolute_shift:.2f} days >= {pass_threshold} days)")
        logging.warning(
            "Proxy is unstable. This must be noted as a limitation in the final paper. Proceeding with caution.")


def main():
    logging.info("--- Starting Phase 0: GDD Proxy Estimation and Validation ---")

    # --- Step 1: Data Loading ---
    yield_file = CONFIG['FILE_PATHS']['HISTORICAL_YIELDS']
    try:
        df_yield = pd.read_csv(yield_file)
        df_yield['district_no'] = df_yield['district_no'].astype(str).str.zfill(5)
    except FileNotFoundError:
        logging.error(f"FATAL: Historical yield file not found at {yield_file}. Aborting.")
        sys.exit(1)

    target_combinations = df_yield[['year', 'district_no']].drop_duplicates()
    years_to_process = sorted(target_combinations['year'].unique())

    # --- Step 2: GDD Calculation for All Thresholds ---
    all_results = []
    weather_dir = CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR']
    base_temp = CONFIG['GDD_PARAMS']['BASE_TEMPERATURE_C']
    gdd_thresholds = CONFIG['GDD_PARAMS']['THRESHOLDS_TO_TEST']

    for year in tqdm(years_to_process, desc="Processing Years"):
        weather_file_path = weather_dir / f"historical_daily_weather_era5_{year}.csv"
        if not weather_file_path.exists(): continue
        df_weather_year = pd.read_csv(weather_file_path, parse_dates=['date'])
        df_weather_year['district_no'] = df_weather_year['district_no'].astype(str).str.zfill(5)
        df_weather_year['tmean'] = (df_weather_year['tmin'] + df_weather_year['tmax']) / 2.0
        districts_in_year = target_combinations[target_combinations['year'] == year]['district_no']

        for district in districts_in_year:
            district_weather = df_weather_year[df_weather_year['district_no'] == district].copy()
            district_weather = district_weather[district_weather['date'].dt.month < 8].sort_values(by='date')
            if district_weather.empty: continue
            for threshold in gdd_thresholds:
                est_date = estimate_planting_date_for_district(district_weather, base_temp, threshold)
                all_results.append(
                    {'year': year, 'district_no': district, 'gdd_threshold': threshold, 'est_planting_date': est_date})

    if not all_results:
        logging.error("No planting dates were estimated. Aborting.")
        sys.exit(1)

    output_df = pd.DataFrame(all_results)

    # --- Step 3: Run Quality Gates ---
    run_quality_gate_0_1_plausibility(output_df)
    run_quality_gate_0_2_stability(output_df)

    # --- Step 4: Save Deliverables ---
    output_dir = CONFIG['FILE_PATHS']['OUTPUT_DIR']
    output_path = output_dir / CONFIG['FILE_PATHS']['OUTPUT_FILENAME']
    os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logging.info(f"✓ Success! Sensitivity analysis data saved to: {output_path}")

    plot_path = CONFIG['ANALYSIS']['PLOT_OUTPUT_DIR'] / CONFIG['ANALYSIS']['PLOT_FILENAME']
    create_sensitivity_plot(output_df, plot_path)

    logging.info("\n" + "=" * 70 + "\n✓ PHASE 0 COMPLETED SUCCESSFULLY!\n" + "=" * 70)


if __name__ == "__main__":
    main()