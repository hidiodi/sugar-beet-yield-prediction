# File: src/02_models/Wofost7.1/validation_dashboard.py
# Description: Validation Dashboard (v3.1 - Dynamic Columns).
#              - Adapts to exact CSV columns (No KeyErrors).
#              - Visualizes the metrics YOU actually saved (Mud days, Frost, Respiration).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# --- Config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG
# Ensure the path is absolute to avoid FileNotFoundError on saving
OUTPUT_DIR = (config.BASE_DIR / 'reports/figures/validation_dashboard').resolve()


def load_all_data():
    logging.info("Loading datasets...")

    # 1. Create Output Directory
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create output directory {OUTPUT_DIR}: {e}")
        sys.exit(1)

    # 2. Simulation Results
    sim_path = CONFIG['FILE_PATHS']['OUTPUT_DIR'] / "forecast_ensemble_results_raw.csv"
    if not sim_path.exists():
        logging.error(f"Missing file: {sim_path}")
        sys.exit(1)

    df_sim = pd.read_csv(sim_path, dtype={'district_no': str})
    df_sim['district_no'] = df_sim['district_no'].astype(str).str.zfill(5)
    df_sim['year'] = df_sim['year'].astype(int)

    # 3. Actual Yields
    yield_path = CONFIG['FILE_PATHS']['YIELD_DATA']
    df_act = pd.read_csv(yield_path, dtype={'district_no': str})
    df_act['district_no'] = df_act['district_no'].astype(str).str.zfill(5)
    df_act['year'] = df_act['year'].astype(int)
    df_act.rename(columns={'yield': 'actual_yield'}, inplace=True)

    # 4. Initial Conditions
    ic_path = config.PROCESSED_DATA_DIR / 'InitialConditions.csv'
    df_ic = pd.DataFrame()
    if ic_path.exists():
        df_ic = pd.read_csv(ic_path, dtype={'district_no': str})
        df_ic['district_no'] = df_ic['district_no'].astype(str).str.zfill(5)
        df_ic['year'] = df_ic['year'].astype(int)

    logging.info(f"Loaded {len(df_sim)} simulation rows.")
    logging.info(f"Simulation Columns found: {list(df_sim.columns)}")

    return df_sim, df_act, df_ic


def analyze_inputs(df_ic):
    """Plots distributions of Sowing Dates and Initial Water."""
    if df_ic.empty: return

    logging.info("Plotting Input Diagnostics...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Sowing Date
    if 'sowing_date' in df_ic.columns:
        df_ic['sow_doy'] = pd.to_datetime(df_ic['sowing_date']).dt.dayofyear
        sns.histplot(df_ic['sow_doy'], bins=30, kde=True, ax=axes[0], color='green')
        axes[0].set_title("Sowing Date Distribution (DOY)")
        axes[0].axvline(60, color='k', linestyle='--', label='March 1')
        axes[0].legend()

    # WAV
    if 'WAV' in df_ic.columns:
        sns.histplot(df_ic['WAV'], bins=30, kde=True, ax=axes[1], color='blue')
        axes[1].set_title("Initial Available Water (WAV)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_input_diagnostics.png")
    plt.close()


def analyze_internal_dynamics(df_sim):
    """
    Plots internal model states.
    Dynamically picks relevant numeric columns from the CSV.
    """
    logging.info("Plotting Internal Dynamics...")

    # Define priority columns we want to see if they exist
    # Based on your CSV header
    potential_cols = [
        'cumulative_water_stress',
        'max_lai_achieved',
        'spring_mud_days',
        'summer_heavy_rain_events',
        'harvest_respiration_gdd'
    ]

    # Filter to what actually exists
    valid_cols = [c for c in potential_cols if c in df_sim.columns]

    if not valid_cols:
        logging.warning("No internal dynamic columns found to plot.")
        return

    # Create grid
    n_cols = len(valid_cols)
    rows = (n_cols // 3) + 1
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(valid_cols):
        ax = axes[i]
        # Drop NAs and Infs
        data = df_sim[col].replace([np.inf, -np.inf], np.nan).dropna()

        if not data.empty:
            sns.histplot(data, bins=30, kde=True, ax=ax, color='orange')
            ax.set_title(f"{col}")
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
            ax.legend()

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_internal_dynamics.png")
    plt.close()


def process_ensemble(df_sim):
    """Aggregates ensemble to mean."""
    DMC = CONFIG['CONSTANTS']['DMC_SUGARBEET']

    # Safety check for yield column
    if 'yield_water_limited' not in df_sim.columns:
        # Try fallback names if schema changed
        if 'yield_wlp' in df_sim.columns:
            df_sim.rename(columns={'yield_wlp': 'yield_water_limited'}, inplace=True)
        else:
            logging.error("CRITICAL: No 'yield_water_limited' column found.")
            sys.exit(1)

    df_sim['yield_fresh_dt'] = (df_sim['yield_water_limited'] / DMC) / 100.0

    # Aggregate
    agg_dict = {'yield_fresh_dt': 'mean'}

    # Dynamically add means for other numeric columns
    numeric_cols = df_sim.select_dtypes(include=np.number).columns
    for c in numeric_cols:
        if c not in ['yield_fresh_dt', 'year', 'member']:
            agg_dict[c] = 'mean'

    # Group
    df_agg = df_sim.groupby(['year', 'district_no']).agg(agg_dict).reset_index()
    df_agg.rename(columns={'yield_fresh_dt': 'sim_yield_mean'}, inplace=True)

    return df_agg


def plot_calibration(df_merged):
    """Generates Calibration Plots (Slope & Bias)."""
    logging.info("Generating Calibration Plots...")

    df_clean = df_merged.dropna(subset=['actual_yield', 'sim_yield_mean'])
    if df_clean.empty: return

    bias = (df_clean['sim_yield_mean'] - df_clean['actual_yield']).mean()

    # 1. Trend
    yearly = df_clean.groupby('year')[['actual_yield', 'sim_yield_mean']].mean().reset_index()
    plt.figure(figsize=(14, 7))
    plt.plot(yearly['year'], yearly['actual_yield'], 'k-o', label='Actual')
    plt.plot(yearly['year'], yearly['sim_yield_mean'], 'b--', label='WOFOST Sim')

    if len(yearly) > 1:
        s_act = np.polyfit(yearly['year'], yearly['actual_yield'], 1)[0]
        s_sim = np.polyfit(yearly['year'], yearly['sim_yield_mean'], 1)[0]
        plt.title(f"Trend Check\nSlope Actual: {s_act:.2f} | Slope Sim: {s_sim:.2f}")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "03_calibration_trend.png")
    plt.close()

    # 2. Bias
    plt.figure(figsize=(10, 6))
    residuals = df_clean['sim_yield_mean'] - df_clean['actual_yield']
    sns.histplot(residuals, bins=30, kde=True, color='purple')
    plt.axvline(bias, color='r', label=f'Bias: {bias:.1f}')
    plt.title(f"Bias Distribution (Target: 0)")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "04_calibration_bias.png")
    plt.close()


def main():
    df_sim_raw, df_act, df_ic = load_all_data()

    # Plotting
    analyze_inputs(df_ic)
    analyze_internal_dynamics(df_sim_raw)

    # Process & Validate
    df_sim_agg = process_ensemble(df_sim_raw)
    df_merged = pd.merge(df_sim_agg, df_act, on=['year', 'district_no'], how='inner')

    if not df_merged.empty:
        plot_calibration(df_merged)
        # Save CSV
        df_merged.to_csv(OUTPUT_DIR / "wofost_validation_merged.csv", index=False)

        # Log Metrics
        bias = (df_merged['sim_yield_mean'] - df_merged['actual_yield']).mean()
        r2 = r2_score(df_merged['actual_yield'], df_merged['sim_yield_mean'])
        logging.info(f"--- RESULTS ---")
        logging.info(f"R2: {r2:.3f}")
        logging.info(f"Bias: {bias:.1f} dt/ha")
        logging.info(f"Full report in: {OUTPUT_DIR}")
    else:
        logging.error("No overlap between simulation and actuals.")


if __name__ == "__main__":
    main()