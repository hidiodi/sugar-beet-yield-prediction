# File: src/02_models/Wofost7.1/validation_dashboard.py
# Description: Validation Dashboard (v3.2 - Stress Diagnostics).
#              - Standard Calibration (R2, Bias).
#              - Comparative Diagnosis (Yield vs Potential).
#              - ADDED: Stress Impact Analysis (Correlates Yield Dips with Water Stress).

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
OUTPUT_DIR = (config.BASE_DIR / 'reports/figures/validation_dashboard').resolve()


def load_all_data():
    logging.info("Loading datasets...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Simulation Results
    sim_path = CONFIG['FILE_PATHS']['OUTPUT_DIR'] / "forecast_ensemble_results_raw.csv"
    if not sim_path.exists():
        logging.error(f"Missing file: {sim_path}")
        sys.exit(1)

    df_sim = pd.read_csv(sim_path, dtype={'district_no': str})
    df_sim['district_no'] = df_sim['district_no'].astype(str).str.zfill(5)
    df_sim['year'] = df_sim['year'].astype(int)

    # 2. Actual Yields
    yield_path = CONFIG['FILE_PATHS']['YIELD_DATA']
    df_act = pd.read_csv(yield_path, dtype={'district_no': str})
    df_act['district_no'] = df_act['district_no'].astype(str).str.zfill(5)
    df_act['year'] = df_act['year'].astype(int)
    df_act.rename(columns={'yield': 'actual_yield'}, inplace=True)

    # 3. Initial Conditions
    ic_path = config.PROCESSED_DATA_DIR / 'InitialConditions.csv'
    df_ic = pd.DataFrame()
    if ic_path.exists():
        df_ic = pd.read_csv(ic_path, dtype={'district_no': str})
        df_ic['district_no'] = df_ic['district_no'].astype(str).str.zfill(5)
        df_ic['year'] = df_ic['year'].astype(int)

    logging.info(f"Loaded {len(df_sim)} simulation rows.")
    return df_sim, df_act, df_ic


def analyze_inputs(df_ic):
    """Plots distributions of Sowing Dates and Initial Water."""
    if df_ic.empty: return
    logging.info("Plotting Input Diagnostics...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if 'sowing_date' in df_ic.columns:
        df_ic['sow_doy'] = pd.to_datetime(df_ic['sowing_date']).dt.dayofyear
        sns.histplot(df_ic['sow_doy'], bins=30, kde=True, ax=axes[0], color='green')
        axes[0].set_title("Sowing Date Distribution (DOY)")

    if 'WAV' in df_ic.columns:
        sns.histplot(df_ic['WAV'], bins=30, kde=True, ax=axes[1], color='blue')
        axes[1].set_title("Initial Available Water (WAV)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_input_diagnostics.png")
    plt.close()


def analyze_internal_dynamics(df_sim):
    """Plots internal model states (Histograms)."""
    logging.info("Plotting Internal Dynamics...")
    potential_cols = [
        'cumulative_water_stress', 'max_lai_achieved',
        'spring_mud_days', 'summer_heavy_rain_events', 'harvest_respiration_gdd'
    ]
    valid_cols = [c for c in potential_cols if c in df_sim.columns]

    if not valid_cols: return

    rows = (len(valid_cols) // 3) + 1
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(valid_cols):
        data = df_sim[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not data.empty:
            sns.histplot(data, bins=30, kde=True, ax=axes[i], color='orange')
            axes[i].set_title(f"{col}")
            axes[i].axvline(data.mean(), color='red', linestyle='--')

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_internal_dynamics.png")
    plt.close()


def process_ensemble(df_sim):
    """Aggregates ensemble to mean."""
    DMC = CONFIG['CONSTANTS']['DMC_SUGARBEET']

    if 'yield_water_limited' not in df_sim.columns:
        if 'yield_wlp' in df_sim.columns:
            df_sim.rename(columns={'yield_wlp': 'yield_water_limited'}, inplace=True)
        else:
            logging.error("CRITICAL: No 'yield_water_limited' column found.")
            sys.exit(1)

    df_sim['yield_fresh_dt'] = (df_sim['yield_water_limited'] / DMC) / 100.0

    agg_dict = {'yield_fresh_dt': 'mean'}
    numeric_cols = df_sim.select_dtypes(include=np.number).columns
    for c in numeric_cols:
        if c not in ['yield_fresh_dt', 'year', 'member']:
            agg_dict[c] = 'mean'

    df_agg = df_sim.groupby(['year', 'district_no']).agg(agg_dict).reset_index()
    df_agg.rename(columns={'yield_fresh_dt': 'sim_yield_mean'}, inplace=True)
    return df_agg


def plot_calibration(df_merged):
    """Generates Calibration Plots (Slope & Bias) + LOGS TABLE."""
    logging.info("Generating Calibration Plots...")
    df_clean = df_merged.dropna(subset=['actual_yield', 'sim_yield_mean'])
    if df_clean.empty: return

    bias = (df_clean['sim_yield_mean'] - df_clean['actual_yield']).mean()

    # Trend
    yearly = df_clean.groupby('year')[['actual_yield', 'sim_yield_mean']].mean().reset_index()

    # LOGGING
    logging.info("\n--- Yearly Trend Data (Actual vs Simulated Mean) ---\n" + yearly.to_string(index=False))

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

    # Bias
    plt.figure(figsize=(10, 6))
    residuals = df_clean['sim_yield_mean'] - df_clean['actual_yield']
    sns.histplot(residuals, bins=30, kde=True, color='purple')
    plt.axvline(bias, color='r', label=f'Bias: {bias:.1f}')
    plt.title(f"Bias Distribution")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "04_calibration_bias.png")
    plt.close()


def analyze_stress_impact(df_sim_raw, df_merged):
    """
    NEW: Visualizes WHY predictions might be depressed.
    Correlates Yield Drops with Cumulative Water Stress.
    """
    logging.info("Generating 06_stress_impact_analysis.png ...")

    # Check if we have stress data
    if 'cumulative_water_stress' not in df_sim_raw.columns:
        logging.warning("Column 'cumulative_water_stress' not found. Skipping Stress Analysis.")
        return

    # Aggregate Stress by Year (Mean of all districts/ensembles)
    stress_yearly = df_sim_raw.groupby('year')['cumulative_water_stress'].mean().reset_index()

    # Aggregate Yields by Year
    yield_yearly = df_merged.groupby('year')[['actual_yield', 'sim_yield_mean']].mean().reset_index()

    # Merge
    df_plot = pd.merge(yield_yearly, stress_yearly, on='year')

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top: Yields
    ax1.plot(df_plot['year'], df_plot['actual_yield'], 'k-o', label='Actual Yield')
    ax1.plot(df_plot['year'], df_plot['sim_yield_mean'], 'b--', label='Simulated Yield')
    ax1.set_ylabel("Yield (dt/ha)")
    ax1.set_title("Yield Performance vs. Water Stress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: Stress
    # Color bars: Red if high stress (> threshold?), Orange otherwise
    # Normalized stress helps visualization
    colors = ['red' if s > df_plot['cumulative_water_stress'].mean() else 'orange' for s in
              df_plot['cumulative_water_stress']]

    ax2.bar(df_plot['year'], df_plot['cumulative_water_stress'], color=colors, alpha=0.7,
            label='Cumulative Water Stress')
    ax2.axhline(df_plot['cumulative_water_stress'].mean(), color='grey', linestyle='--', label='Avg Stress')

    ax2.set_ylabel("Water Stress Index (Simulated)")
    ax2.set_xlabel("Year")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_stress_impact_analysis.png")
    plt.close()

    # Log correlation
    corr = df_plot['sim_yield_mean'].corr(df_plot['cumulative_water_stress'])
    logging.info(f"Correlation (Sim Yield vs Water Stress): {corr:.3f}")


def generate_comparative_diagnosis(df_sim_raw, df_act):
    """Existing comparative graph generation."""
    logging.info("Generating 05_comparative_performance_diagnosis.png ...")
    DMC = CONFIG['CONSTANTS']['DMC_SUGARBEET']
    hist_path = CONFIG['FILE_PATHS']['OUTPUT_DIR'] / "historical_validation_results.csv"

    # Prepare Forecast Stats
    df_sim = df_sim_raw.copy()
    if 'yield_water_limited' not in df_sim.columns: df_sim.rename(columns={'yield_wlp': 'yield_water_limited'},
                                                                  inplace=True)
    if 'yield_potential' not in df_sim.columns: df_sim['yield_potential'] = 0

    df_sim['yield_fresh'] = (df_sim['yield_water_limited'] / DMC) / 100.0
    df_sim['pot_fresh'] = (df_sim['yield_potential'] / DMC) / 100.0

    def q10(x):
        return x.quantile(0.10)

    def q90(x):
        return x.quantile(0.90)

    df_agg = df_sim.groupby(['year', 'district_no']).agg({
        'yield_fresh': ['mean', q10, q90],
        'pot_fresh': 'mean'
    }).reset_index()
    df_agg.columns = ['year', 'district_no', 'fc_mean', 'fc_p10', 'fc_p90', 'pot_mean']

    # Merge with Actuals
    df_merged = pd.merge(df_agg, df_act, on=['year', 'district_no'], how='inner')

    # Try load historical
    if hist_path.exists():
        df_hist = pd.read_csv(hist_path, dtype={'district_no': str})
        df_hist['district_no'] = df_hist['district_no'].astype(str).str.zfill(5)
        df_hist['year'] = df_hist['year'].astype(int)
        if 'lintul_yield_perfect_weather' in df_hist.columns:
            df_hist['perf_yield'] = (df_hist['lintul_yield_perfect_weather'] / DMC) / 100.0
            df_merged = pd.merge(df_merged, df_hist[['year', 'district_no', 'perf_yield']], on=['year', 'district_no'],
                                 how='left')

    df_plot = df_merged.dropna(subset=['actual_yield'])
    if df_plot.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(f"WOFOST Performance Diagnosis ({df_plot['year'].min()}-{df_plot['year'].max()})", fontsize=16)

    all_vals = list(df_plot['actual_yield']) + list(df_plot['fc_mean'])
    min_val, max_val = min(all_vals) * 0.9, max(all_vals) * 1.05

    # Left: Perfect Weather (if available)
    ax0 = axes[0]
    if 'perf_yield' in df_plot.columns:
        sub = df_plot.dropna(subset=['perf_yield'])
        if not sub.empty:
            mae = mean_absolute_error(sub['actual_yield'], sub['perf_yield'])
            r2 = r2_score(sub['actual_yield'], sub['perf_yield'])
            ax0.scatter(sub['actual_yield'], sub['perf_yield'], c='steelblue', alpha=0.7,
                        label='Simulated (Perfect Weather)')
            ax0.set_title(f"Perfect Weather\nMAE={mae:.2f}, R2={r2:.3f}", fontsize=14)
    ax0.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax0.set_xlim(min_val, max_val);
    ax0.set_ylim(min_val, max_val)
    ax0.grid(True, alpha=0.3)
    ax0.set_xlabel("Actual Yield");
    ax0.set_ylabel("Simulated Yield")

    # Right: Forecast
    ax1 = axes[1]
    mae = mean_absolute_error(df_plot['actual_yield'], df_plot['fc_mean'])
    r2 = r2_score(df_plot['actual_yield'], df_plot['fc_mean'])

    y_err_low = (df_plot['fc_mean'] - df_plot['fc_p10']).clip(lower=0)
    y_err_high = (df_plot['fc_p90'] - df_plot['fc_mean']).clip(lower=0)
    ax1.errorbar(df_plot['actual_yield'], df_plot['fc_mean'], yerr=[y_err_low, y_err_high], fmt='none',
                 ecolor='lightgrey', alpha=0.6)
    ax1.scatter(df_plot['actual_yield'], df_plot['fc_mean'], c='orange', marker='s', edgecolor='grey',
                label='Ensemble Mean')

    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax1.set_title(f"Forecast Weather\nMean MAE={mae:.2f}, Mean R2={r2:.3f}", fontsize=14)
    ax1.set_xlabel("Actual Yield");
    ax1.set_ylabel("Simulated Yield")
    ax1.set_xlim(min_val, max_val);
    ax1.set_ylim(min_val, max_val)
    ax1.grid(True, alpha=0.3);
    ax1.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_comparative_performance_diagnosis.png", dpi=300)
    plt.close()


def main():
    df_sim_raw, df_act, df_ic = load_all_data()

    analyze_inputs(df_ic)
    analyze_internal_dynamics(df_sim_raw)

    df_sim_agg = process_ensemble(df_sim_raw)
    df_merged = pd.merge(df_sim_agg, df_act, on=['year', 'district_no'], how='inner')

    if not df_merged.empty:
        plot_calibration(df_merged)

        # --- NEW ANALYSIS ---
        analyze_stress_impact(df_sim_raw, df_merged)
        # --------------------

        generate_comparative_diagnosis(df_sim_raw, df_act)

        # Save Results
        df_merged.to_csv(OUTPUT_DIR / "wofost_validation_merged.csv", index=False)

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