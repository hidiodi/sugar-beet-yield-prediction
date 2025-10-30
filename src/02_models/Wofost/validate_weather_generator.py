# File: validate_weather_generator.py
# Description: A diagnostic script to visually compare historical weather data
#              against synthetic data produced by the WeatherGenerator to
#              identify and understand model biases.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# --- CRITICAL IMPORT ---
try:
    from run_lintul_one_year_test_pipeline import WeatherGenerator, CONFIG
except ImportError:
    print("FATAL: Could not import from 'run_lintul_one_year_test_pipeline.py'.")
    print("Please ensure this script is in the same directory as the main pipeline script.")
    exit()

# --- Configuration ---
TEST_DISTRICT = '01051'
TEST_YEAR = 2018
OUTPUT_DIR = 'data/07_model_diagnostics'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')


def convert_units_for_plotting(df):
    """Converts the raw data units to human-readable units for plotting."""
    df_out = df.copy()
    df_out['precip_mm'] = df_out['precip'] / 100000.0
    df_out['srad_kj_m2'] = (df_out['srad'] * 10.0) / 1000.0
    return df_out


def create_diagnostic_plots(df_hist, df_synth, district, year, output_dir):
    """Generates and saves a dashboard of comparison plots."""
    logging.info("Creating diagnostic plots...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Weather Generator Diagnostics - District {district} ({year})', fontsize=20, fontweight='bold')

    # --- 1. Rainfall Distribution Plot (The "Drizzle Problem") ---
    ax1 = axes[0, 0]
    hist_rain = df_hist[df_hist['precip_mm'] > 0.1]['precip_mm']
    synth_rain = df_synth[df_synth['precip_mm'] > 0.1]['precip_mm']

    sns.histplot(hist_rain, ax=ax1, color='blue', label=f'Historical (Mean: {hist_rain.mean():.2f} mm)', kde=True,
                 stat='density', binwidth=2)
    sns.histplot(synth_rain, ax=ax1, color='red', label=f'Synthetic (Mean: {synth_rain.mean():.2f} mm)', kde=True,
                 stat='density', binwidth=2)
    ax1.set_title('Rainfall Distribution on Wet Days', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Precipitation (mm/day)')
    ax1.legend()
    ax1.text(0.95, 0.95,
             'Hypothesis: Synthetic data lacks extreme rainfall events\nand creates too much "average" drizzle.',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- 2. Tmax vs. Solar Radiation Correlation Plot ---
    ax2 = axes[0, 1]
    ax2.scatter(df_hist['tmax'], df_hist['srad_kj_m2'], alpha=0.5, label='Historical', color='blue')
    ax2.scatter(df_synth['tmax'], df_synth['srad_kj_m2'], alpha=0.5, label='Synthetic', color='red')
    ax2.set_title('Correlation between Tmax and Solar Radiation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Maximum Temperature (°C)')
    ax2.set_ylabel('Solar Radiation (kJ/m²/day)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.95, 0.95,
             'Hypothesis: Synthetic data is uncorrelated.\nReal weather shows that hotter days are usually sunnier.',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- 3. Summer Time Series Plot ---
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1, :]: ax.remove()
    ax_ts = fig.add_subplot(gs[1, :])

    df_hist_summer = df_hist[(df_hist['date'].dt.month >= 6) & (df_hist['date'].dt.month <= 8)]
    df_synth_summer = df_synth[(df_synth['date'].dt.month >= 6) & (df_synth['date'].dt.month <= 8)]

    ax_ts.plot(df_hist_summer['date'], df_hist_summer['tmax'], label='Historical Tmax', color='darkblue', alpha=0.8)
    ax_ts.plot(df_synth_summer['date'], df_synth_summer['tmax'], label='Synthetic Tmax', color='darkred',
               linestyle='--', alpha=0.8)

    ax_ts.set_ylabel('Maximum Temperature (°C)', color='darkred')
    ax_ts.set_title('Summer (Jun-Aug) Time Series Comparison', fontsize=14, fontweight='bold')
    ax_ts.tick_params(axis='y', labelcolor='darkred')
    ax_ts.legend(loc='upper left')
    ax_ts.grid(True, alpha=0.3)

    ax_precip = ax_ts.twinx()
    ax_precip.bar(df_hist_summer['date'], df_hist_summer['precip_mm'], label='Historical Precip', color='blue',
                  alpha=0.3, width=0.8)
    ax_precip.bar(df_synth_summer['date'] + pd.Timedelta(hours=12), df_synth_summer['precip_mm'],
                  label='Synthetic Precip', color='red', alpha=0.3, width=0.4)
    ax_precip.set_ylabel('Precipitation (mm)', color='darkblue')
    ax_precip.tick_params(axis='y', labelcolor='darkblue')
    ax_precip.legend(loc='upper right')
    ax_precip.set_ylim(0, max(df_hist_summer['precip_mm'].max(), df_synth_summer['precip_mm'].max()) * 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = os.path.join(output_dir, f'weather_generator_dashboard_{district}_{year}.png')
    plt.savefig(plot_path, dpi=150)
    logging.info(f"Diagnostic dashboard saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info("Loading full historical weather dataset for fitting...")
    try:
        df_all_hist_raw = pd.read_csv(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER'], parse_dates=['date'])

        # --- FIX: Ensure district_no is a zero-padded string for consistent filtering ---
        df_all_hist_raw['district_no'] = df_all_hist_raw['district_no'].astype(str).str.zfill(5)

    except FileNotFoundError:
        logging.error(f"FATAL: Historical weather file not found at {CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER']}")
        exit()

    logging.info("Initializing and fitting WeatherGenerator...")
    wg = WeatherGenerator()
    wg.fit(df_all_hist_raw)
    logging.info("Fitting complete.")

    logging.info(f"Generating synthetic weather for district {TEST_DISTRICT}, year {TEST_YEAR}...")
    start_date = f'{TEST_YEAR}-01-01'
    end_date = f'{TEST_YEAR}-12-31'
    zero_anomalies = {}
    df_synth_raw = wg.generate(TEST_DISTRICT, start_date, end_date, zero_anomalies)

    if df_synth_raw is None or df_synth_raw.empty:
        logging.error("WeatherGenerator failed to produce a synthetic dataset. Aborting.")
        exit()

    df_hist_raw = df_all_hist_raw[
        (df_all_hist_raw['district_no'] == TEST_DISTRICT) & (df_all_hist_raw['date'].dt.year == TEST_YEAR)].copy()
    if df_hist_raw.empty:
        logging.error(f"Could not find historical data for district {TEST_DISTRICT} in year {TEST_YEAR}. Aborting.")
        exit()

    logging.info("Converting data to standard units for plotting...")
    df_hist_plot = convert_units_for_plotting(df_hist_raw)
    df_synth_plot = convert_units_for_plotting(df_synth_raw)

    create_diagnostic_plots(df_hist_plot, df_synth_plot, TEST_DISTRICT, TEST_YEAR, OUTPUT_DIR)

    logging.info("Diagnostic script finished successfully.")