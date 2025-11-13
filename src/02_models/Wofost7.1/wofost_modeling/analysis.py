import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from src import config

CONFIG = config.WOFOST_CONFIG

def analyze_v2_model_inputs(df_static_all, cropdata):
    """
    V2: Performs a diagnostic analysis on the ACTUAL model-ready inputs.
    This checks the pre-calculated physical parameters, not the raw ingredients.
    """
    logging.info(
        "=" * 80 +
        "\n--- V2 INPUT ANALYSIS: Checking pre-calculated physical parameters ---\n"
        + "=" * 80)
    analysis_passed = True

    try:
        logging.info("[1/3] Analyzing static data integrity...")
        required_cols = [
            'SMW', 'SMFCF', 'CRAIRC', 'K0', 'WAV', 'RDMSOL', 'NOTINF',
            'SSMAX', 'latitude', 'longitude', 'avg_elevation'
        ]
        missing_cols = [
            col for col in required_cols if col not in df_static_all.columns]
        if missing_cols:
            logging.error(
                f"    [FAIL] Missing critical columns in merged static data: "
                f"{missing_cols}")
            analysis_passed = False
        else:
            logging.info("    [OK] All required physics columns are present.")

        logging.info("[2/3] Analyzing physical parameter ranges...")
        descriptions = df_static_all[required_cols].describe().T
        logging.info("\n" + descriptions.to_string())

        # Specific checks for plausibility
        if not (0.05 < descriptions.loc['SMFCF', 'mean'] < 0.6):
            logging.warning(
                f"    [WARNING] Mean Field Capacity (SMFCF) is "
                f"{descriptions.loc['SMFCF', 'mean']:.3f} (fraction), "
                "which seems unusual. Expected between 0.05 and 0.6.")
        if descriptions.loc['CRAIRC', 'mean'] < 0.01 or \
           descriptions.loc['CRAIRC', 'mean'] > 0.1:
            logging.warning(
                f"    [WARNING] Mean Critical Air Content (CRAIRC) is "
                f"{descriptions.loc['CRAIRC', 'mean']:.4f} (fraction). "
                "Expected between 0.01 and 0.1 for most soils.")
        if descriptions.loc['WAV', 'min'] < -10.0:
            logging.error(
                f"    [FAIL] Minimum Initial Available Water (WAV) is "
                f"extremely negative ({descriptions.loc['WAV', 'min']:.2f} cm). "
                "Check the winter balance script and soil parameters.")
            analysis_passed = False
        if descriptions.loc['WAV', 'max'] > 50.0:
            logging.warning(
                f"    [WARNING] Maximum Initial Available Water (WAV) is "
                f"very high ({descriptions.loc['WAV', 'max']:.2f} cm). "
                "Check winter balance logic or RDMSOL/SMFCF.")
        if descriptions.loc['RDMSOL', 'mean'] < 100.0 or \
           descriptions.loc['RDMSOL', 'mean'] > 250.0:
            logging.warning(
                f"    [WARNING] Mean Soil-Limited Rooting Depth (RDMSOL) is "
                f"{descriptions.loc['RDMSOL', 'mean']:.1f} cm. "
                "Review its source/default.")

        logging.info(
            "    [OK] Physical parameter ranges appear plausible "
            "(manual review recommended for warnings).")

        logging.info(
            "[3/3] Analyzing crop parameters (subset for sanity check)...")
        if not (500 < cropdata.get('TSUM1', 0) < 1200):
            logging.warning(
                f"    [WARNING] TSUM1 is {cropdata.get('TSUM1')}, "
                "seems unusual for sugarbeet. Expected 500-1200 C.d.")
        if not (0.3 < cropdata.get('CFET', 0) < 1.3):
            logging.warning(
                f"    [WARNING] CFET (Crop Factor) is {cropdata.get('CFET')}. "
                "Typical range is 0.3-1.3, check for plausibility.")

        logging.info(
            f"    [OK] Crop parameters loaded (e.g., TSUM1: "
            f"{cropdata.get('TSUM1')}, CFET: {cropdata.get('CFET')}).")

    except Exception as e:
        logging.error(
            f"    [FAIL] An error occurred during input analysis: {e}",
            exc_info=True)
        analysis_passed = False

    logging.info("=" * 80)
    if analysis_passed:
        logging.info("--- ANALYSIS V2 COMPLETE: Inputs seem plausible. ---")
    else:
        logging.error(
            "--- ANALYSIS V2 FAILED: Critical errors found in model inputs. ---")
    logging.info("=" * 80)
    return analysis_passed

def analyze_and_plot_ensemble_results(df_hist, df_fcst_ensemble, output_dir,
                                      start_year, end_year):
    """
    MODIFIED (v4): Re-added the ensemble save line, plus text-based logging.
    """
    logging.info(
        "=" * 70 + "\n[ANALYSIS v4] Starting Text-Based Debug Analysis\n" +
        "=" * 70)

    # --- 1. Load Config & *** SAVE RAW ENSEMBLE DATA *** ---
    dmc = CONFIG['CONSTANTS']['DMC_SUGARBEET']

    # *** THIS IS THE FIX: Re-added the save command ***
    if not df_fcst_ensemble.empty:
        fcst_output_path = output_dir / \
            f'forecast_ensemble_{start_year}-{end_year}.csv'
        df_fcst_ensemble.to_csv(fcst_output_path, index=False)
        logging.info(
            f"✓ Full forecast ensemble results saved to {fcst_output_path}")
    # *** END OF FIX ***

    if df_fcst_ensemble.empty or df_hist.empty:
        logging.error(
            "[ANALYSIS] No data in forecast or historical results. "
            "Cannot analyze.")
        return

    # --- 2. Convert ALL Yields to Fresh Weight (dt/ha) ---
    df_fcst_ensemble['yield_wlp_fresh_dt'] = \
        (df_fcst_ensemble['yield_water_limited_dry_kgha'] / dmc) / 100.0
    df_fcst_ensemble['yield_pp_fresh_dt'] = \
        (df_fcst_ensemble['yield_potential_dry_kgha'] / dmc) / 100.0
    df_hist['perfect_yield_dt'] = \
        (df_hist['lintul_yield_perfect_weather'] / dmc) / 100.0

    # --- 3. Aggregate Ensemble Data ---
    logging.info("[ANALYSIS] Aggregating ensemble data...")
    df_fcst_agg = df_fcst_ensemble.groupby(['year', 'district_no']).agg(
        forecast_yield_mean=('yield_wlp_fresh_dt', 'mean'),
        forecast_yield_p10=('yield_wlp_fresh_dt', lambda x: x.quantile(0.10)),
        forecast_yield_p90=('yield_wlp_fresh_dt', lambda x: x.quantile(0.90)),
        potential_yield_mean=('yield_pp_fresh_dt', 'mean'),
        sim_failure_rate=('simulation_failed', 'mean')
    ).reset_index()

    # --- 4. Merge Historical and Forecast Data ---
    df_final = pd.merge(
        df_hist[['year', 'district_no', 'actual_yield', 'perfect_yield_dt']],
        df_fcst_agg,
        on=['year', 'district_no']
    )

    # --- 5. START OF TEXT-BASED ANALYSIS ---
    logging.info("\n" + "=" * 80)
    logging.info("--- DETAILED TEXT-BASED ANALYSIS ---")
    logging.info(f"Analysis for: {start_year}-{end_year}")
    logging.info(f"Dry Matter Content (DMC_SUGARBEET) used for conversion: {dmc}")

    logging.info("\n[DEBUG] 1. Checking for raw 'perfect_yield_dt' from historical run:")
    logging.info(df_hist['perfect_yield_dt'].describe().to_string())
    nan_count_hist = df_hist['perfect_yield_dt'].isna().sum()
    logging.info(f"Historical NaNs: {nan_count_hist} / {len(df_hist)}")

    logging.info("\n[DEBUG] 2. Checking for raw 'forecast_yield_mean' from ensemble:")
    logging.info(df_fcst_agg['forecast_yield_mean'].describe().to_string())

    logging.info("\n[DEBUG] 3. Final Merged DataFrame (Head):")
    logging.info(df_final.head(15).to_string())

    logging.info("\n[DEBUG] 4. Final Merged DataFrame (Statistical Summary BEFORE dropna):")
    logging.info(df_final.describe().to_string())

    # --- 6. Final Data Integrity Check ---
    logging.info("\n[DEBUG] 5. Final Data Integrity & Counts:")
    logging.info(f"Total rows in df_hist: {len(df_hist)}")
    logging.info(f"Total rows in df_fcst_agg: {len(df_fcst_agg)}")
    logging.info(f"Total rows in df_final (after merge): {len(df_final)}")

    df_final_clean = df_final.dropna(
        subset=['actual_yield', 'perfect_yield_dt', 'forecast_yield_mean'])
    logging.info(
        f"Total rows in df_final (after dropna): {len(df_final_clean)}")

    logging.info("=" * 80 + "\n")
    # --- END OF TEXT-BASED ANALYSIS ---

    if df_final_clean.empty:
        logging.error(
            "[ANALYSIS] FINAL FAILURE: No valid, non-NaN merged results "
            "were found.")
        return

    # --- 7. Calculate & Print Metrics (on cleaned data) ---
    mae_p = mean_absolute_error(df_final_clean['actual_yield'],
                                df_final_clean['perfect_yield_dt'])
    r2_p = r2_score(df_final_clean['actual_yield'],
                    df_final_clean['perfect_yield_dt'])
    mae_f = mean_absolute_error(df_final_clean['actual_yield'],
                                df_final_clean['forecast_yield_mean'])
    r2_f = r2_score(df_final_clean['actual_yield'],
                    df_final_clean['forecast_yield_mean'])

    print(
        "\n--- Overall Performance Metrics (Fresh Weight dt/ha, "
        "based on Ensemble Mean) ---")
    print(f"  Perfect Weather (WLP):  MAE = {mae_p:.2f}, R² = {r2_p:.3f}")
    print(f"  Forecast Weather (WLP): MAE = {mae_f:.2f}, R² = {r2_f:.3f}\n")

    # --- 8. Create Diagnostic Plots (only if data is valid) ---
    logging.info("[ANALYSIS] Generating diagnostic plots (if any valid data exists)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle(f'WOFOST Performance Diagnosis ({start_year}-{end_year})', fontsize=16)

    min_val_data = df_final_clean[
        ['actual_yield', 'perfect_yield_dt', 'forecast_yield_p10', 'potential_yield_mean']].min()
    max_val_data = df_final_clean[
        ['actual_yield', 'perfect_yield_dt', 'forecast_yield_p90', 'potential_yield_mean']].max()
    min_val = min_val_data.min() * 0.95
    max_val = max_val_data.max() * 1.05

    # === PLOT 1: PERFECT WEATHER ===
    axes[0].scatter(df_final_clean['actual_yield'],
                    df_final_clean['perfect_yield_dt'], alpha=0.6,
                    label='Simulated (Perfect Weather)')
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--',
                 label='1:1 Line')
    axes[0].scatter(df_final_clean['actual_yield'],
                    df_final_clean['potential_yield_mean'],
                    marker='x', color='red', alpha=0.5,
                    label='Mean Potential Yield (PP)')
    axes[0].set_title(f'Perfect Weather\nMAE={mae_p:.2f}, R²={r2_p:.3f}')
    axes[0].set_xlabel('Actual Yield (dt/ha)')
    axes[0].set_ylabel('Simulated Yield (dt/ha)')

    # === PLOT 2: FORECAST WEATHER (ENSEMBLE) ===
    lower_error = df_final_clean['forecast_yield_mean'] - df_final_clean['forecast_yield_p10']
    upper_error = df_final_clean['forecast_yield_p90'] - \
        df_final_clean['forecast_yield_mean']
    y_err = [lower_error.values, upper_error.values]

    axes[1].errorbar(df_final_clean['actual_yield'],
                     df_final_clean['forecast_yield_mean'], yerr=y_err,
                     fmt='o', color='orange', ecolor='lightgray',
                     elinewidth=3, capsize=0, alpha=0.8,
                     label='Ensemble Mean & 10-90th Pct. Range')
    axes[1].plot([min_val, max_val], [min_val, max_val],
                 'r--', label='1:1 Line')
    axes[1].scatter(df_final_clean['actual_yield'],
                    df_final_clean['potential_yield_mean'],
                    marker='x', color='red', alpha=0.5,
                    label='Mean Potential Yield (PP)')
    axes[1].set_title(
        f'Forecast Weather (Ensemble Range)\n'
        f'Mean MAE={mae_f:.2f}, Mean R²={r2_f:.3f}')
    axes[1].set_xlabel('Actual Yield (dt/ha)')

    # --- 9. Finalize and Save Plot ---
    for ax in axes:
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / \
        f'results_scatter_with_POTENTIAL_{start_year}-{end_year}.png'
    plt.savefig(plot_path, dpi=300)
    logging.info(
        f"[ANALYSIS] ✓ Diagnostic plot (with potential yield) saved to "
        f"{plot_path}")
    plt.show()


def aggregate_and_save_extreme_weather_metrics(df_fcst_ensemble, output_path):
    """
    Calculates and saves the distributional features for the new extreme
    weather and drought stress metrics.
    """
    logging.info(
        "=" * 70 +
        "\n[ANALYSIS] Aggregating new in-season, weather, and drought stress "
        "risk features...\n" + "=" * 70)
    if df_fcst_ensemble.empty:
        logging.warning(
            "[ANALYSIS] Forecast ensemble dataframe is empty. "
            "Skipping extreme metrics.")
        return

    # Define aggregation functions
    aggs = {
        'consecutive_tmax_gt_30c': [
            'mean', 'std', lambda x: x.quantile(0.90),
            lambda x: (x > 10).mean()
        ],
        'consecutive_dry_days': [
            'mean', 'std', lambda x: x.quantile(0.90),
            lambda x: (x > 21).mean()
        ],
        'drought_stress_index': [
            'mean', 'std', lambda x: x.quantile(0.90),
            lambda x: (x > 0.5).mean()
        ],
        'simulation_failed': ['mean'],
        'days_to_anthesis': ['mean', 'std', lambda x: x.quantile(0.90)],
        'max_lai_achieved': [
            'mean', 'std', lambda x: x.quantile(0.10)
        ],
        'cumulative_water_stress': [
            'mean', 'std', lambda x: x.quantile(0.90)
        ]
    }

    # Perform aggregation
    df_extreme_metrics = df_fcst_ensemble.groupby(
        ['year', 'district_no']).agg(aggs).reset_index()

    # Flatten the multi-level column names
    df_extreme_metrics.columns = [
        'year', 'district_no',
        'mean_consecutive_days_above_30c',
        'std_dev_consecutive_days_above_30c',
        'p90_consecutive_days_above_30c', 'prob_heatwave_gt_10_days',
        'mean_consecutive_dry_days', 'std_dev_consecutive_dry_days',
        'p90_consecutive_dry_days', 'prob_drought_spell_gt_21_days',
        'mean_drought_stress_index', 'std_drought_stress_index',
        'p90_drought_stress_index', 'prob_severe_drought_stress',
        'prob_simulation_failure',
        'mean_days_to_anthesis', 'std_days_to_anthesis',
        'p90_days_to_anthesis',
        'mean_max_lai_achieved', 'std_max_lai_achieved',
        'p10_max_lai_achieved',
        'mean_cumulative_water_stress', 'std_cumulative_water_stress',
        'p90_cumulative_water_stress'
    ]

    # Save the aggregated metrics to a new CSV file
    df_extreme_metrics.to_csv(output_path, index=False)
    logging.info(f"[ANALYSIS] ✓ All risk features saved to {output_path}")
