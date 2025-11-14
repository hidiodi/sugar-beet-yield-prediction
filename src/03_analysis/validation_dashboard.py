# File: src/03_analysis/validation_dashboard.py
# Description: A validation dashboard to analyze the inputs and outputs of the refactored WOFOST pipeline.
# ENHANCEMENT 2: Added Phase 4 for spatial analysis.

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging
import geopandas as gpd

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
OUTPUT_DIR = project_root / "reports" / "figures" / "validation_dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

def phase_1_input_sanity_checks(genes_path, initial_conditions_path):
    """
    Performs Phase 1 checks on the input data assets.
    """
    logging.info("--- Phase 1: Running Input Data Sanity Checks ---")

    # 1. Genetic Gain Plot
    logging.info("Generating Genetic Gain Plot...")
    with open(genes_path, 'r') as f:
        genes_data = json.load(f)
    df_genes = pd.DataFrame.from_dict(genes_data, orient='index')
    df_genes.index = df_genes.index.astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Genetic Gain Verification (AMAX, TSUM1, RUE vs. Year)', fontsize=16)

    sns.lineplot(data=df_genes, x=df_genes.index, y='AMAX', ax=axes[0], marker='o')
    axes[0].set_title('AMAX vs. Year')
    axes[0].set_ylabel('AMAX')

    sns.lineplot(data=df_genes, x=df_genes.index, y='TSUM1', ax=axes[1], marker='o')
    axes[1].set_title('TSUM1 vs. Year')
    axes[1].set_ylabel('TSUM1')

    sns.lineplot(data=df_genes, x=df_genes.index, y='RUE', ax=axes[2], marker='o')
    axes[2].set_title('RUE vs. Year')
    axes[2].set_ylabel('RUE')
    axes[2].set_xlabel('Year')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = OUTPUT_DIR / "P1_genetic_gain_plot.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved plot to {plot_path}")

    # 2. Initial Conditions Distribution
    logging.info("Generating Initial Conditions Distribution Plots...")
    df_ic = pd.read_csv(initial_conditions_path, parse_dates=['sowing_date'])
    df_ic['sowing_date_doy'] = df_ic['sowing_date'].dt.dayofyear

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Initial Conditions Distribution (WAV, TDWI, Sowing Date)', fontsize=16)

    sns.histplot(df_ic['WAV'], kde=True, ax=axes[0])
    axes[0].set_title('Distribution of Initial Soil Moisture (WAV)')
    axes[0].set_xlabel('WAV (cm)')

    sns.histplot(df_ic['TDWI'], kde=True, ax=axes[1])
    axes[1].set_title('Distribution of Initial Dry Weight (TDWI)')
    axes[1].set_xlabel('TDWI (kg/ha)')

    sns.histplot(df_ic['sowing_date_doy'], kde=True, ax=axes[2])
    axes[2].set_title('Distribution of Sowing Date (Day of Year)')
    axes[2].set_xlabel('Day of Year')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = OUTPUT_DIR / "P1_initial_conditions_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved plot to {plot_path}")
    return df_genes, df_ic

def phase_2_output_sanity_checks(wofost_output_path):
    """
    Performs Phase 2 checks on the output data.
    """
    logging.info("--- Phase 2: Running Output Data Sanity Checks ---")

    try:
        df_wofost = pd.read_csv(wofost_output_path)
    except FileNotFoundError:
        logging.error(f"FATAL: WOFOST output file not found at {wofost_output_path}. Cannot run Phase 2 & 3 checks.")
        return None

    # 1. Aggregate and convert yield
    df_agg = df_wofost.groupby(['year', 'district_no']).agg(
        yield_water_limited_dry_kgha=('yield_water_limited_dry_kgha', 'mean'),
        yield_potential_dry_kgha=('yield_potential_dry_kgha', 'mean'),
        drought_stress_index=('drought_stress_index', 'mean')
    ).reset_index()

    dmc = config.WOFOST_CONFIG['CONSTANTS']['DMC_SUGARBEET']
    df_agg['wofost_forecast_yield_fresh_dt'] = (df_agg['yield_water_limited_dry_kgha'] / dmc) / 100.0

    # 2. Output Yield Distribution Plot
    logging.info("Generating Output Yield Distribution Plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df_agg['wofost_forecast_yield_fresh_dt'], kde=True)
    plt.title('Distribution of "Pure WOFOST" Mean Forecast Yield')
    plt.xlabel('Yield (Fresh Weight dt/ha)')
    plt.ylabel('Frequency')
    plot_path = OUTPUT_DIR / "P2_output_yield_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved plot to {plot_path}")

    # 3. Missing Data Report
    logging.info("Generating Missing Data Report...")
    missing_yields = df_agg['wofost_forecast_yield_fresh_dt'].isnull().sum()
    total_records = len(df_agg)
    logging.info(f"Missing Data Report: {missing_yields} / {total_records} ('wofost_forecast_yield_fresh_dt') are null.")
    if missing_yields > 0:
        logging.warning("WARNING: Missing yield data detected in the aggregated WOFOST output.")

    return df_agg

def phase_3_model_behavior_validation(df_genes, df_ic, df_agg_wofost):
    """
    Performs Phase 3 checks on the model's behavior (input vs. output).
    """
    if df_agg_wofost is None:
        logging.error("Skipping Phase 3 due to missing WOFOST output data.")
        return

    logging.info("--- Phase 3: Running Model Behavior Validation ---")

    # 1. The Water Stress Test
    logging.info("Generating Water Stress Test Plot...")
    df_merged_ic = pd.merge(df_ic, df_agg_wofost, on=['year', 'district_no'])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_merged_ic, x='WAV', y='drought_stress_index')
    plt.title('Water Stress Test: Initial Water vs. Drought Stress')
    plt.xlabel('Initial Available Water (WAV, cm)')
    plt.ylabel('Drought Stress Index')
    plot_path = OUTPUT_DIR / "P3_water_stress_test.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved plot to {plot_path}")

    # 2. The Genetic Gain Test
    logging.info("Generating Genetic Gain Test Plot...")
    df_genes_reset = df_genes.reset_index().rename(columns={'index': 'year'})
    df_merged_genes = pd.merge(df_genes_reset, df_agg_wofost, on='year')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_merged_genes, x='AMAX', y='yield_potential_dry_kgha')
    plt.title('Genetic Gain Test: AMAX vs. Potential Yield')
    plt.xlabel('AMAX (kg CO2/ha/hr)')
    plt.ylabel('Potential Yield (Dry kg/ha)')
    plot_path = OUTPUT_DIR / "P3_genetic_gain_test.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved plot to {plot_path}")

    # 3. The Sowing Date Test
    logging.info("Generating Sowing Date Test Plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_merged_ic, x='sowing_date_doy', y='wofost_forecast_yield_fresh_dt')
    plt.title('Sowing Date Test: Sowing Date vs. Final Yield')
    plt.xlabel('Sowing Date (Day of Year)')
    plt.ylabel('Final Yield (Fresh dt/ha)')
    plot_path = OUTPUT_DIR / "P3_sowing_date_test.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved plot to {plot_path}")

def phase_4_spatial_analysis(df_agg_wofost, df_ic):
    """
    Performs Phase 4 checks by creating spatial plots (choropleth maps).
    """
    if df_agg_wofost is None or df_ic is None:
        logging.error("Skipping Phase 4 due to missing data.")
        return

    logging.info("--- Phase 4: Running Spatial Analysis ---")

    try:
        gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH).rename(columns={'id': 'district_no'})
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    except FileNotFoundError:
        logging.error("FATAL: GeoJSON file for districts not found. Cannot run spatial analysis.")
        return

    # Average metrics over all years for a stable map
    df_spatial_yield = df_agg_wofost.groupby('district_no')['wofost_forecast_yield_fresh_dt'].mean().reset_index()
    df_spatial_stress = df_agg_wofost.groupby('district_no')['drought_stress_index'].mean().reset_index()
    df_ic['sowing_date_doy'] = df_ic['sowing_date'].dt.dayofyear
    df_spatial_sowing = df_ic.groupby('district_no')['sowing_date_doy'].mean().reset_index()

    gdf_yield = pd.merge(gdf_districts, df_spatial_yield, on='district_no')
    gdf_stress = pd.merge(gdf_districts, df_spatial_stress, on='district_no')
    gdf_sowing = pd.merge(gdf_districts, df_spatial_sowing, on='district_no')

    # Plot 1: Mean Forecasted Yield
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf_yield.plot(column='wofost_forecast_yield_fresh_dt', ax=ax, legend=True, cmap='viridis',
                 legend_kwds={'label': "Mean Yield (Fresh dt/ha)", 'orientation': "horizontal"})
    ax.set_title('Mean Forecasted Yield by District')
    ax.set_axis_off()
    plot_path = OUTPUT_DIR / "P4_map_mean_yield.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Saved map to {plot_path}")

    # Plot 2: Mean Drought Stress
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf_stress.plot(column='drought_stress_index', ax=ax, legend=True, cmap='plasma',
                  legend_kwds={'label': "Mean Drought Stress Index", 'orientation': "horizontal"})
    ax.set_title('Mean Drought Stress by District')
    ax.set_axis_off()
    plot_path = OUTPUT_DIR / "P4_map_drought_stress.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Saved map to {plot_path}")

    # Plot 3: Mean Sowing Date
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf_sowing.plot(column='sowing_date_doy', ax=ax, legend=True, cmap='cividis',
                  legend_kwds={'label': "Mean Sowing Date (Day of Year)", 'orientation': "horizontal"})
    ax.set_title('Mean Sowing Date by District')
    ax.set_axis_off()
    plot_path = OUTPUT_DIR / "P4_map_sowing_date.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Saved map to {plot_path}")

def main():
    """
    Main function to run the validation dashboard.
    """
    logging.info("--- Starting Validation Dashboard ---")

    # Define file paths
    genes_path = PROCESSED_DATA_DIR / 'SugarbeetGenes.json'
    initial_conditions_path = PROCESSED_DATA_DIR / 'InitialConditions.csv'
    wofost_output_path = config.WOFOST_CONFIG['FILE_PATHS']['OUTPUT_DIR'] / f"forecast_ensemble_{config.WOFOST_CONFIG['START_YEAR']}-{config.WOFOST_CONFIG['END_YEAR']}.csv"

    logging.info(f"Output will be saved to: {OUTPUT_DIR}")

    # --- Run Phases ---
    df_genes, df_ic = phase_1_input_sanity_checks(genes_path, initial_conditions_path)
    df_agg_wofost = phase_2_output_sanity_checks(wofost_output_path)
    phase_3_model_behavior_validation(df_genes, df_ic, df_agg_wofost)
    phase_4_spatial_analysis(df_agg_wofost, df_ic)

    logging.info("--- Validation Dashboard Complete ---")

if __name__ == "__main__":
    main()
