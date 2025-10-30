# File: src/analysis/analyze_pipeline_stages.py
# Description: A comprehensive diagnostic script that analyzes each major stage of the
#              agricultural forecasting pipeline.
#
# REVISED VERSION 3: This version performs a full, data-leak-proof, walk-forward
# validation for its analysis. It no longer uses a "perfect" retrospective trend,
# ensuring the analysis honestly reflects the performance of the actual forecasting process.

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
import numpy as np
from pygam import LinearGAM, s
from tqdm import tqdm

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# --- Input Files ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURES_FILE = PROJECT_ROOT / 'data/05_model_input/stage1_preseason_features.csv'
WEATHER_DIR = PROJECT_ROOT / 'data/02_intermediate/daily_weather'
GEOJSON_FILE = PROJECT_ROOT / 'data/01_raw/districts_official.geojson'
RAW_WOFOST_OUTPUT_FILE = PROJECT_ROOT / 'data/06_model_output/multi_year_final/final_comparison_1981-2024.csv'
SUGARBEET_YIELD_FILE = PROJECT_ROOT / 'data/02_intermediate/sugarbeet_yield.csv'
DMC_SUGAR_BEET = 0.25
MIN_TRAIN_YEARS = 10  # Minimum years of data required before making a trend prediction

# --- Output Directory ---
OUTPUT_DIR = Path('reports/figures/pipeline_stage_analysis')


# --- Helper & Core Logic Functions ---
def setup_plot_style():
    """Sets a consistent and readable style for all plots."""
    sns.set_style("whitegrid")
    plt.rcParams.update({'figure.figsize': (20, 10), 'axes.titlesize': 22, 'axes.labelsize': 18, 'xtick.labelsize': 14,
                         'ytick.labelsize': 14, 'legend.fontsize': 16, 'figure.titlesize': 28})


def load_data_safely(file_path, is_geojson=False):
    """Loads a file with standardized error handling."""
    logging.info(f"Loading {file_path.name}...")
    if not file_path.exists():
        logging.error(f"FATAL: Required file not found at {file_path}");
        return None
    try:
        return gpd.read_file(file_path) if is_geojson else pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"FATAL: Could not load or read {file_path}. Error: {e}");
        return None


def perform_walk_forward_detrending(df_yield):
    """
    Calculates the technology trend and weather residual using a strict, leak-proof
    walk-forward methodology, mirroring the actual forecasting process.
    """
    logging.info("Performing honest, walk-forward detrending on all districts...")
    all_honest_trends = []

    for district, group in tqdm(df_yield.groupby('district_no'), desc="Walk-Forward Detrending"):
        group = group.sort_values('year').reset_index(drop=True)
        if len(group) < MIN_TRAIN_YEARS:
            continue

        for i in range(MIN_TRAIN_YEARS, len(group)):
            train_df = group.iloc[:i]
            current_row = group.iloc[i]

            x_train, y_train = train_df['year'].values, train_df['yield'].values
            current_year = current_row['year']

            try:
                # Fit model ONLY on past data
                gam = LinearGAM(s(0, n_splines=8)).fit(x_train, y_train)

                # Predict trend for the single, current year
                y_tech_forecast = gam.predict([current_year])[0]

                all_honest_trends.append({
                    'district_no': district,
                    'year': current_year,
                    'yield': current_row['yield'],
                    'y_tech_honest': y_tech_forecast,
                    'y_weather_honest': current_row['yield'] - y_tech_forecast
                })
            except Exception as e:
                logging.warning(f"Could not create trend forecast for {district} in {current_year}: {e}")
                continue

    logging.info("✓ Walk-forward detrending complete.")
    return pd.DataFrame(all_honest_trends)


# ==============================================================================
# === PART 1: ANALYSIS OF RAW INPUTS (Unaffected by change) ===
# ==============================================================================
def part_1_analyze_raw_inputs(weather_df, features_df, gdf, output_dir):
    """Analyzes the spatial variability and characteristics of the primary model inputs."""
    logging.info("\n" + "=" * 60 + "\n--- PART 1: Analyzing Raw Model Inputs ---\n" + "=" * 60)
    # This part remains the same as it analyzes raw inputs.
    annual_weather = weather_df.groupby('district_no').agg(total_precip_mm=('precip', lambda x: x.sum() * 86400),
                                                           mean_temp_c=(
                                                           'tmax', lambda x: (x - 273.15).mean())).reset_index()
    gdf_weather = gdf.merge(annual_weather, left_on='id', right_on='district_no', how='left')
    fig, axes = plt.subplots(1, 2, figsize=(24, 12));
    fig.suptitle(f'Spatial Variability of Raw Weather Inputs (Sample Year: {weather_df.date.dt.year.iloc[0]})', y=0.95)
    gdf_weather.plot(column='total_precip_mm', ax=axes[0], legend=True, cmap='viridis',
                     legend_kwds={'label': "Total Annual Precipitation (mm)", "shrink": 0.6});
    axes[0].set_title("Annual Precipitation");
    axes[0].set_axis_off()
    gdf_weather.plot(column='mean_temp_c', ax=axes[1], legend=True, cmap='plasma',
                     legend_kwds={'label': "Mean Annual Temperature (°C)", "shrink": 0.6});
    axes[1].set_title("Annual Mean Temperature");
    axes[1].set_axis_off()
    plt.savefig(output_dir / '1a_raw_input_weather_maps.png', dpi=300, bbox_inches='tight');
    plt.close()
    gdf_soil = gdf.merge(features_df[['district_no', 'avg_sand_0_100cm', 'avg_clay_0_100cm']].drop_duplicates(),
                         left_on='id', right_on='district_no', how='left')
    fig, axes = plt.subplots(1, 2, figsize=(24, 12));
    fig.suptitle('Spatial Variability of Raw Soil Inputs', y=0.95)
    gdf_soil.plot(column='avg_sand_0_100cm', ax=axes[0], legend=True, cmap='YlOrBr',
                  legend_kwds={'label': "Sand Content (%)", "shrink": 0.6});
    axes[0].set_title("Sand Content (0-100cm)");
    axes[0].set_axis_off()
    gdf_soil.plot(column='avg_clay_0_100cm', ax=axes[1], legend=True, cmap='Blues',
                  legend_kwds={'label': "Clay Content (%)", "shrink": 0.6});
    axes[1].set_title("Clay Content (0-100cm)");
    axes[1].set_axis_off()
    plt.savefig(output_dir / '1b_raw_input_soil_maps.png', dpi=300, bbox_inches='tight');
    plt.close()
    logging.info("--- Part 1 Complete ---")


# ==============================================================================
# === PART 2: ANALYSIS OF RAW WOFOST OUTPUT (Unaffected by change) ===
# ==============================================================================
def part_2_analyze_raw_wofost(df_raw_wofost, gdf, output_dir):
    """Analyzes the performance of the WOFOST model before any detrending or correction."""
    logging.info("\n" + "=" * 60 + "\n--- PART 2: Analyzing Raw WOFOST Output ---\n" + "=" * 60)
    # This part remains the same as it analyzes the raw model output before detrending.
    df_raw_wofost['Error_Raw_vs_Actual'] = df_raw_wofost['actual_yield_dt'] - df_raw_wofost['perfect_yield_dt']
    mean_bias = df_raw_wofost.groupby('district_no')['Error_Raw_vs_Actual'].mean()
    gdf_bias = gdf.merge(mean_bias, left_on='id', right_on='district_no', how='left')
    vmax = abs(gdf_bias['Error_Raw_vs_Actual']).quantile(0.95)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12));
    gdf_bias.plot(column='Error_Raw_vs_Actual', ax=ax, legend=True, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  missing_kwds={"color": "lightgrey"},
                  legend_kwds={'label': "Average Error (Actual - Simulated) [dt/ha]", "shrink": 0.6})
    ax.set_title("Systematic Bias of Raw WOFOST Simulation (All Years)");
    ax.set_axis_off();
    plt.savefig(output_dir / '2a_raw_wofost_bias_map.png', dpi=300, bbox_inches='tight');
    plt.close()
    fig, ax = plt.subplots(figsize=(12, 8));
    sns.scatterplot(data=df_raw_wofost, x='actual_yield_dt', y='perfect_yield_dt', hue='year', palette='viridis', ax=ax,
                    s=20, alpha=0.6)
    min_val, max_val = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]);
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    ax.set_title("Raw WOFOST vs. Actual Yield: The Technology Trend Problem");
    ax.set_xlabel("Actual Yield (dt/ha)");
    ax.set_ylabel("Raw Simulated Yield (dt/ha)");
    ax.legend()
    plt.savefig(output_dir / '2b_raw_wofost_scatter.png', dpi=300, bbox_inches='tight');
    plt.close()
    logging.info("--- Part 2 Complete ---")


# ==============================================================================
# === PART 3: REVISED ANALYSIS OF THE DETRENDING PROCESS ===
# ==============================================================================
def part_3_analyze_detrending(df_honest_trends, output_dir):
    """Visualizes the HONEST, walk-forward detrending process."""
    logging.info("\n" + "=" * 60 + "\n--- PART 3: Analyzing the HONEST Detrending Process ---\n" + "=" * 60)
    logging.info("Purpose: To show how the trend is forecast year-by-year using only past data.")

    example_districts = df_honest_trends['district_no'].sample(n=4, random_state=42).tolist()
    df_examples = df_honest_trends[df_honest_trends['district_no'].isin(example_districts)]

    fig = plt.figure(figsize=(20, 14));
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2);
    axes = gs.subplots()
    fig.suptitle('The HONEST Detrending Process: One-Year-Ahead Trend Forecasts', y=0.95)

    for ax, district in zip(axes.flatten(), example_districts):
        subset = df_examples[df_examples['district_no'] == district]
        ax.plot(subset['year'], subset['yield'], 'o-', label='Observed Yield', markersize=4, alpha=0.7, zorder=1)
        ax.plot(subset['year'], subset['y_tech_honest'], 'o', color='red', label='Trend Forecast (Honest)',
                markersize=5, zorder=2)
        ax.set_title(f'District: {district}');
        ax.set_xlabel('Year');
        ax.set_ylabel('Yield (dt/ha)');
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.savefig(output_dir / '3a_detrending_examples_HONEST.png', dpi=300, bbox_inches='tight');
    plt.close()
    logging.info("✓ Generated plots demonstrating the HONEST walk-forward detrending process.")
    logging.info("--- Part 3 Complete ---")


# ==============================================================================
# === PART 4: REVISED ANALYSIS OF THE FINAL, CORRECTED OUTPUT ===
# ==============================================================================
def part_4_analyze_final_output(df_raw_wofost, df_features, df_honest_trends, gdf, output_dir):
    """Performs a definitive comparison using the HONEST, walk-forward detrended data."""
    logging.info(
        "\n" + "=" * 60 + "\n--- PART 4: Analyzing the Final Corrected Output (HONEST Validation) ---\n" + "=" * 60)

    # Merge all data sources. The key is to merge with df_honest_trends.
    df_analysis = pd.merge(df_raw_wofost[['year', 'district_no', 'perfect_yield_dt']], df_features,
                           on=['year', 'district_no'], how='inner')
    df_analysis = pd.merge(df_analysis, df_honest_trends, on=['year', 'district_no'], how='inner')

    # Define the Ground Truth: The HONEST weather-driven residual
    # This is now 'y_weather_honest' from our walk-forward calculation.
    df_analysis.rename(columns={'y_weather_honest': 'Actual_Yield_Detrended'}, inplace=True)

    # Define predictors
    df_analysis.rename(columns={'perfect_yield_dt': 'Simulated_Yield_Raw'}, inplace=True)
    df_analysis['Simulated_Yield_Detrended'] = df_analysis['wofost_forecast_yield_fresh_dt'] * DMC_SUGAR_BEET

    # Define errors against the HONEST ground truth
    df_analysis['Error_Raw_Simulation'] = df_analysis['Actual_Yield_Detrended'] - df_analysis['Simulated_Yield_Raw']
    df_analysis['Error_Detrended_Simulation'] = df_analysis['Actual_Yield_Detrended'] - df_analysis[
        'Simulated_Yield_Detrended']
    df_analysis.dropna(inplace=True)
    logging.info(f"✓ Master analysis DataFrame created with {len(df_analysis)} HONEST records.")

    # --- Performance Map Comparison (Correlation) ---
    corr_raw = df_analysis.groupby('district_no')[
                   ['Simulated_Yield_Raw', 'Actual_Yield_Detrended']].corr().unstack().iloc[:, 1].rename('correlation')
    corr_detrended = df_analysis.groupby('district_no')[
                         ['Simulated_Yield_Detrended', 'Actual_Yield_Detrended']].corr().unstack().iloc[:, 1].rename(
        'correlation')
    gdf_raw_corr = gdf.merge(corr_raw, left_on='id', right_on='district_no', how='left')
    gdf_detrended_corr = gdf.merge(corr_detrended, left_on='id', right_on='district_no', how='left')
    fig, axes = plt.subplots(1, 2, figsize=(24, 12));
    fig.suptitle('HONEST Performance: Correlation with Weather-Driven Yield', y=0.95)
    gdf_raw_corr.plot(column='correlation', ax=axes[0], legend=True, cmap='coolwarm', vmin=-0.5, vmax=0.5);
    axes[0].set_title("BEFORE Correction\n(Raw WOFOST vs. Honest Weather-Yield)");
    axes[0].set_axis_off()
    gdf_detrended_corr.plot(column='correlation', ax=axes[1], legend=True, cmap='coolwarm', vmin=-0.5, vmax=0.5);
    axes[1].set_title("AFTER Correction\n(Final Feature vs. Honest Weather-Yield)");
    axes[1].set_axis_off()
    plt.savefig(output_dir / '4a_comparative_correlation_map_HONEST.png', dpi=300, bbox_inches='tight');
    plt.close()
    logging.info("✓ Generated HONEST comparative correlation maps.")

    # --- Error Bias Map Comparison (Centered) ---
    bias_raw = df_analysis.groupby('district_no')['Error_Raw_Simulation'].mean()
    bias_detrended = df_analysis.groupby('district_no')['Error_Detrended_Simulation'].mean()
    gdf_bias_raw = gdf.merge((bias_raw - bias_raw.mean()).rename('Error_Centered'), left_on='id',
                             right_on='district_no', how='left')
    gdf_bias_detrended = gdf.merge((bias_detrended - bias_detrended.mean()).rename('Error_Centered'), left_on='id',
                                   right_on='district_no', how='left')
    max_abs_bias = abs(pd.concat([gdf_bias_raw['Error_Centered'], gdf_bias_detrended['Error_Centered']])).quantile(0.95)
    fig, axes = plt.subplots(1, 2, figsize=(24, 12));
    fig.suptitle('HONEST Error Bias: Does Correction Reduce Systematic Error?', y=0.95)
    gdf_bias_raw.plot(column='Error_Centered', ax=axes[0], legend=True, cmap='RdBu_r', vmin=-max_abs_bias,
                      vmax=max_abs_bias);
    axes[0].set_title("BEFORE Correction\n(Error of Raw Simulation)");
    axes[0].set_axis_off()
    gdf_bias_detrended.plot(column='Error_Centered', ax=axes[1], legend=True, cmap='RdBu_r', vmin=-max_abs_bias,
                            vmax=max_abs_bias);
    axes[1].set_title("AFTER Correction\n(Error of Final Feature)");
    axes[1].set_axis_off()
    plt.savefig(output_dir / '4b_comparative_error_bias_map_HONEST.png', dpi=300, bbox_inches='tight');
    plt.close()
    logging.info("✓ Generated HONEST comparative error bias maps.")
    logging.info("--- Part 4 Complete ---")


# ==============================================================================
# === MAIN ORCHESTRATION FUNCTION ===
# ==============================================================================
def main():
    """Main function to load all data and run the four-part analysis pipeline."""
    logging.info("=" * 70 + "\n   STARTING COMPREHENSIVE & HONEST PIPELINE STAGE ANALYSIS\n" + "=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    # --- Load all necessary raw data sources ---
    gdf = load_data_safely(GEOJSON_FILE, is_geojson=True)
    df_features = load_data_safely(FEATURES_FILE)
    df_raw_wofost = load_data_safely(RAW_WOFOST_OUTPUT_FILE)
    df_yield = load_data_safely(SUGARBEET_YIELD_FILE)
    df_weather = load_data_safely(WEATHER_DIR / 'historical_daily_weather_era5_2018.csv')
    if df_weather is not None: df_weather['date'] = pd.to_datetime(df_weather['date'])

    if any(df is None for df in [gdf, df_features, df_raw_wofost, df_yield, df_weather]):
        logging.error("One or more essential data files failed to load. Aborting analysis.");
        return

    for df in [gdf, df_features, df_raw_wofost, df_yield, df_weather]:
        col = 'id' if 'id' in df.columns else 'district_no'
        df[col] = df[col].astype(str).str.zfill(5)

    # --- CORE LOGIC: Generate the HONEST, leak-proof trend data FIRST ---
    df_honest_trends = perform_walk_forward_detrending(df_yield)
    if df_honest_trends.empty:
        logging.error("Walk-forward detrending produced no data. Cannot proceed with analysis.");
        return

    # --- Run Analysis Parts Sequentially with the appropriate data ---
    part_1_analyze_raw_inputs(df_weather, df_features, gdf, OUTPUT_DIR)
    part_2_analyze_raw_wofost(df_raw_wofost, gdf, OUTPUT_DIR)
    part_3_analyze_detrending(df_honest_trends, OUTPUT_DIR)
    part_4_analyze_final_output(df_raw_wofost, df_features, df_honest_trends, gdf, OUTPUT_DIR)

    logging.info(
        "\n" + "=" * 70 + "\n   ✅ HONEST PIPELINE STAGE ANALYSIS COMPLETED SUCCESSFULLY!\n" + f"   All plots saved to: {OUTPUT_DIR.resolve()}\n" + "=" * 70)


if __name__ == "__main__":
    main()