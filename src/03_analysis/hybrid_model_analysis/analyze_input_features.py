# File: src/analysis/analyze_input_features.py
# Description: A dedicated script for the deep analysis of the input features generated
#              by the feature engineering pipeline. This script explains the "why" behind
#              our features by exploring their distributions, relationships, and raw
#              correlation with the target variable, WITHOUT using a trained model.

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Input Files ---
FEATURES_FILE = Path('data/05_model_input/stage1_preseason_features.csv')
GEOJSON_FILE = Path('data/01_raw/districts_official.geojson')

# --- Output Directory ---
OUTPUT_DIR = Path('reports/figures/feature_analysis_dashboard')


# --- Helper Function for Consistent Plotting ---
def setup_plot_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (14, 8), 'axes.titlesize': 20, 'axes.labelsize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 14
    })


# ==============================================================================
# === PART 1: THE LANDSCAPE (STATIC & ECONOMIC FEATURES) ===
# ==============================================================================
def part_1_analyze_landscape(df, gdf, output_dir):
    """Analyzes the static geographic and long-term economic context."""
    logging.info("--- 1. Analyzing the Static Landscape (Geography & Economics) ---")

    # Geospatial Feature Mapping (Sand, Clay, Elevation)
    df_static = df.groupby('district_no')[['avg_sand_0_30cm', 'avg_clay_0_30cm', 'avg_elevation']].mean().reset_index()
    gdf_merged = gdf.merge(df_static, left_on='id', right_on='district_no')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    gdf_merged.plot(column='avg_sand_0_30cm', ax=axes[0], legend=True, cmap='YlOrBr', legend_kwds={'label': "Sand %"});
    axes[0].set_title("Sand Content (0-30cm)")
    gdf_merged.plot(column='avg_clay_0_30cm', ax=axes[1], legend=True, cmap='Blues', legend_kwds={'label': "Clay %"});
    axes[1].set_title("Clay Content (0-30cm)")
    gdf_merged.plot(column='avg_elevation', ax=axes[2], legend=True, cmap='Greens',
                    legend_kwds={'label': "Elevation (m)"});
    axes[2].set_title("Average Elevation")
    for ax in axes: ax.set_axis_off()
    plt.suptitle("Geospatial Distribution of Key Static Features", fontsize=24)
    plt.savefig(output_dir / '1a_geospatial_feature_maps.png', dpi=300, bbox_inches='tight');
    plt.close()

    # Economic Context: The "Margin Squeeze"
    df_econ = df.groupby('year')[['producer_price_index_lag1', 'fertilizer_price_index_lag1']].mean().reset_index()
    plt.figure()
    plt.plot(df_econ['year'], df_econ['producer_price_index_lag1'], label='Producer Price Index (Lagged 1yr)',
             color='green', marker='o')
    plt.plot(df_econ['year'], df_econ['fertilizer_price_index_lag1'], label='Fertilizer Price Index (Lagged 1yr)',
             color='red', marker='x')
    plt.title("Economic Context: The Farmer's Margin Squeeze")
    plt.ylabel("Price Index (Base 100)")
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    plt.savefig(output_dir / '1b_economic_context.png', dpi=300, bbox_inches='tight');
    plt.close()

    logging.info("✓ Part 1 complete.")


# ==============================================================================
# === PART 2: THE TIME-SERIES HEARTBEAT (OUR STRONGEST PREDICTOR) ===
# ==============================================================================
def part_2_analyze_timeseries_forecast(df, output_dir):
    """
    Analyzes the raw predictive power of the time-series forecast feature.
    Hypothesis: The historical trend is the single best predictor of future yield.
    """
    logging.info("--- 2. Analyzing the Time-Series Heartbeat Feature ---")
    df_plot = df.dropna(subset=['wofost_forecast_yield_fresh_dt', 'kreisYield'])

    plt.figure()
    sns.regplot(x='kreisYield', y='wofost_forecast_yield_fresh_dt', data=df_plot,
                scatter_kws={'alpha': 0.2, 's': 10}, line_kws={'color': 'red'})

    r2 = df_plot[['kreisYield', 'wofost_forecast_yield_fresh_dt']].corr().iloc[0, 1] ** 2
    plt.title(f"Predictive Power of the Time-Series Forecast Feature\n(Raw Out-of-Sample R² ≈ {r2:.3f})")
    plt.xlabel("Actual Yield (dt/ha)")
    plt.ylabel("Time-Series Forecast Feature (dt/ha)")
    plt.savefig(output_dir / '2a_timeseries_forecast_power.png', dpi=300);
    plt.close()

    logging.info("✓ Part 2 complete.")


# ==============================================================================
# === PART 3: WEATHER & EXTREMES (BEYOND THE AVERAGES) ===
# ==============================================================================
def part_3_analyze_weather_features(df, output_dir):
    """
    Analyzes the relationship between weather features and yield.
    Hypothesis: Yield is sensitive to both seasonal anomalies and extreme events.
    """
    logging.info("--- 3. Analyzing Key Weather Features ---")

    # Binning to see the trend: How does yield change with summer temperature anomaly?
    df['temp_bin'] = pd.qcut(df['summer_temp_anomaly_forecast'], q=10, labels=False, duplicates='drop')
    df_binned_temp = df.groupby('temp_bin')['kreisYield'].mean().reset_index()
    # Get the average anomaly for each bin for a meaningful x-axis
    bin_labels = df.groupby('temp_bin')['summer_temp_anomaly_forecast'].mean()

    plt.figure()
    sns.barplot(x=df_binned_temp['temp_bin'], y=df_binned_temp['kreisYield'], color='coral')
    plt.title("Impact of Summer Temperature Anomaly on Average Yield")
    plt.xlabel("Summer Temperature Anomaly Forecast (°C)")
    plt.ylabel("Average Actual Yield (dt/ha)")
    plt.xticks(ticks=df_binned_temp['temp_bin'], labels=bin_labels.round(2), rotation=45)
    plt.savefig(output_dir / '3a_temp_anomaly_impact.png', dpi=300, bbox_inches='tight');
    plt.close()

    # Extreme Events: How does the number of heatwave days relate to yield?
    df_extreme = df.groupby('summer_days_tmax_gt_30c')['kreisYield'].mean().reset_index()
    df_extreme = df_extreme[df_extreme['summer_days_tmax_gt_30c'] <= 30]  # Filter for plausible range
    plt.figure()
    sns.lineplot(data=df_extreme, x='summer_days_tmax_gt_30c', y='kreisYield', marker='o')
    plt.title("Impact of Extreme Heatwave Days on Average Yield")
    plt.xlabel("Number of Summer Days with Temperature > 30°C")
    plt.ylabel("Average Actual Yield (dt/ha)")
    plt.savefig(output_dir / '3b_heatwave_impact.png', dpi=300, bbox_inches='tight');
    plt.close()

    logging.info("✓ Part 3 complete.")


# ==============================================================================
# === PART 4: INTERACTION HYPOTHESIS (TESTING SYNERGY IN THE DATA) ===
# ==============================================================================
def part_4_analyze_interaction_hypothesis(df, output_dir):
    """
    Tests the hypothesis that features interact, using the raw data.
    Hypothesis: The effect of a drought depends on soil type.
    """
    logging.info("--- 4. Testing an Interaction Hypothesis in Raw Data ---")

    # Create categories for soil type and precipitation
    df['sand_category'] = pd.qcut(df['avg_sand_0_30cm'], q=3,
                                  labels=['Low Sand (Clayey)', 'Medium Sand', 'High Sand (Sandy)'])
    df['precip_category'] = pd.qcut(df['summer_precip_anomaly_forecast'], q=3,
                                    labels=['Dry Summer', 'Normal Summer', 'Wet Summer'])

    plt.figure()
    sns.pointplot(data=df, x='precip_category', y='kreisYield', hue='sand_category', dodge=True)
    plt.title("Interaction Hypothesis: Drought is Worse on Sandy Soil")
    plt.xlabel("Summer Precipitation Anomaly Forecast")
    plt.ylabel("Average Actual Yield (dt/ha)")
    plt.grid(True, which='both', linestyle=':')
    plt.savefig(output_dir / '4a_interaction_hypothesis_test.png', dpi=300);
    plt.close()

    logging.info("✓ Part 4 complete. The plot should show non-parallel lines if an interaction exists.")


# ==============================================================================
# === MAIN ORCHESTRATION FUNCTION ===
# ==============================================================================
def main():
    """Main function to run the entire feature analysis pipeline."""
    logging.info("====== Starting Input Feature Analysis Dashboard Generation ======")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    # --- Load Data ---
    try:
        df = pd.read_csv(FEATURES_FILE)
        gdf = gpd.read_file(GEOJSON_FILE)
        gdf['id'] = gdf['id'].astype(str).str.zfill(5)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    except FileNotFoundError as e:
        logging.error(f"❌ FATAL: A required file was not found. Details: {e}");
        return

    # Ensure data is clean for analysis
    df.dropna(subset=['kreisYield'], inplace=True)

    # --- Run Analysis for Each Part ---
    part_1_analyze_landscape(df, gdf, OUTPUT_DIR)
    part_2_analyze_timeseries_forecast(df, OUTPUT_DIR)
    part_3_analyze_weather_features(df, OUTPUT_DIR)
    part_4_analyze_interaction_hypothesis(df, OUTPUT_DIR)

    logging.info("\n====== ✅ Feature Analysis Dashboard Generation Complete! ======")
    logging.info(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()