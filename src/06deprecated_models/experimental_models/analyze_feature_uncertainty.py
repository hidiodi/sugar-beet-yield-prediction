# File: src/visualization/analyze_feature_uncertainty.py
# Description: A diagnostic script to perform a full "Uncertainty Audit" of the
#              entire input feature set. It analyzes and visualizes the certainty
#              of forecast features and the historical volatility of observed features.

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
# The main, aggregated feature file is our source of data
STATIC_FEATURES_PATH = 'data/05_model_input/stage1_preseason_features.csv'
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = 'reports/figures/diagnostics/feature_uncertainty_audit'

# --- Features to Analyze ---

# Category 3: Probabilistic Forecasts. How certain is the forecast?
# A value of 0.5 (50%) is max uncertainty. 0 or 1 is max certainty.
PROB_FORECASTS_TO_ANALYZE = {
    'summer_temp_certainty': 'summer_temp_prob_warm_forecast',
    'summer_precip_certainty': 'summer_precip_prob_wet_forecast',
}

# Category 2: Observed Historical Features. How volatile has this been in the past?
HISTORICAL_FEATURES_TO_ANALYZE = {
    'fertilizer_price_volatility': 'fertilizer_price_index_lag1_anomaly_capped',
    'winter_ndvi_volatility': 'winter_cropland_ndvi_anomaly',
    'antecedent_gdd_volatility': 'antecedent_gdd_sum_anomaly'
}


def analyze_forecast_certainty(df: pd.DataFrame, gdf: gpd.GeoDataFrame):
    """
    Analyzes the certainty of probabilistic forecasts.
    A forecast is certain if its probability is close to 0 or 1.
    It is uncertain if its probability is close to 0.5.
    We measure average certainty by calculating abs(probability - 0.5).
    """
    print("--- Analyzing Certainty of Probabilistic Forecasts ---")

    for name, col in PROB_FORECASTS_TO_ANALYZE.items():
        # A value of 0.5 is max uncertainty. A value of 0 or 1 is max certainty.
        # We calculate the deviation from 0.5 to measure certainty.
        df[name] = (df[col] - 0.5).abs()

        avg_certainty = df.groupby('district_no')[name].mean().reset_index()

        merged_gdf = gdf.merge(avg_certainty, on='district_no', how='left')

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        merged_gdf.plot(column=name, cmap='Greens', linewidth=0.5, ax=ax, edgecolor='0.8',
                        legend=True, legend_kwds={'label': "Average Forecast Certainty (0=Uncertain, 0.5=Certain)",
                                                  'orientation': "horizontal"},
                        missing_kwds={'color': 'lightgrey'})
        ax.set_title(f'Average Certainty of SEAS5 Probabilistic Forecasts\n({name.replace("_", " ").title()})',
                     fontsize=16)
        ax.set_axis_off()
        output_path = os.path.join(REPORT_DIR, f'01_forecast_certainty_{name}.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"-> Certainty map for '{name}' saved.")


def analyze_historical_volatility(df: pd.DataFrame, gdf: gpd.GeoDataFrame):
    """
    Analyzes the historical volatility (standard deviation over time) of key observed features.
    """
    print("\n--- Analyzing Historical Volatility of Observed Features ---")

    for name, col in HISTORICAL_FEATURES_TO_ANALYZE.items():
        # Calculate the standard deviation for each district over the years
        volatility = df.groupby('district_no')[col].std().reset_index().rename(columns={col: name})

        merged_gdf = gdf.merge(volatility, on='district_no', how='left')

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        merged_gdf.plot(column=name, cmap='Oranges', linewidth=0.5, ax=ax, edgecolor='0.8',
                        legend=True, legend_kwds={'label': "Standard Deviation over Time", 'orientation': "horizontal"},
                        missing_kwds={'color': 'lightgrey'})
        ax.set_title(f'Historical Volatility of Key Inputs\n({name.replace("_", " ").title()})', fontsize=16)
        ax.set_axis_off()
        output_path = os.path.join(REPORT_DIR, f'02_historical_volatility_{name}.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"-> Volatility map for '{name}' saved.")


def main():
    """Main function to run the full feature uncertainty audit."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("===== STARTING: Feature Uncertainty Audit =====")

    try:
        df = pd.read_csv(STATIC_FEATURES_PATH)
        gdf = gpd.read_file(GEOJSON_PATH)
        gdf.rename(columns={'id': 'district_no'}, inplace=True)
        # Enforce consistent data types
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        gdf['district_no'] = gdf['district_no'].astype(str).str.zfill(5)
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: Could not find a required file. Details: {e}")
        return

    analyze_forecast_certainty(df, gdf)
    analyze_historical_volatility(df, gdf)

    print("\n===== Audit Finished. =====")
    print(f"All reports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()