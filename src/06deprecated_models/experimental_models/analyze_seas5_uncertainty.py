# File: src/visualization/validate_seas5_uncertainty_signal.py
# Description: COMPLETE and CORRECTED script to validate the SEAS5 uncertainty signal.
#              Includes the fix for the data type mismatch error.

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
MEMBER_FEATURES_PATH = 'data/02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv'
STATIC_FEATURES_PATH = 'data/05_model_input/stage1_preseason_features.csv'
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = 'reports/figures/diagnostics'
VARIABLE_PAIRS_TO_VALIDATE = {
    'summer_temp': ('summer_temp_anomaly_forecast', 'summer_temp_anomaly_forecast'),
    'summer_precip': ('summer_precip_anomaly_forecast', 'summer_precip_anomaly_forecast'),
}


def load_data():
    """Loads and prepares the necessary datasets with corrected data types."""
    print("--- Loading Datasets ---")
    try:
        member_df = pd.read_csv(MEMBER_FEATURES_PATH)
        static_df = pd.read_csv(STATIC_FEATURES_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)
        gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)

        # --- BUG FIX APPLIED HERE ---
        # Enforce consistent, zero-padded string type for the merge key
        member_df['district_no'] = member_df['district_no'].astype(str).str.zfill(5)
        static_df['district_no'] = static_df['district_no'].astype(str).str.zfill(5)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        # --- END BUG FIX ---

        return member_df, static_df, gdf_districts
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: Could not find a required file. Details: {e}")
        return None, None, None


def calculate_yearly_metrics(member_df, static_df):
    """Calculates yearly forecasted uncertainty and actual extremity."""
    print("-> Calculating yearly forecasted uncertainty and actual extremity...")
    forecast_uncertainty_by_district = member_df.groupby(['year', 'district_no'])[
        [v[0] for v in VARIABLE_PAIRS_TO_VALIDATE.values()]].std()
    yearly_forecast_uncertainty = forecast_uncertainty_by_district.groupby('year').mean().reset_index()

    static_df_abs = static_df.copy()
    for _, static_col in VARIABLE_PAIRS_TO_VALIDATE.values():
        static_df_abs[static_col] = static_df_abs[static_col].abs()
    yearly_actual_extremity = static_df_abs.groupby('year')[
        [v[1] for v in VARIABLE_PAIRS_TO_VALIDATE.values()]].mean().reset_index()

    validation_df = pd.merge(yearly_forecast_uncertainty, yearly_actual_extremity, on='year',
                             suffixes=('_forecast_uncertainty', '_actual_extremity'))
    return validation_df


def plot_uncertainty_vs_extremity_timeline(validation_df):
    """Generates the core validation timeline plot."""
    print("-> Generating validation timeline plots...")
    for var_name, (member_col, static_col) in VARIABLE_PAIRS_TO_VALIDATE.items():
        uncertainty_col = f'{member_col}_forecast_uncertainty';
        extremity_col = f'{static_col}_actual_extremity'
        fig, ax1 = plt.subplots(figsize=(15, 8))
        color1 = 'darkgrey';
        ax1.set_xlabel('Year');
        ax1.set_ylabel('Forecast Uncertainty (Avg. Std Dev across SEAS5 Members)', color=color1)
        ax1.bar(validation_df['year'], validation_df[uncertainty_col], color=color1, alpha=0.6,
                label='Forecasted Uncertainty Signal')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax2 = ax1.twinx();
        color2 = 'crimson';
        ax2.set_ylabel('Actual Weather Extremity (Avg. Absolute Anomaly)', color=color2)
        ax2.plot(validation_df['year'], validation_df[extremity_col], color=color2, marker='o', linestyle='-',
                 label='Observed Weather Extremity')
        ax2.tick_params(axis='y', labelcolor=color2)
        plt.title(
            f'Validation: Does SEAS5 Uncertainty Predict Real-World Extremity?\n({var_name.replace("_", " ").title()})',
            fontsize=16, pad=20)
        fig.tight_layout();
        lines, labels = ax1.get_legend_handles_labels();
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        output_path = os.path.join(REPORT_DIR, f'validation_timeline_{var_name}.png')
        plt.savefig(output_path, bbox_inches='tight');
        plt.close()
        print(f"   -> Validation plot saved to {output_path}")


def plot_average_uncertainty_map(member_df, gdf_districts):
    """Generates a map showing which districts are chronically uncertain for SEAS5."""
    print("-> Generating map of chronic SEAS5 uncertainty...")
    uncertainty_by_district_year = member_df.groupby(['year', 'district_no'])[
        [v[0] for v in VARIABLE_PAIRS_TO_VALIDATE.values()]].std()
    avg_uncertainty_by_district = uncertainty_by_district_year.groupby('district_no').mean().reset_index()
    for var_name, (member_col, _) in VARIABLE_PAIRS_TO_VALIDATE.items():
        merged_gdf = gdf_districts.merge(avg_uncertainty_by_district, on='district_no', how='left')
        fig, ax = plt.subplots(1, 1, figsize=(12, 12));
        merged_gdf.plot(column=member_col, cmap='viridis', linewidth=0.5, ax=ax, edgecolor='0.8',
                        legend=True, legend_kwds={'label': f"Avg. Forecast Std Dev", 'orientation': "horizontal"},
                        missing_kwds={'color': 'lightgrey'})
        ax.set_title(f'Average SEAS5 Forecast Disagreement (1981-2024)\n({var_name.replace("_", " ").title()})',
                     fontsize=16)
        ax.set_axis_off();
        output_path = os.path.join(REPORT_DIR, f'chronic_uncertainty_map_{var_name}.png')
        plt.savefig(output_path, bbox_inches='tight');
        plt.close()
        print(f"   -> Chronic uncertainty map saved to {output_path}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True);
    member_df, static_df, gdf_districts = load_data()
    if member_df is None: return
    validation_df = calculate_yearly_metrics(member_df, static_df)
    if not validation_df.empty:
        plot_uncertainty_vs_extremity_timeline(validation_df); plot_average_uncertainty_map(member_df, gdf_districts)
    else:
        print("❌ Could not generate validation metrics. Aborting plotting.")


if __name__ == "__main__":
    main();
    print("\n--- SEAS5 Uncertainty Validation Complete ---")