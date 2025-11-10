# File: src/03_analysis/run_outlier_analysis.py
# Description: Performs a systematic error analysis on the champion model's backtest
#              results to identify the root causes of major failures, both in
#              outlier years and outlier districts.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
# Use the results from our best model so far: the Hybrid Ensemble
BACKTEST_RESULTS_FILE = 'reports/figures/district_level_diagnostics/final_ensemble_champion/full_backtest_predictions.csv'
FEATURE_DATA_FILE = 'data/05_model_input/stage1_preseason_features.csv'

OUTPUT_DIR = Path('reports/figures/outlier_analysis')

# --- Analysis Parameters ---
NUM_OUTLIER_YEARS = 5  # How many worst years to analyze
SAMPLE_NORMAL_YEAR = 2017  # A year with average national error for spatial analysis
OUTLIER_DISTRICT_THRESHOLD = 1.5  # Number of std deviations from the mean error to be an outlier

# --- Features to Investigate ---
# Temporal features that might explain bad years
TEMPORAL_FEATURES = [
    'summer_days_tmax_gt_30c',
    'summer_precip_anomaly_forecast',
    'antecedent_frost_days_anomaly',
    'spring_soil_temp_l1_anomaly_forecast',
    'fertilizer_price_index_lag1_anomaly'
]
# Static features that might explain bad districts
SPATIAL_FEATURES = [
    'avg_sand_0_30cm',
    'avg_clay_0_30cm',
    'avg_elevation',
    'lat'
]


def analyze_outlier_years(df_merged):
    """Identifies the worst-performing years and compares their feature distributions."""
    print("\n--- 1. Temporal Failure Analysis: Outlier Years ---")

    # Calculate national average absolute error for each year
    yearly_error = df_merged.groupby('year')['abs_error'].mean().sort_values(ascending=False)
    outlier_years = yearly_error.head(NUM_OUTLIER_YEARS).index.tolist()
    normal_years = yearly_error.index.difference(outlier_years).tolist()

    print(f"Identified Top {NUM_OUTLIER_YEARS} Outlier Years (by highest avg. error): {outlier_years}")

    df_merged['year_type'] = df_merged['year'].apply(lambda y: 'Outlier' if y in outlier_years else 'Normal')

    # Plot the distributions
    num_features = len(TEMPORAL_FEATURES)
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 5 * num_features))
    fig.suptitle('Feature Distributions: Outlier vs. Normal Years', fontsize=20, y=1.02)

    for i, feature in enumerate(TEMPORAL_FEATURES):
        ax = axes[i]
        sns.boxplot(data=df_merged, x=feature, y='year_type', ax=ax, palette={'Normal': 'skyblue', 'Outlier': 'salmon'})
        ax.set_title(f'Distribution of "{feature}"', fontsize=16)
        ax.set_ylabel('')
        ax.set_xlabel('Feature Value', fontsize=12)

    plt.tight_layout()
    output_path = OUTPUT_DIR / '01_temporal_outlier_analysis.png'
    plt.savefig(output_path, dpi=300)
    print(f"✓ Temporal analysis plot saved to {output_path}")
    plt.close()


def analyze_outlier_districts(df_merged):
    """In a normal year, identifies the worst-performing districts and analyzes their static features."""
    print(f"\n--- 2. Spatial Failure Analysis: Outlier Districts in a Normal Year ({SAMPLE_NORMAL_YEAR}) ---")

    df_year = df_merged[df_merged['year'] == SAMPLE_NORMAL_YEAR].copy()
    if df_year.empty:
        print(f"❌ Could not find data for sample year {SAMPLE_NORMAL_YEAR}. Skipping spatial analysis.")
        return

    # Define outliers as districts with error > X standard deviations from the mean
    mean_error = df_year['abs_error'].mean()
    std_error = df_year['abs_error'].std()
    threshold = mean_error + OUTLIER_DISTRICT_THRESHOLD * std_error

    df_year['district_type'] = df_year['abs_error'].apply(lambda e: 'Outlier' if e > threshold else 'Normal')

    outlier_districts = df_year[df_year['district_type'] == 'Outlier']
    if outlier_districts.empty:
        print(f"✓ No significant outlier districts found in {SAMPLE_NORMAL_YEAR}. This is a good sign.")
        return

    print(
        f"Identified {len(outlier_districts)} outlier districts in {SAMPLE_NORMAL_YEAR} (error > {threshold:.2f} dt/ha).")

    # Compare the average static feature values
    comparison_table = df_year.groupby('district_type')[SPATIAL_FEATURES].mean().T
    comparison_table['difference'] = comparison_table['Outlier'] - comparison_table['Normal']

    print("\nComparison of Static Features (Outlier vs. Normal Districts):")
    print(comparison_table)

    # Save the table
    output_path = OUTPUT_DIR / '02_spatial_outlier_summary.csv'
    comparison_table.to_csv(output_path)
    print(f"\n✓ Spatial analysis summary table saved to {output_path}")


def main():
    """Main orchestration function for the outlier analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("--- Starting Systematic Error Analysis ---")

    try:
        df_results = pd.read_csv(BACKTEST_RESULTS_FILE)
        df_features = pd.read_csv(FEATURE_DATA_FILE)
    except FileNotFoundError as e:
        print(f"❌ FATAL: Input file not found. Ensure backtests have run. Details: {e}")
        return

    # Merge results with features to have everything in one place
    df_merged = pd.merge(df_results, df_features, on=['district_no', 'year'], how='left')

    # Run the two main analysis functions
    analyze_outlier_years(df_merged)
    analyze_outlier_districts(df_merged)

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()