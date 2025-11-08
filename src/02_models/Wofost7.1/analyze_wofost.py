import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def analyze_wofost_output(file_path):
    """
    Performs a comprehensive and robust analysis of Wofost ensemble output,
    gracefully handling common data issues like empty or constant-value columns.

    Args:
        file_path (str): The full path to the ensemble output CSV file.
    """
    print("--- Starting Wofost Ensemble Analysis (Robust Version) ---")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from: {file_path}")
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return

    output_dir = "analysis_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"\nPlots will be saved in the '{output_dir}/' directory.")

    # --- 2. Data Inspection and Cleaning ---
    print("\n--- 2. Data Inspection ---")
    print("Data Info:")
    df.info()

    missing_values = df.isnull().sum()
    print("\nMissing Values per Column:")
    print(missing_values[missing_values > 0])

    # Robustly handle missing values for 'days_to_anthesis'
    if 'days_to_anthesis' in df.columns:
        if df['days_to_anthesis'].isnull().all():
            print("\nWARNING: 'days_to_anthesis' column is completely empty. It will be excluded from analysis.")
            # Drop the column so it doesn't interfere with later steps
            df = df.drop(columns=['days_to_anthesis'])
        else:
            # FIX: Use modern pandas assignment to avoid FutureWarning
            median_anthesis = df['days_to_anthesis'].median()
            df['days_to_anthesis'] = df['days_to_anthesis'].fillna(median_anthesis)
            print(f"\nFilled missing 'days_to_anthesis' with median value ({median_anthesis:.2f}).")

    # --- 3. Overall Descriptive Statistics ---
    print("\n--- 3. Descriptive Statistics (Overall) ---")
    print(df[['yield_water_limited_dry_kgha', 'drought_stress_index', 'max_lai_achieved']].describe())

    # Plot overall yield distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(df['yield_water_limited_dry_kgha'], color='skyblue', kde=True, label='Water-Limited Yield', bins=30)
    sns.histplot(df['yield_potential_dry_kgha'], color='salmon', kde=True, label='Potential Yield', bins=30)
    plt.title('Overall Distribution of Simulated Yields')
    plt.xlabel('Yield (kg/ha)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '01_overall_yield_distribution.png'))
    plt.close()
    print("\nGenerated plot: 01_overall_yield_distribution.png")

    # --- 4. Analysis by Year ---
    print("\n--- 4. Analysis by Year ---")
    yearly_summary = df.groupby('year')['yield_water_limited_dry_kgha'].agg(['mean', 'std', 'min', 'max']).reset_index()
    print("Yearly Average Yields:")
    print(yearly_summary)

    # Plot average yield over time
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_summary['year'], yearly_summary['mean'], marker='o', linestyle='-')
    plt.title('Average Water-Limited Yield Over Years')
    plt.xlabel('Year')
    plt.ylabel('Average Yield (kg/ha)')
    plt.xticks(yearly_summary['year'], rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_yearly_average_yield.png'))
    plt.close()
    print("Generated plot: 02_yearly_average_yield.png")

    # Boxplot of yields per year
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x='year', y='yield_water_limited_dry_kgha')
    plt.title('Distribution of Water-Limited Yield by Year (Ensemble Spread)')
    plt.xlabel('Year')
    plt.ylabel('Yield (kg/ha)')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, '03_yearly_yield_boxplot.png'))
    plt.close()
    print("Generated plot: 03_yearly_yield_boxplot.png")

    # --- 5. Analysis by District (NEW) ---
    print("\n--- 5. Analysis by District ---")
    # Convert district_no to string for better plotting
    df['district_no_str'] = df['district_no'].astype(str)
    district_summary = df.groupby('district_no_str')['yield_water_limited_dry_kgha'].agg(
        ['mean', 'std']).reset_index().sort_values('mean', ascending=False)
    print("Top 10 Districts by Average Yield:")
    print(district_summary.head(10))

    # Plot average yield by district
    plt.figure(figsize=(16, 8))
    sns.barplot(data=district_summary.head(30), x='district_no_str', y='mean', color='teal')
    plt.title('Top 30 Districts by Average Water-Limited Yield')
    plt.xlabel('District Number')
    plt.ylabel('Average Yield (kg/ha)')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_district_average_yield.png'))
    plt.close()
    print("Generated plot: 04_district_average_yield.png")

    # --- 6. Stress Indicator Analysis ---
    print("\n--- 6. Stress Indicator Analysis ---")
    stress_columns = [
        'yield_water_limited_dry_kgha', 'consecutive_tmax_gt_30c', 'consecutive_dry_days',
        'drought_stress_index', 'cumulative_water_stress', 'max_lai_achieved'
    ]
    # Filter out columns that might have been dropped (like days_to_anthesis)
    stress_columns = [col for col in stress_columns if col in df.columns]

    # NEW: Identify and remove zero-variance columns before correlation
    df_for_corr = df[stress_columns]
    stds = df_for_corr.std()
    zero_std_cols = stds[stds == 0].index.tolist()
    if zero_std_cols:
        print(
            f"\nWARNING: The following columns have zero variance and will be excluded from the correlation analysis: {zero_std_cols}")
        df_for_corr = df_for_corr.drop(columns=zero_std_cols)

    correlation_matrix = df_for_corr.corr()

    print("\nCorrelation Matrix of Yield and Stress Indicators:")
    print(correlation_matrix)

    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Yield and Key Variables (with variance)')
    plt.savefig(os.path.join(output_dir, '05_correlation_heatmap.png'))
    plt.close()
    print("Generated plot: 05_correlation_heatmap.png")

    # Scatter plots
    plt.figure(figsize=(14, 6))

    if 'drought_stress_index' in df_for_corr.columns:
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df, x='drought_stress_index', y='yield_water_limited_dry_kgha', alpha=0.5)
        plt.title('Drought Stress Index vs. Yield')
        plt.grid(True)

    if 'cumulative_water_stress' in df_for_corr.columns:
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=df, x='cumulative_water_stress', y='yield_water_limited_dry_kgha', alpha=0.5)
        plt.title('Cumulative Water Stress vs. Yield')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_stress_vs_yield_scatter.png'))
    plt.close()
    print("Generated plot: 06_stress_vs_yield_scatter.png")

    # --- 7. Simulation Failure Analysis ---
    print("\n--- 7. Simulation Failure Analysis ---")
    if not df[df['simulation_failed']].empty:
        failure_rate = df.groupby('year')['simulation_failed'].mean() * 100
        print("Percentage of Failed Simulations per Year:")
        print(failure_rate)

        plt.figure(figsize=(10, 5))
        failure_rate.plot(kind='bar', color='crimson')
        plt.title('Simulation Failure Rate by Year')
        plt.ylabel('Failure Rate (%)')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(output_dir, '07_simulation_failure_rate.png'))
        plt.close()
        print("Generated plot: 07_simulation_failure_rate.png")
    else:
        print("No simulation failures recorded in the dataset.")

    print("\n--- Analysis Complete ---")


if __name__ == '__main__':
    FILE_PATH = 'data/06_model_output/multi_year_final/forecast_ensemble_2023-2024.csv'
    analyze_wofost_output(FILE_PATH)