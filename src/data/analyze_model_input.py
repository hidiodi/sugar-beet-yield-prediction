import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def analyze_dataset(file_path):
    """
    Performs a detailed analysis of the model input dataset and saves reports.
    """
    # --- Setup ---
    print("--- Starting Detailed Data Analysis ---")

    # Create a directory to save reports and figures
    reports_path = 'reports/'
    figures_path = os.path.join(reports_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)

    try:
        df = pd.read_csv(file_path)
        print(f"\nSuccessfully loaded '{file_path}'.")
    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'. Please ensure the path is correct.")
        return

    # --- 1. Overall Summary ---
    print("\n--- 1. Overall Summary ---")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nData types of each column:")
    print(df.info())

    # --- 2. Missing Data Analysis ---
    print("\n--- 2. Missing Data Analysis ---")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage (%)': missing_percentage})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage (%)', ascending=False)

    if not missing_df.empty:
        print("Columns with missing data:")
        print(missing_df)
        # Save to a CSV file for review
        missing_df.to_csv(os.path.join(reports_path, 'missing_data_report.csv'))
    else:
        print("No missing data found in the dataset. Excellent!")

    # --- 3. Descriptive Statistics ---
    print("\n--- 3. Descriptive Statistics for Numerical Features ---")
    pd.set_option('display.float_format', lambda x: '%.2f' % x)  # Format for readability
    descriptive_stats = df.describe().transpose()
    print(descriptive_stats)
    descriptive_stats.to_csv(os.path.join(reports_path, 'descriptive_statistics.csv'))

    # --- 4. Target Variable (Yield) Analysis ---
    print("\n--- 4. Target Variable ('yield') Analysis ---")
    if 'yield' in df.columns:
        print(df['yield'].describe())

        # Plot the distribution of the yield
        plt.figure(figsize=(10, 6))
        sns.histplot(df['yield'].dropna(), kde=True, bins=30)
        plt.title('Distribution of Yield')
        plt.xlabel('Yield')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(figures_path, 'yield_distribution.png'))
        plt.close()
        print(f"Saved yield distribution plot to '{figures_path}/yield_distribution.png'")
    else:
        print("Target variable 'yield' not found.")

    # --- 5. Correlation Analysis ---
    print("\n--- 5. Correlation Analysis ---")
    # Exclude non-numeric columns if any exist before calculating correlation
    numeric_df = df.select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr()

    # Save the full correlation matrix
    correlation_matrix.to_csv(os.path.join(reports_path, 'correlation_matrix.csv'))
    print(f"Full correlation matrix saved to '{reports_path}/correlation_matrix.csv'")

    if 'yield' in correlation_matrix.columns:
        yield_correlations = correlation_matrix['yield'].sort_values(ascending=False)
        print("\nTop features correlated with 'yield':")
        print(yield_correlations)
        yield_correlations.to_csv(os.path.join(reports_path, 'yield_correlations.csv'))

        # Plot a heatmap of the correlation matrix
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
        plt.title('Feature Correlation Heatmap')
        plt.savefig(os.path.join(figures_path, 'correlation_heatmap.png'))
        plt.close()
        print(f"Saved correlation heatmap to '{figures_path}/correlation_heatmap.png'")

    # --- 6. Specific Anomaly Detection ---
    print("\n--- 6. Anomaly Detection ---")
    # Check for the campaign duration anomaly
    if all(col in df.columns for col in ['national_campaign_start_day_of_year', 'national_campaign_end_day_of_year']):
        # Find rows where the end day is before the start day (and not in the next year)
        # This simple check assumes end dates in Jan/Feb are from the *next* year
        anomaly_df = df[
            (df['national_campaign_end_day_of_year'] < df['national_campaign_start_day_of_year']) &
            (df['national_campaign_end_day_of_year'] > 60)  # Heuristic: exclude plausible Jan/Feb end dates
            ]

        if not anomaly_df.empty:
            print(f"\n[!!] WARNING: Found {len(anomaly_df)} rows with potential campaign date anomalies.")
            print("These rows have an end_day_of_year that is before the start_day_of_year.")
            print("Example Anomalies:")
            print(anomaly_df[['year', 'national_campaign_start_day_of_year', 'national_campaign_end_day_of_year',
                              'national_campaign_duration']].head())
            anomaly_df.to_csv(os.path.join(reports_path, 'campaign_date_anomalies.csv'), index=False)
            print(f"Full list of anomalies saved to '{reports_path}/campaign_date_anomalies.csv'")
        else:
            print("No obvious campaign date anomalies were found.")

    print("\n--- Analysis Complete ---")


if __name__ == '__main__':
    # Define the path to your model input file
    model_input_file = 'data/05_model_input/model_input.csv'
    analyze_dataset(model_input_file)