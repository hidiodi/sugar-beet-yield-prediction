import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

sns.set_theme(style="whitegrid")

OUTLIERS_PATH = os.path.join('reports', 'figures', 'district_level_diagnostics', 'quantile_model_diagnostics',
                             'interval_outliers_analysis.csv')
FEATURES_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
REPORT_DIR = os.path.join('reports', 'figures', 'outlier_deep_dive')
os.makedirs(REPORT_DIR, exist_ok=True)

KEY_FEATURES = [
    'summer_temp_anomaly_forecast',
    'summer_precip_anomaly_forecast',
    'spring_precip_anomaly_forecast',
    'antecedent_gdd_sum_anomaly',
    'winter_cropland_ndvi_anomaly',
    'avg_bdod_0_30cm'
]


def load_and_merge_data():
    """Loads outliers and merges them with the full feature set for comparison."""
    print("Loading data...")
    try:
        outliers_df = pd.read_csv(OUTLIERS_PATH)
        features_df = pd.read_csv(FEATURES_PATH)
        outliers_df['district_no'] = outliers_df['district_no'].astype(str).str.zfill(5)
        features_df['district_no'] = features_df['district_no'].astype(str).str.zfill(5)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Missing input file. {e}")
        print("Please run 'backtesting_quantile.py' first.")
        exit()

    backtested_years = outliers_df['year'].unique()
    master_df = features_df[features_df['year'].isin(backtested_years)].copy()

    master_df = pd.merge(master_df, outliers_df[
        ['district_no', 'year', 'predicted_yield_lower', 'predicted_yield_median', 'predicted_yield_upper',
         'is_outside_interval']],
                         on=['district_no', 'year'], how='left')

    master_df['is_outside_interval'] = master_df['is_outside_interval'].fillna(False)
    master_df['status'] = master_df['is_outside_interval'].apply(lambda x: 'Outlier' if x else 'In-lier')

    def get_outlier_type(row):
        if not row['is_outside_interval']: return 'In-lier'
        if 'kreisYield' in row:
            if row['kreisYield'] < row['predicted_yield_lower']: return 'Unexpected LOW (Disaster)'
            if row['kreisYield'] > row['predicted_yield_upper']: return 'Unexpected HIGH (Bumper)'
        return 'Outlier (Unknown Dir)'

    master_df['outlier_type'] = master_df.apply(get_outlier_type, axis=1)

    print(f"Data merged. Total rows for analysis: {len(master_df)}")
    print(f"Outlier count in master set: {master_df['is_outside_interval'].sum()}")
    return master_df


def analyze_temporal_patterns(df):
    """Checks if specific years are responsible for the majority of outliers."""
    print("Analyzing Temporal Patterns...")
    yearly_stats = df.groupby('year').agg(
        total_districts=('district_no', 'count'),
        outlier_count=('is_outside_interval', 'sum')
    )
    yearly_stats['outlier_rate'] = (yearly_stats['outlier_count'] / yearly_stats['total_districts']) * 100
    yearly_stats = yearly_stats.sort_values('outlier_rate', ascending=False)

    print("Top 5 Years by Outlier Rate:")
    print(yearly_stats.head(5))

    plt.figure(figsize=(12, 6))
    sns.barplot(x=yearly_stats.index, y=yearly_stats['outlier_rate'], color='salmon')
    plt.axhline(5, color='black', linestyle='--', label='Target Outlier Rate (5%)')
    plt.title("Percentage of Districts Outside 95% Prediction Interval by Year")
    plt.ylabel("Outlier Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '01_outlier_rate_by_year.png'))
    plt.close()


def analyze_directional_bias(df):
    """Checks if the model fails more often by overestimating or underestimating."""
    print("Analyzing Directional Bias...")
    outliers_only = df[df['is_outside_interval']]
    bias_counts = outliers_only['outlier_type'].value_counts()

    print("Outlier Breakdown by Direction:")
    print(bias_counts)

    plt.figure(figsize=(8, 8))
    plt.pie(bias_counts, labels=bias_counts.index, autopct='%1.1f%%', colors=['tomato', 'skyblue'])
    plt.title("Directional Bias of Outliers")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '02_outlier_directional_bias.png'))
    plt.close()


def analyze_uncertainty_awareness(df):
    """Checks if the model produced wider intervals for the points that became outliers."""
    print("Analyzing Uncertainty Awareness...")
    df['interval_width'] = df['predicted_yield_upper'] - df['predicted_yield_lower']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='outlier_type', y='interval_width', palette='Set2')
    plt.title("Did the model predict wider intervals for the points that became outliers?")
    plt.ylabel("Predicted 95% Interval Width (dt/ha)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '03_interval_width_comparison.png'))
    plt.close()

    avg_width_inlier = df[df['status'] == 'In-lier']['interval_width'].mean()
    avg_width_outlier = df[df['status'] == 'Outlier']['interval_width'].mean()
    print(f"Average Interval Width for In-liers: {avg_width_inlier:.2f}")
    print(f"Average Interval Width for Outliers: {avg_width_outlier:.2f}")
    if avg_width_outlier > avg_width_inlier:
        print("Finding: The model is correctly identifying higher uncertainty for outlier cases.")
    else:
        print("Finding: The model is not detecting the higher risk for outlier cases.")


def analyze_feature_differences(df):
    """Compares feature distributions for outliers vs in-liers to find blind spots."""
    print("Analyzing Feature Differences (Outlier vs In-lier)...")

    plot_data = df.melt(id_vars=['status', 'outlier_type'], value_vars=KEY_FEATURES,
                        var_name='Feature', value_name='Value')

    g = sns.FacetGrid(plot_data, col="Feature", col_wrap=3, sharex=False, sharey=False, height=4)
    g.map_dataframe(sns.violinplot, x="status", y="Value", palette={"In-lier": "grey", "Outlier": "red"}, alpha=0.6)
    g.add_legend()
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Feature Distributions: Do Outliers have extreme input values?")
    plt.savefig(os.path.join(REPORT_DIR, '04_feature_forensics.png'))
    plt.close()
    print("Feature forensics plots saved.")


def main():
    print("Starting Outlier Deep-Dive Analysis")
    master_df = load_and_merge_data()

    analyze_temporal_patterns(master_df)
    analyze_directional_bias(master_df)
    analyze_uncertainty_awareness(master_df)
    analyze_feature_differences(master_df)

    print(f"\nAnalysis complete. Check reports in: {REPORT_DIR}")


if __name__ == "__main__":
    main()