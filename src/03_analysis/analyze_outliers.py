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

    if 'kreisYield' not in master_df.columns:
        print("Warning: 'kreisYield' not found in features file. Directional analysis might be incomplete.")
        master_df['kreisYield'] = np.nan

    master_df = pd.merge(master_df, outliers_df[
        ['district_no', 'year', 'predicted_yield_lower', 'predicted_yield_median', 'predicted_yield_upper',
         'is_outside_interval']],
                         on=['district_no', 'year'], how='left')

    master_df['is_outside_interval'] = master_df['is_outside_interval'].fillna(False)
    master_df['status'] = master_df['is_outside_interval'].apply(lambda x: 'Outlier' if x else 'In-lier')

    def get_outlier_type(row):
        if not row['is_outside_interval']: return 'In-lier'
        if pd.notna(row['kreisYield']):
            if row['kreisYield'] < row['predicted_yield_lower']: return 'Unexpected LOW (Disaster)'
            if row['kreisYield'] > row['predicted_yield_upper']: return 'Unexpected HIGH (Bumper)'
        return 'Outlier (Unknown Dir)'

    master_df['outlier_type'] = master_df.apply(get_outlier_type, axis=1)

    print(f"Data merged. Total rows for analysis: {len(master_df)}")
    print(f"Outlier count in master set: {master_df['is_outside_interval'].sum()}")
    return master_df


def analyze_temporal_patterns(df):
    """
    Checks if specific years are responsible for the majority of outliers
    and returns the top 5 worst years.
    """
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
    plt.xlabel("Year")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '01_outlier_rate_by_year.png'))
    plt.close()

    return yearly_stats.head(5).index.tolist()


def analyze_directional_bias(df):
    """Checks if the model fails more often by overestimating or underestimating."""
    print("Analyzing Directional Bias...")
    outliers_only = df[df['is_outside_interval']]
    bias_counts = outliers_only['outlier_type'].value_counts()
    bias_counts = bias_counts[bias_counts.index.str.contains("Unexpected")]

    print("Outlier Breakdown by Direction:")
    print(bias_counts)

    plt.figure(figsize=(8, 8))
    plt.pie(bias_counts, labels=bias_counts.index, autopct='%1.1f%%', colors=['tomato', 'skyblue', 'lightgreen'])
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
    plt.xlabel("Outlier Type")
    plt.xticks(rotation=10)
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
    g.fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='grey', alpha=0.6),
                          plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.6)],
                 labels=['In-lier', 'Outlier'],
                 loc='upper right')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Feature Distributions: Do Outliers have extreme input values?")
    plt.savefig(os.path.join(REPORT_DIR, '04_feature_forensics.png'))
    plt.close()
    print("Feature forensics plots saved.")


def analyze_worst_years_bias(df, worst_years_list):
    """Creates a new graph showing outlier direction, count, and percentage for the worst years."""
    print(f"Analyzing Directional Bias for Worst Years: {worst_years_list}...")

    # Get total counts for each of the worst years for percentage calculation
    total_counts_per_year = df[df['year'].isin(worst_years_list)].groupby('year')['district_no'].count()

    # Filter for directional outliers in the specified worst years
    worst_years_df = df[df['year'].isin(worst_years_list) & (df['status'] == 'Outlier')]
    directional_outliers = worst_years_df[worst_years_df['outlier_type'].str.contains("Unexpected")]

    if directional_outliers.empty:
        print("No directional outliers found for the specified worst years. Skipping plot.")
        return

    plt.figure(figsize=(14, 8))
    ax = sns.countplot(data=directional_outliers, x='year', hue='outlier_type',
                       order=sorted(worst_years_list),
                       palette={'Unexpected LOW (Disaster)': 'tomato', 'Unexpected HIGH (Bumper)': 'skyblue'})

    # --- Add new annotations (count and percentage) and update x-axis labels ---

    # 1. Update x-axis tick labels to include total counts
    new_labels = []
    for tick_label in ax.get_xticklabels():
        year = int(tick_label.get_text())
        total_count = total_counts_per_year.get(year, 0)
        new_labels.append(f'{year}\n(N={total_count})')
    ax.set_xticklabels(new_labels)

    # 2. Add count and percentage labels to each bar
    for p in ax.patches:
        height = p.get_height()
        if height == 0: continue

        # Get the year for the current bar
        year_str = p.get_x() + p.get_width() / 2.
        year = sorted(worst_years_list)[int(np.round(year_str))]

        total_for_year = total_counts_per_year[year]
        percentage = (height / total_for_year) * 100

        # Format the label string
        label = f'{int(height)}\n({percentage:.1f}%)'

        ax.annotate(label, (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=10, color='black',
                    xytext=(0, 9), textcoords='offset points')

    plt.title('Direction of Prediction Errors in 5 Worst-Performing Years')
    plt.ylabel('Number of Outlier Districts')
    plt.xlabel('Year\n(Total Forecasts)')
    plt.legend(title='Error Type')
    plt.ylim(top=ax.get_ylim()[1] * 1.15)  # Adjust y-limit to make space for labels

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, '05_worst_years_directional_bias.png'))
    plt.close()
    print("Worst years directional bias plot saved.")


def main():
    print("Starting Outlier Deep-Dive Analysis")
    master_df = load_and_merge_data()

    worst_5_years = analyze_temporal_patterns(master_df)

    analyze_directional_bias(master_df)
    analyze_uncertainty_awareness(master_df)
    analyze_feature_differences(master_df)

    analyze_worst_years_bias(master_df, worst_5_years)

    print(f"\nAnalysis complete. Check reports in: {REPORT_DIR}")


if __name__ == "__main__":
    main()