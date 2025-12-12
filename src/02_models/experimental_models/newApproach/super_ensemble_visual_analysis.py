import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from sklearn.metrics import mean_absolute_error

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])

# File paths
FINAL_FORECAST_PATH = OUTPUT_DIR / 'super_ensemble_final_forecast_TSCV.csv'
INPUT_DATA_PATH = OUTPUT_DIR / 'super_ensemble_training_data.csv'

# Test period for analysis
TEST_START_YEAR = 2014


def load_data_and_prepare_analysis_frame():
    """Loads and combines all necessary data, calculating errors and gain metrics."""
    if not FINAL_FORECAST_PATH.exists() or not INPUT_DATA_PATH.exists():
        logging.error("Required data files not found. Cannot generate plots.")
        return pd.DataFrame(), {}

    # Load the base component predictions and true yield
    df_base = pd.read_csv(INPUT_DATA_PATH)

    # Load the Super Ensemble prediction and the Meta-Learner's decision
    df_final = pd.read_csv(FINAL_FORECAST_PATH)[['year', 'district_no', 'Super_Ensemble_pred', 'Predicted_Best_Model']]

    # Merge
    df_analysis = pd.merge(df_base, df_final, on=['year', 'district_no'], how='inner').copy()

    # Define models for comparison (must match column names)
    models = {
        'Super Ensemble': 'Super_Ensemble_pred',
        'V31 Solar Gated': 'V31_Solar_Gated_pred',
        'Statistical Trend': 'Statistical_Trend_pred',
        'Native Ensemble': 'Native_Ensemble_pred',
    }
    pred_cols = list(models.values())

    # Filter for the relevant volatile period and clean NaNs
    df_clean = df_analysis[df_analysis['year'] >= TEST_START_YEAR].dropna(subset=['kreisYield'] + pred_cols).copy()

    # Calculate all errors
    df_clean['SE_Error'] = (df_clean['Super_Ensemble_pred'] - df_clean['kreisYield']).abs()

    # Find the error of the WORST model (for relative gain calculation)
    df_clean['Worst_Error'] = df_clean[[col for col in pred_cols if 'Native_Ensemble' in col]].apply(
        lambda x: (x - df_clean['kreisYield']).abs().max(),
        axis=1)  # Using Native Ensemble as proxy for a historically poor performer

    # Calculate the GAIN: How much did the Super Ensemble reduce the error relative to the worst component?
    # This shows the value of the switch
    df_clean['SE_Gain_Over_Worst'] = df_clean['Worst_Error'] - df_clean['SE_Error']

    return df_clean, models


def generate_error_landscape_heatmap(df):
    """
    Diagnostic Plot 1: Shows the gain/loss of the Super Ensemble by Actual Yield (stress level).
    """
    logging.info("\n--- Generating Diagnostic Plot 1: Error Landscape and Super Ensemble Gain ---")

    # Bin the actual yield to create discrete stress levels
    df['Yield_Bin'] = pd.cut(df['kreisYield'], bins=np.linspace(df['kreisYield'].min(), df['kreisYield'].max(), 10),
                             include_lowest=True)

    # Calculate median metrics per bin
    heatmap_data = df.groupby('Yield_Bin', observed=True).agg(
        Count=('year', 'count'),
        Median_SE_Error=('SE_Error', 'median'),
        Median_V31_Error=('V31_Solar_Gated_pred', lambda x: (x - df.loc[x.index, 'kreisYield']).abs().median()),
        Median_Gain=('SE_Gain_Over_Worst', 'median')
    ).reset_index()

    plt.figure(figsize=(10, 8))

    # We will plot the median SE Error against the Yield Bin (stress level)
    sns.lineplot(x=heatmap_data.index, y='Median_SE_Error', data=heatmap_data, label='Median Super Ensemble Error',
                 color='k', linewidth=3)

    # Add V31 Error for context
    sns.lineplot(x=heatmap_data.index, y='Median_V31_Error', data=heatmap_data, label='Median V31 Error', color='red',
                 linestyle='--')

    # Add the Gain as a bar chart (secondary Y-axis) to show where the switch adds value
    ax2 = plt.gca().twinx()
    ax2.bar(heatmap_data.index, heatmap_data['Median_Gain'], alpha=0.3, color='g', label='Median Gain (vs Worst Base)')

    plt.xticks(heatmap_data.index, [f'{b.left:.0f}-{b.right:.0f}' for b in heatmap_data['Yield_Bin']], rotation=45,
               ha='right')

    plt.title('Error Landscape: Super Ensemble Error by Yield Stress Level (2014+)', fontsize=14)
    plt.xlabel('Actual Yield Bin (dt/ha) - Proxy for Stress Level')
    plt.ylabel('Error (dt/ha)')
    ax2.set_ylabel('Median Gain (dt/ha) Over Worst Baseline', color='g')

    # Combine legends
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    chart_path = OUTPUT_DIR / 'super_ensemble_error_landscape.png'
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    logging.info(f"✓ Saved Error Landscape Plot to {chart_path}")


def generate_switching_efficiency_boxplot(df):
    """
    Diagnostic Plot 2: Shows the distribution of absolute error only for the predictions
    that the Meta-Learner selected. This validates the decision quality.
    """
    logging.info("\n--- Generating Diagnostic Plot 2: Switching Efficiency Box Plot ---")

    # Prepare data for box plot: We need the error of the *selected* model
    df['Selected_Error'] = (df['Super_Ensemble_pred'] - df['kreisYield']).abs()

    # The 'Predicted_Best_Model' column now serves as the group for the box plot

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Predicted_Best_Model', y='Selected_Error', data=df, palette='viridis')

    # Calculate the average MAE for each selected group for verification
    group_mae = df.groupby('Predicted_Best_Model')['Selected_Error'].mean().sort_values(ascending=False)

    # Add text labels for the overall group MAE (Mean Absolute Error)
    for i, model in enumerate(group_mae.index):
        plt.text(i, group_mae.iloc[i] + 5, f'MAE: {group_mae.iloc[i]:.1f}', ha='center', color='k', fontsize=10,
                 weight='bold')

    plt.title('Meta-Learner Efficiency: Error Distribution of Selected Components (2014+)', fontsize=14)
    plt.xlabel('Component Model Selected by Meta-Learner')
    plt.ylabel('Absolute Error (dt/ha)')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.xticks(rotation=15, ha='right')

    chart_path = OUTPUT_DIR / 'super_ensemble_switching_efficiency.png'
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    logging.info(f"✓ Saved Switching Efficiency Box Plot to {chart_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, models = load_data_and_prepare_analysis_frame()

    if df.empty:
        logging.error("Data loading failed. Cannot generate visuals.")
        return

    generate_error_landscape_heatmap(df)
    generate_switching_efficiency_boxplot(df)


if __name__ == '__main__':
    main()