import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import sys

# --- Setup ---
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
OUTPUT_DIR = config.DATA_DIR / '07_paper_figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File Paths (Adjust if your filenames differ)
FILES = {
    'Trend (Baseline)': config.MODEL_COMPARISON_CONFIG['STATISTICAL_TREND_FILE'],
    'Standalone XGB': config.MODEL_COMPARISON_CONFIG['STANDALONE_XGB_PREDICTIONS_FILE'],
    'Native Ensemble': config.DATA_DIR / '06_model_output/native_ensemble_champion/native_ensemble_forecasts.csv',
    'Regime Switch V24': config.DATA_DIR / '06_model_output/final_switched_forecast.csv'
}


def load_and_merge_models():
    print("Loading all model predictions...")

    # 1. Start with the Master Dataset (Actual Yields)
    # We use the Regime Switch file as the base because it has 'Actual' and 'Strategy_Mode'
    base = pd.read_csv(FILES['Regime Switch V24'])
    base = base[['year', 'district_no', 'Actual', 'Final_Pred', 'Strategy_Mode', 'Cluster_ID']]
    base.rename(columns={'Final_Pred': 'Regime Switch V24'}, inplace=True)

    # 2. Merge other models
    for name, path in FILES.items():
        if name == 'Regime Switch V24': continue  # Already loaded

        try:
            df = pd.read_csv(path)
            # Standardize columns
            if 'final_corrected_forecast' in df.columns:
                col = 'final_corrected_forecast'
            elif 'predicted_yield_median' in df.columns:
                col = 'predicted_yield_median'
            elif 'Ensemble_Pred' in df.columns:
                col = 'Ensemble_Pred'
            else:
                continue

            df = df[['year', 'district_no', col]]
            df.rename(columns={col: name}, inplace=True)

            # Merge
            base = pd.merge(base, df, on=['year', 'district_no'], how='left')

        except Exception as e:
            print(f"Warning: Could not load {name} from {path}. Error: {e}")

    # 3. Clean up (Drop NaNs if any model is missing a year)
    # Note: Standalone XGB might be missing < 2005. We forward fill with Trend for fair comparison if needed.
    if 'Trend (Baseline)' in base.columns:
        for col in base.columns:
            if base[col].isnull().any():
                print(f"Filling missing values in {col} with Trend (Baseline)...")
                base[col] = base[col].fillna(base['Trend (Baseline)'])

    return base.dropna()


def calculate_metrics(df, models):
    results = []
    for m in models:
        mae = mean_absolute_error(df['Actual'], df[m])
        r2 = r2_score(df['Actual'], df[m])
        results.append({'Model': m, 'MAE': mae, 'R²': r2})

    return pd.DataFrame(results).sort_values('MAE', ascending=False)


def plot_conditional_performance(df, models):
    """Shows how models perform in Normal vs. Crisis years."""
    print("\nGenerating Conditional Performance Plot...")

    # Define Regimes based on V24 Logic
    # We group clusters into broad categories for plotting
    df['Condition'] = 'Normal'
    # Crisis Clusters from V24 (2, 3, 6) -> Heat/Anoxia
    # Opportunity Clusters (0, 1, 5) -> Bumper
    # Check your specific V24 output for which IDs map to which strategy!
    # Here we use the Strategy_Mode column directly
    df['Condition'] = df['Strategy_Mode'].apply(lambda x: x.split(' ')[0])  # 'Crisis', 'Opportunity', 'Normal'

    # Calculate MAE per Condition per Model
    plot_data = []
    for m in models:
        for cond in df['Condition'].unique():
            subset = df[df['Condition'] == cond]
            if len(subset) < 50: continue  # Skip small groups
            mae = mean_absolute_error(subset['Actual'], subset[m])
            plot_data.append({'Model': m, 'Condition': cond, 'MAE': mae})

    plot_df = pd.DataFrame(plot_data)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Condition', y='MAE', hue='Model', palette='viridis')
    plt.title("Model Robustness: Performance by Regime (Scenario Analysis)", fontsize=16)
    plt.ylabel("Mean Absolute Error (dt/ha)")
    plt.xlabel("Climatic Condition (Identified by Cluster Analysis)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_model_comparison.png', dpi=300)
    print(f"Saved comparison plot to {OUTPUT_DIR}")


def main():
    df = load_and_merge_models()

    # Define models to compare (excluding metadata columns)
    models = [c for c in df.columns if
              c not in ['year', 'district_no', 'Actual', 'Strategy_Mode', 'Cluster_ID', 'Condition']]

    # 1. Overall Metrics
    print("\n" + "=" * 40)
    print(" OVERALL PERFORMANCE (2000-2024)")
    print("=" * 40)
    metrics = calculate_metrics(df, models)
    print(metrics.to_string(index=False))

    # 2. Breakdown by Regime (The "Value Add")
    print("\n" + "=" * 40)
    print(" PERFORMANCE BY REGIME")
    print("=" * 40)
    for regime in df['Strategy_Mode'].unique():
        sub = df[df['Strategy_Mode'] == regime]
        print(f"\n--- {regime} (N={len(sub)}) ---")
        print(calculate_metrics(sub, models).to_string(index=False))

    # 3. Plot
    plot_conditional_performance(df, models)


if __name__ == "__main__":
    main()