import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')


def analyze_systematic_errors():
    # 1. Load Data
    pred_path = config.BASE_DIR / 'reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv'
    feat_path = config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']['OUTPUT_FILE']

    logging.info(f"Loading Predictions: {pred_path}")
    logging.info(f"Loading Features:    {feat_path}")

    try:
        df_pred = pd.read_csv(pred_path, dtype={'district_no': str})
        df_feat = pd.read_csv(feat_path, dtype={'district_no': str})
    except Exception as e:
        logging.error(f"Failed to load files: {e}")
        return

    # Standardize IDs
    df_pred['district_no'] = df_pred['district_no'].str.zfill(5)
    df_feat['district_no'] = df_feat['district_no'].str.zfill(5)

    # Clean duplicates before merge
    cols_to_drop = ['kreisYield', 'stage1_forecast']
    cols_to_drop = [c for c in cols_to_drop if c in df_feat.columns]
    df_feat_clean = df_feat.drop(columns=cols_to_drop)

    # Merge
    df = pd.merge(df_pred, df_feat_clean, on=['district_no', 'year'], how='inner')

    # Calculate Residuals
    # Residual = Actual - Predicted
    # Negative = Overprediction (Model too optimistic)
    # Positive = Underprediction (Model too pessimistic)
    df['residual'] = df['kreisYield'] - df['predicted_yield_median']
    df['abs_error'] = df['residual'].abs()

    # 2. Define Physical Features to Audit (Ignoring Teleconnections)
    phys_cols = [
        'sowing_doy',
        'winter_precip_sum',
        'effective_winter_water',
        'summer_days_tmax_gt_30c',
        'summer_water_balance_anomaly',
        'solar_capture_potential',
        'flash_drought_index',
        'optimal_growth_index',
        'spring_soil_temp_l1_anomaly_forecast'
    ]
    # Filter for existing columns
    phys_cols = [c for c in phys_cols if c in df.columns]

    # 3. Identify the "Worst Years"
    yearly_stats = df.groupby('year').agg({
        'abs_error': 'mean',
        'residual': 'mean',
        'kreisYield': 'mean'
    }).sort_values('abs_error', ascending=False)

    print("\n" + "=" * 60)
    print("       SYSTEMATIC ERROR AUDIT (Worst 5 Years)")
    print("=" * 60)
    print(yearly_stats.head(5).to_string())

    # 4. Deep Dive into the Top 5 Failures
    worst_years = yearly_stats.head(5).index.tolist()

    # Calculate Long-Term Averages (Baseline)
    lt_means = df[phys_cols].mean()

    for year in worst_years:
        print(f"\n\n>>> DIAGNOSIS FOR YEAR: {year} <<<")
        df_year = df[df['year'] == year]

        # A. The Error Type
        avg_res = df_year['residual'].mean()
        error_type = "OVERPREDICTION (Optimism)" if avg_res < 0 else "UNDERPREDICTION (Pessimism)"
        print(f"Model Bias: {avg_res:.2f} dt/ha -> {error_type}")

        # B. The Physical State (Reality Check)
        print("\n--- Physical State vs Long-Term Avg ---")
        year_means = df_year[phys_cols].mean()

        diffs = pd.DataFrame({
            'Year_Avg': year_means,
            'Global_Avg': lt_means,
            'Diff': year_means - lt_means,
            'Diff_%': ((year_means - lt_means) / lt_means.abs() * 100).round(1)
        })

        # Highlight significant anomalies (>20% deviation)
        def highlight(row):
            if row['Diff_%'] > 20: return "HIGH (+)"
            if row['Diff_%'] < -20: return "LOW (-)"
            return "Normal"

        diffs['Status'] = diffs.apply(highlight, axis=1)
        print(diffs[['Year_Avg', 'Global_Avg', 'Status']].to_string())

        # C. The Blame Game (Correlation with Error)
        # Which feature correlates with the Residual *in this year*?
        # If Corr is High, the model didn't use this feature enough.
        print("\n--- What drove the Error? (Correlation with Residual) ---")
        corrs = df_year[phys_cols].corrwith(df_year['residual']).sort_values(key=abs, ascending=False)
        print(corrs.head(5).to_string())

        print("\nINTERPRETATION:")
        if avg_res < 0:  # Overprediction
            print(f"Model predicted too HIGH. Look for features that should have been PENALTIES but weren't.")
            print(f"Example: If 'solar_capture_potential' is positive but yield crashed, the kill-switch failed.")
        else:  # Underprediction
            print(f"Model predicted too LOW. Look for features that provided a BUFFER/BONUS but were ignored.")
            print(f"Example: If 'effective_winter_water' is high, the model might have ignored the root depth.")


if __name__ == '__main__':
    analyze_systematic_errors()