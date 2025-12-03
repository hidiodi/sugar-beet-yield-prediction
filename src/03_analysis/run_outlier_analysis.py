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


def main():
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

    # Ensure District IDs are standard
    df_pred['district_no'] = df_pred['district_no'].str.zfill(5)
    df_feat['district_no'] = df_feat['district_no'].str.zfill(5)

    # --- FIX: Prevent Column Collision ---
    # Both files contain 'kreisYield' and 'stage1_forecast'.
    # Merge creates _x and _y suffixes, breaking the script.
    # We remove these columns from df_feat because df_pred has the authoritative versions for analysis.
    cols_to_drop = ['kreisYield', 'stage1_forecast']
    # Only drop if they actually exist in df_feat
    cols_to_drop = [c for c in cols_to_drop if c in df_feat.columns]

    df_feat_clean = df_feat.drop(columns=cols_to_drop)

    # 2. Merge (Join Predictions with the Physics that created them)
    logging.info("Merging Datasets...")
    df_full = pd.merge(df_pred, df_feat_clean, on=['district_no', 'year'], how='inner')

    # 3. Calculate Error Metrics
    # Residual: Positive = Yield was HIGHER than predicted (Underprediction)
    #           Negative = Yield was LOWER than predicted (Overprediction)
    if 'predicted_yield_median' in df_full.columns and 'kreisYield' in df_full.columns:
        df_full['error_residual'] = df_full['kreisYield'] - df_full['predicted_yield_median']
        df_full['abs_error'] = df_full['error_residual'].abs()
    else:
        logging.error("Critical columns missing after merge. Check CSVs.")
        return

    # 4. Define "The Usual Suspects" (Features to inspect)
    diagnostic_cols = [
        'district_no', 'year', 'kreisYield', 'predicted_yield_median', 'abs_error', 'error_residual',
        'stage1_forecast',
        'sowing_doy',
        'summer_days_tmax_gt_30c',
        'summer_water_balance_anomaly',
        'solar_capture_potential',
        'late_sowing_x_summer_heat',
        'effective_winter_water',
        'flash_drought_index'
    ]
    # Filter only existing cols
    diagnostic_cols = [c for c in diagnostic_cols if c in df_full.columns]

    # 5. Export The Master File
    output_file = config.BASE_DIR / 'reports/figures/district_level_diagnostics/full_model_forensics.csv'
    df_full.sort_values('abs_error', ascending=False, inplace=True)

    df_full.to_csv(output_file, index=False)
    logging.info(f"\n✓ Saved Master Forensic Table to: {output_file}")

    # 6. Console Report
    logging.info("\n=== TOP 20 WORST PREDICTIONS (Largest Absolute Errors) ===")
    print(df_full[diagnostic_cols].head(20).to_string(index=False))

    # 7. Deep Dive: 2014 vs 2018 Analysis
    logging.info("\n=== YEARLY FORENSICS (2014 vs 2018) ===")
    for year in [2014, 2018]:
        logging.info(f"\n--- Analysis for {year} ---")
        df_year = df_full[df_full['year'] == year]

        if df_year.empty:
            logging.warning(f"No data found for {year}")
            continue

        # Calculate Mean Error for the year
        mae_year = df_year['abs_error'].mean()
        bias_year = df_year['error_residual'].mean()

        print(f"Mean Absolute Error (MAE): {mae_year:.2f}")
        print(f"Mean Bias (Residual):      {bias_year:.2f} (Positive=Underpredicted, Negative=Overpredicted)")

        # Show top 5 worst districts for this specific year
        print(f"\nTop 5 Failures in {year}:")
        print(df_year[diagnostic_cols].sort_values('abs_error', ascending=False).head(5).to_string(index=False))


if __name__ == '__main__':
    main()