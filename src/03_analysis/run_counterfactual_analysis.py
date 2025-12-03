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


def analyze_plausibility():
    feature_file = config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']['OUTPUT_FILE']
    logging.info(f"--- INSPECTING: {feature_file} ---")

    try:
        df = pd.read_csv(feature_file)
    except Exception as e:
        logging.error(f"Could not load file: {e}")
        return

    # 1. Define the "Forensic Targets"
    # We expect:
    # 2003: High Heat, Low Precip (Drought)
    # 2014: High Winter Precip, Early Sowing, Moderate Summer Heat (Bumper)
    # 2018: Low Winter Precip, Late Sowing, High Heat (Disaster)

    years_of_interest = [2003, 2014, 2018]

    # Features to inspect for "Physical Reality"
    check_cols = [
        'stage1_forecast',  # Is the baseline already delusional?
        'sowing_doy',  # Is 2014 actually lower (earlier) than 2018?
        'winter_precip_sum',  # Is the "Gas Tank" full in 2014?
        'winter_gdd',  # Was 2014 actually mild?
        'summer_days_tmax_gt_30c',  # Did we capture the heatwaves?
        'summer_water_balance_anomaly',  # Is 2018 negative?
        'late_sowing_x_summer_heat',  # The new interaction
        'flash_drought_index'  # The kill switch
    ]

    # Filter only columns that exist
    check_cols = [c for c in check_cols if c in df.columns]

    # 2. Global Averages (The Baseline)
    logging.info("\n=== GLOBAL AVERAGES (1981-2024) ===")
    global_means = df[check_cols].mean()
    print(global_means.to_string())

    # 3. The "Yearly Report Card"
    logging.info("\n=== CRITICAL YEAR DIAGNOSTICS ===")

    for year in years_of_interest:
        year_data = df[df['year'] == year]
        if year_data.empty:
            logging.warning(f"Year {year} not found in dataset!")
            continue

        logging.info(f"\n--- YEAR {year} (Average across all districts) ---")

        # Calculate mean for this year
        year_means = year_data[check_cols].mean()

        # Compare to global average
        comparison = pd.DataFrame({
            'Year_Avg': year_means,
            'Global_Avg': global_means,
            'Diff': year_means - global_means,
            'Status': ['Normal'] * len(check_cols)
        })

        # Add simple text status
        for idx, row in comparison.iterrows():
            if row['Diff'] > (0.2 * abs(row['Global_Avg'])):
                comparison.loc[idx, 'Status'] = 'HIGH (+)'
            elif row['Diff'] < (-0.2 * abs(row['Global_Avg'])):
                comparison.loc[idx, 'Status'] = 'LOW (-)'

        print(comparison[['Year_Avg', 'Status', 'Diff']].to_string())

    # 4. Correlation Check (Does X actually affect Y?)
    logging.info("\n=== CAUSALITY CHECK (Correlation with Residual) ===")
    # Create the target: Residual (Actual Yield - Stage1 Forecast)
    # Positive Residual = Stage1 underpredicted (Yield was better than expected)
    # Negative Residual = Stage1 overpredicted (Yield was worse than expected)
    if 'kreisYield' in df.columns and 'stage1_forecast' in df.columns:
        df['forecast_residual'] = df['kreisYield'] - df['stage1_forecast']

        correlations = df[check_cols].corrwith(df['forecast_residual']).sort_values(ascending=False)
        print("Correlation with Forecast Residual (What drives the error?):")
        print(correlations.to_string())

        logging.info("\nINTERPRETATION GUIDE:")
        logging.info("Positive Corr: Higher feature value -> Higher Actual Yield (relative to forecast).")
        logging.info("Negative Corr: Higher feature value -> Lower Actual Yield (relative to forecast).")
    else:
        logging.warning("Cannot calc correlations (missing yield or stage1_forecast).")


if __name__ == '__main__':
    analyze_plausibility()