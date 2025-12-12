import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')

# =======================================================
# --- MOCK CONFIGURATION (USER MUST ADJUST PATHS) ---
# =======================================================
# IMPORTANT: REPLACE THESE WITH YOUR ACTUAL PATHS
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"

MOCK_CONFIG = {
    'FILE_PATHS': {
        # 1. Your main feature file (to correlate against)
        'MASTER_DATASET': DATA_DIR / '04_master/master_dataset.csv',
        # 2. Directory containing daily weather CSVs (to calculate ground truth)
        'DAILY_WEATHER_DIR': DATA_DIR / '02_intermediate/daily_weather',
    }
}
# Define the start and end years for analysis
START_YEAR = 1981
END_YEAR = 2024


# =======================================================

# =======================================================
# --- GROUND TRUTH CALCULATION FUNCTIONS ---
# (Copied from build_stage1_features.py for consistency)
# =======================================================

def calculate_z_score(series):
    series = series.fillna(series.mean())
    std = series.std()
    if std == 0: return pd.Series(0, index=series.index)
    return (series - series.mean()) / std


def create_granular_weather_features(weather_dir: Path, start_year: int, end_year: int):
    logging.info("--- Generating Observed Heat Days (Ground Truth) ---")
    all_weather_files = list(weather_dir.glob("*.csv"))
    if not all_weather_files:
        logging.error(f"No weather files found in {weather_dir}")
        return pd.DataFrame()

    df_list = []
    # Limit to a reasonable number of files for quick testing, but loop through all
    for f in tqdm(all_weather_files, desc="Loading Daily Weather"):
        try:
            # Note: 'prec' is kept for backward compatibility if 'precip' is missing
            temp = pd.read_csv(f, usecols=lambda x: x in ['district_no', 'date', 'tmin', 'tmax', 'precip', 'prec'])
            if 'date' in temp.columns:
                temp['date'] = pd.to_datetime(temp['date'])
                temp['year'] = temp['date'].dt.year
                temp = temp[(temp['year'] >= start_year) & (temp['year'] <= end_year)]
                if not temp.empty: df_list.append(temp)
        except Exception as e:
            # logging.warning(f"Skipping file {f.name} due to error: {e}") # Too noisy
            continue

    if not df_list:
        logging.error("No valid daily weather data loaded.")
        return pd.DataFrame()

    df_daily = pd.concat(df_list, ignore_index=True)
    df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)
    df_daily['month'] = df_daily['date'].dt.month
    if 'prec' in df_daily.columns: df_daily.rename(columns={'prec': 'precip'}, inplace=True)

    # Simplified logic for ground truth: ONLY need the heat count

    def calc_heat_count(g):
        # Count days > 25 in June, July, August (Months 6, 7, 8)
        heat = ((g['month'].isin([6, 7, 8])) & (g['tmax'] > 25)).sum()
        return pd.Series({
            '__obs_heat': heat
        })

    return df_daily.groupby(['district_no', 'year']).apply(calc_heat_count).reset_index()


# =======================================================
# --- ANALYSIS SCRIPT ---
# =======================================================

def run_correlation_analysis():
    paths = MOCK_CONFIG['FILE_PATHS']

    # 1. Load Master Dataset
    try:
        df_master = pd.read_csv(paths['MASTER_DATASET'])
        df_master['district_no'] = df_master['district_no'].astype(str).str.zfill(5)
        logging.info(f"Loaded master dataset: {len(df_master)} rows.")
    except Exception as e:
        logging.error(f"Failed to load MASTER_DATASET: {e}")
        return

    # 2. Generate Ground Truth (Target)
    df_obs = create_granular_weather_features(paths['DAILY_WEATHER_DIR'], START_YEAR, END_YEAR)
    if df_obs.empty:
        return

    # 3. Merge Target onto Master Data
    df_merged = pd.merge(df_master, df_obs, on=['district_no', 'year'], how='inner')
    logging.info(f"Merged master data with ground truth: {len(df_merged)} samples.")

    # 4. Filter for Correlation Analysis
    # Exclude non-feature columns that are not predictors
    exclude_cols = ['district_no', 'year', 'state_name', 'yield', 'kreisYield', '__obs_heat']

    # Select all numeric columns for correlation calculation
    df_corr = df_merged.select_dtypes(include=np.number)

    # Remove explicitly excluded columns, handling case where they might be numeric
    df_corr = df_corr.drop(columns=[c for c in exclude_cols if c in df_corr.columns], errors='ignore')

    # Drop columns with zero variance (constants)
    df_corr = df_corr.loc[:, df_corr.apply(pd.Series.nunique) != 1]

    # Final target series
    target_series = df_merged['__obs_heat']

    # 5. Calculate Correlation
    # Align the indices before calculating correlation
    target_aligned = target_series.reindex(df_corr.index)

    # Calculate Pearson correlation for all features against the target
    correlations = df_corr.apply(lambda x: x.corr(target_aligned)).sort_values(ascending=False)

    # 6. Report
    logging.info("=" * 70)
    logging.info(f"FEATURE CORRELATION REPORT vs. Observed Heat Days (__obs_heat)")
    logging.info(f"N = {len(df_merged)} samples, Years {START_YEAR}-{END_YEAR}")
    logging.info("=" * 70)

    logging.info("\n>>> TOP 15 POSITIVELY CORRELATED FEATURES (Predicting Heat)")
    logging.info(correlations.head(15).to_string(float_format='%.3f'))

    logging.info("\n>>> TOP 15 NEGATIVELY CORRELATED FEATURES (Predicting Coolness/Rain)")
    logging.info(correlations.tail(15).to_string(float_format='%.3f'))

    logging.info("=" * 70)
    logging.info("Analysis Complete.")


if __name__ == '__main__':
    run_correlation_analysis()