import geopandas as gpd
import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
import importlib

# Try to import config, but handle failure if paths are wrong in config (we override anyway)
try:
    config = importlib.import_module("src.03_analysis.config")
    analysis_config = importlib.import_module("src.03_analysis.config")
    CONFIG = config.MODEL_COMPARISON_CONFIG
    OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])
except Exception as e:
    logging.warning(f"Could not import config: {e}")
    OUTPUT_DIR = Path("reports/figures/final_model_comparison")

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Consolidated input path
SUPER_ENSEMBLE_WALKFORWARD_PATH = project_root / 'reports/figures/final_model_comparison/super_ensemble_walkforward_predictions.csv'
MASTER_DATA_PATH = config.DATA_DIR / "04_master/master_dataset.csv"
GEOJSON_PATH = config.DATA_DIR / "01_raw/districts_official.geojson"

def load_and_merge_models():
    """
    Loads model predictions from the consolidated walkforward predictions file.
    Maps columns to standard names used in this script.
    """
    if not SUPER_ENSEMBLE_WALKFORWARD_PATH.exists():
        logging.error(f"Input file not found: {SUPER_ENSEMBLE_WALKFORWARD_PATH}")
        return pd.DataFrame()

    logging.info(f"Loading data from: {SUPER_ENSEMBLE_WALKFORWARD_PATH}")
    df = pd.read_csv(SUPER_ENSEMBLE_WALKFORWARD_PATH)

    rename_map = {
        'Statistical_Trend_pred': 'Statistical Trend_pred',
        'Hybrid_XGB_pred': 'Standalone XGBoost_pred',
        'Robust_Linear_pred': 'Model C (Ridge)_pred',
        'Super_Ensemble_pred': 'Super Ensemble_pred'
    }

    df.rename(columns=rename_map, inplace=True)

    if 'district_no' in df.columns:
        df['district_no'] = df['district_no'].astype(int)

    return df

def diebold_mariano_test(y_true, y_pred_1, y_pred_2, h=1, crit="MAE"):
    """
    Diebold-Mariano test for predictive accuracy.
    H0: Two models have the same predictive accuracy.
    Tests if Model 2 is significantly better than Model 1 (positive statistic).

    Args:
        y_true: Actual values
        y_pred_1: Predictions from Model 1 (Baseline)
        y_pred_2: Predictions from Model 2 (Challenger)
        h: Forecast horizon (default 1)
        crit: Criterion, "MSE" or "MAE"

    Returns:
        dm_stat, p_value (one-sided)
    """
    e1 = y_true - y_pred_1
    e2 = y_true - y_pred_2

    T = float(len(y_true))

    if crit == "MSE":
        d = e1**2 - e2**2
    elif crit == "MAE":
        d = np.abs(e1) - np.abs(e2)

    d_mean = np.mean(d)
    d_var = np.var(d, ddof=0)

    # autocovariance function for Newey-West
    def autocovariance(x, k):
        n = len(x)
        x_mean = np.mean(x)
        return np.sum((x[:n-k] - x_mean) * (x[k:] - x_mean)) / n

    gamma = [autocovariance(d, j) for j in range(h)]
    var_d = d_var + 2 * sum(gamma[1:])

    if var_d > 0:
        dm_stat = d_mean / np.sqrt(var_d / T)
    else:
        dm_stat = 0

    # One-sided p-value (is Model 2 significantly better? i.e. d > 0)
    p_value = 1 - stats.norm.cdf(dm_stat)

    return dm_stat, p_value


def evaluate_timeframe(df, start_year, end_year, title):
    """Calculates strict apples-to-apples metrics for a specific time window."""
    mask = (df['year'] >= start_year) & (df['year'] <= end_year)
    subset = df[mask].copy()

    if subset.empty:
        logging.warning(f"No data for {title}")
        return

    models = [
        'Standalone XGBoost',
        'Model C (Ridge)',
        'Super Ensemble'
    ]

    # Calculate Trend MAE Baseline
    clean_trend = subset.dropna(subset=['Statistical Trend_pred', 'kreisYield'])
    trend_mae_global = mean_absolute_error(clean_trend['kreisYield'], clean_trend['Statistical Trend_pred'])

    logging.info("\n" + "=" * 80)
    logging.info(f"      {title}")
    logging.info("=" * 80)
    logging.info(f"Baseline Trend N-count: {len(clean_trend)}")

    results = []

    results.append({
        'Model': 'Statistical Trend',
        'MAE': trend_mae_global,
        'R2': r2_score(clean_trend['kreisYield'], clean_trend['Statistical Trend_pred']),
        'Skill (%)': 0.00,
        'DM_Stat': np.nan,
        'p_value': np.nan,
        'N_Samples': len(clean_trend),
        'Years_Present': f"{clean_trend['year'].min()}-{clean_trend['year'].max()}"
    })

    for m in models:
        col = f'{m}_pred'
        if col in subset.columns:
            clean = subset.dropna(subset=[col, 'Statistical Trend_pred', 'kreisYield'])
            if clean.empty: continue

            mae = mean_absolute_error(clean['kreisYield'], clean[col])
            r2 = r2_score(clean['kreisYield'], clean[col])

            exact_trend_mae = mean_absolute_error(clean['kreisYield'], clean['Statistical Trend_pred'])

            skill = 0.0
            if exact_trend_mae > 0:
                skill = (1 - (mae / exact_trend_mae)) * 100

            dm_stat, p_val = diebold_mariano_test(
                clean['kreisYield'].values,
                clean['Statistical Trend_pred'].values,
                clean[col].values,
                crit="MAE"
            )

            results.append({
                'Model': m,
                'MAE': mae,
                'R2': r2,
                'Skill (%)': skill,
                'DM_Stat': dm_stat,
                'p_value': p_val,
                'N_Samples': len(clean),
                'Years_Present': f"{clean['year'].min()}-{clean['year'].max()}"
            })

    res_df = pd.DataFrame(results).sort_values('MAE')
    logging.info("\n" + res_df.to_string(index=False, float_format="%.4f"))


def print_anomaly_forensics(df):
    logging.info("\n" + "=" * 80)
    logging.info("      ANOMALY FORENSICS (Black Swan Events)")
    logging.info("=" * 80)

    anomalies = [2003, 2014, 2018, 2022]
    model_map = {
        'TREND': 'Statistical Trend_pred',
        'STANDALONE_XGB': 'Standalone XGBoost_pred',
        'MODEL_C_RIDGE': 'Model C (Ridge)_pred',
        'SUPER_ENSEMBLE': 'Super Ensemble_pred'
    }

    for year in anomalies:
        if year not in df['year'].values:
            logging.warning(f"Year {year} not found in data.")
            continue

        subset = df[df['year'] == year].copy()

        if 'harvested_area' in subset.columns and subset['harvested_area'].sum() > 0:
            actual = np.average(subset['kreisYield'], weights=subset['harvested_area'])
        else:
            actual = subset['kreisYield'].mean()

        logging.info(f"YEAR {year} (Actual National Yield: {actual:.1f} dt/ha)")

        errors = []
        for label, col in model_map.items():
            if col in subset.columns:
                if 'harvested_area' in subset.columns and subset['harvested_area'].sum() > 0:
                     pred_mean = np.average(subset[col], weights=subset['harvested_area'])
                     mae = np.average((subset[col] - subset['kreisYield']).abs(), weights=subset['harvested_area'])
                else:
                     pred_mean = subset[col].mean()
                     mae = (subset[col] - subset['kreisYield']).abs().mean()

                errors.append((label, pred_mean, mae))

        errors.sort(key=lambda x: x[2])

        for label, pred, mae in errors:
            marker = "  "
            if label == 'SUPER_ENSEMBLE': marker = "->"
            logging.info(f"{marker} {label:<16}: Pred {pred:.1f} (Avg MAE: {mae:.1f})")

        logging.info("-" * 40)

def plot_2018_yield_map():
    """Generates Figure 1: 2018 Actual Yield Map."""
    output_path = project_root / 'docs/paper_latex/figures/fig1_regime_map_2018.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not MASTER_DATA_PATH.exists():
        logging.warning(f"Master Dataset not found at {MASTER_DATA_PATH}. Skipping Figure 1.")
        return

    if not GEOJSON_PATH.exists():
        logging.warning(f"GeoJSON not found at {GEOJSON_PATH}. Skipping Figure 1.")
        return

    # Load Data
    df = pd.read_csv(MASTER_DATA_PATH)
    df = df[df['year'] == 2018].copy()

    # Load GeoJSON
    gdf = gpd.read_file(GEOJSON_PATH)

    # Merge
    # Ensure district_no is same type
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    # The GeoJSON uses 'id' for the district code (ARS/RS)
    if 'id' in gdf.columns:
        gdf['id'] = gdf['id'].astype(str).str.zfill(5)
        join_col = 'id'
    elif 'RS' in gdf.columns:
        gdf['RS'] = gdf['RS'].astype(str).str.zfill(5)
        join_col = 'RS'
    else:
        logging.error(f"GeoJSON columns: {gdf.columns}. Cannot find district code (id/RS).")
        return

    merged = gdf.merge(df, left_on=join_col, right_on='district_no', how='inner')

    if merged.empty:
        logging.warning("No matched districts for 2018 map. Check district IDs.")
        return

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged.plot(column='yield', ax=ax, legend=True,
                legend_kwds={'label': "Sugarbeet Yield (dt/ha)", 'orientation': "horizontal"},
                cmap='RdYlGn', edgecolor='black', linewidth=0.5)

    ax.set_title('Figure 1: Observed Sugarbeet Yields (2018 Drought)', fontsize=15)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logging.info(f"✅ Figure 1 (Map) saved to: {output_path}")


def plot_2018_predicted_yield_map(df):
    """Generates Figure 2: 2018 Predicted Yield Map (Super Ensemble)."""
    output_path = project_root / 'docs/paper_latex/figures/fig2_predicted_map_2018.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not GEOJSON_PATH.exists():
        logging.warning(f"GeoJSON not found at {GEOJSON_PATH}. Skipping Predicted Map.")
        return

    # Filter the predictions DataFrame for 2018
    df_2018 = df[df['year'] == 2018].copy()

    if df_2018.empty:
        logging.warning("No prediction data for 2018 found. Skipping Predicted Map.")
        return

    # Load GeoJSON
    gdf = gpd.read_file(GEOJSON_PATH)

    # Ensure district_no is same type (zero-padded string of length 5)
    df_2018['district_no'] = df_2018['district_no'].astype(str).str.zfill(5)

    # The GeoJSON uses 'id' or 'RS' for the district code
    if 'id' in gdf.columns:
        gdf['id'] = gdf['id'].astype(str).str.zfill(5)
        join_col = 'id'
    elif 'RS' in gdf.columns:
        gdf['RS'] = gdf['RS'].astype(str).str.zfill(5)
        join_col = 'RS'
    else:
        logging.error(f"GeoJSON columns: {gdf.columns}. Cannot find district code (id/RS).")
        return

    # Merge spatial data with prediction data
    merged = gdf.merge(df_2018, left_on=join_col, right_on='district_no', how='inner')

    if merged.empty:
        logging.warning("No matched districts for 2018 predicted map. Check district IDs.")
        return

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Note: load_and_merge_models() maps the column to 'Super Ensemble_pred'
    merged.plot(column='Super Ensemble_pred', ax=ax, legend=True,
                legend_kwds={'label': "Predicted Sugarbeet Yield (dt/ha)", 'orientation': "horizontal"},
                cmap='RdYlGn', edgecolor='black', linewidth=0.5)

    ax.set_title('Predicted Sugarbeet Yields (2018 Drought) - Physically-Gated Anomaly Detector', fontsize=15)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logging.info(f"✅ Figure 2 (Predicted Map) saved to: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_merge_models()
    if df.empty:
        logging.error("No data loaded. Exiting.")
        return

    # 1. Long-Term Stability Test (2005-2024)
    evaluate_timeframe(df, 2005, 2024, "LONG-TERM STABILITY (2005-2024)")

    # 2. Recent Volatility Test (2014-2024)
    evaluate_timeframe(df, 2014, 2024, "RECENT VOLATILITY (2014-2024)")

    # 3. Anomaly Check
    print_anomaly_forensics(df)
    plot_2018_yield_map()
    plot_2018_predicted_yield_map(df)

if __name__ == '__main__':
    main()
