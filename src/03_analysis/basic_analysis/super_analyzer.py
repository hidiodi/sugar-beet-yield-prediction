import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# --- Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config as global_config
import importlib

models_config = importlib.import_module("src.02_models.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Adjust paths if necessary based on your current pipeline state
STAGE1_PATH = models_config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
STAGE2_PATH = global_config.DATA_DIR / '05_model_input/stage2_refined_features.csv'
OUTPUT_DIR = global_config.DATA_DIR / '06_model_output/analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    logging.info("--- Loading Data for March 1st Predictability Analysis ---")

    if not STAGE1_PATH.exists():
        logging.error(f"Stage 1 file missing: {STAGE1_PATH}")
        return None
    df = pd.read_csv(STAGE1_PATH)

    if STAGE2_PATH.exists():
        df2 = pd.read_csv(STAGE2_PATH)
        df = pd.merge(df, df2, on=['year', 'district_no'], how='left')
    else:
        logging.warning("Stage 2 features missing, proceeding with Stage 1 only.")

    # We need Actual Yield and a Baseline (Trend or Stage 1 Forecast)
    if 'kreisYield' not in df.columns:
        logging.error("Target 'kreisYield' is missing.")
        return None

    trend_col = 'stage1_forecast' if 'stage1_forecast' in df.columns else 'trend_forecast'
    if trend_col not in df.columns:
        logging.error("No trend/baseline forecast column found.")
        return None

    # Filter out missing values and calculate targets
    df = df.dropna(subset=['kreisYield', trend_col])
    df = df[df[trend_col] > 0]  # Prevent division by zero

    # Calculate the Climate Target: Yield Ratio
    df['yield_ratio'] = df['kreisYield'] / df[trend_col]
    df['yield_residual'] = df['kreisYield'] - df[trend_col]

    logging.info(
        f"Loaded {len(df)} records. Target: Yield Ratio (mean: {df['yield_ratio'].mean():.3f}, std: {df['yield_ratio'].std():.3f})")
    return df


def analyze_correlations(df):
    logging.info("\n--- 1. Bivariate Correlation Analysis (Spearman Rank) ---")
    logging.info("Target: Yield Ratio (Actual / Trend). Positive correlation means feature increases yield.")

    # Define feature groups we want to investigate
    feature_groups = {
        "ECMWF Forecasts (Summer)": [
            'summer_precip_anomaly_forecast', 'summer_temp_anomaly_forecast',
            'summer_solar_rad_anomaly_forecast', 'summer_water_balance_anomaly',
            'pred_days', 'mv_days', 'z_rain'
        ],
        "March 1st Observables (Winter/Spring)": [
            'effective_winter_water', 'winter_precip_sum', 'z_tank',
            'mild_winter_days', 'sowing_doy', 'feb_frost_days', 'spring_precip_anomaly_forecast'
        ],
        "Simulated/Composite Indexes": [
            'wofost_yield_water_limited', 'solar_capture_potential',
            'VegetationVigorIndex', 'RootZoneDepletion', 'Index_Failure', 'Index_Bumper'
        ],
        "Potential Leakage Risk (Observed Summer Data)": [
            'summer_days_tmax_gt_30c', 'heat_stress_sq', 'z_heat'
        ]
    }

    results = []
    for group_name, cols in feature_groups.items():
        logging.info(f"\n[{group_name}]")
        for col in cols:
            if col in df.columns:
                clean_df = df.dropna(subset=[col, 'yield_ratio'])
                if len(clean_df) > 50:
                    corr, p_val = spearmanr(clean_df[col], clean_df['yield_ratio'])
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    logging.info(f"{col:<35} : {corr:>6.3f} {sig}")
                    results.append({'Feature': col, 'Group': group_name, 'Correlation': corr, 'P-val': p_val})
            else:
                pass  # Feature not in dataset

    return pd.DataFrame(results)


def run_multivariate_importance(df):
    logging.info("\n--- 2. Multivariate Feature Importance (Random Forest) ---")

    # Select numeric features excluding targets and IDs
    exclude_cols = ['year', 'district_no', 'kreisYield', 'yield_ratio', 'yield_residual',
                    'stage1_forecast', 'trend_forecast', 'is_gdr', 'state_encoded']

    features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    # Drop columns with too many NaNs
    df_clean = df.dropna(subset=['yield_ratio'])
    features = [f for f in features if df_clean[f].isna().mean() < 0.1]
    df_clean = df_clean.fillna(0)

    X = df_clean[features]
    y = df_clean['yield_ratio']

    rf = RandomForestRegressor(n_estimators=150, max_depth=7, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    logging.info("Top 15 Drivers of Yield Ratio (Multivariate):")
    for idx, row in importances.head(15).iterrows():
        logging.info(f"{row['Feature']:<35} : {row['Importance']:.4f}")

    return importances


def check_extreme_regimes(df):
    logging.info("\n--- 3. Extreme Regime Analysis ---")

    stress_mask = df['yield_ratio'] < 0.85
    bumper_mask = df['yield_ratio'] > 1.10
    normal_mask = (df['yield_ratio'] >= 0.85) & (df['yield_ratio'] <= 1.10)

    logging.info(f"Crash Years (<85% trend): {stress_mask.sum()} samples")
    logging.info(f"Bumper Years (>110% trend): {bumper_mask.sum()} samples")
    logging.info(f"Normal Years: {normal_mask.sum()} samples")

    key_metrics = ['summer_water_balance_anomaly', 'effective_winter_water', 'summer_days_tmax_gt_30c', 'Index_Failure']

    for metric in key_metrics:
        if metric in df.columns:
            crash_mean = df.loc[stress_mask, metric].mean()
            norm_mean = df.loc[normal_mask, metric].mean()
            bump_mean = df.loc[bumper_mask, metric].mean()
            logging.info(f"\nMean of [{metric}]:")
            logging.info(f"  Crash:  {crash_mean:.2f}")
            logging.info(f"  Normal: {norm_mean:.2f}")
            logging.info(f"  Bumper: {bump_mean:.2f}")


def main():
    df = load_and_prepare_data()
    if df is None:
        return

    corr_df = analyze_correlations(df)
    imp_df = run_multivariate_importance(df)
    check_extreme_regimes(df)

    logging.info("\nAnalysis Complete.")


if __name__ == "__main__":
    main()