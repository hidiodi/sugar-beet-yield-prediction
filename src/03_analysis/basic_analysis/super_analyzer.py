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


def evaluate_feature_stability(df, target_col='kreisYield', min_train_years=10):
    """
    Walk-forward Spearman correlation to expose features that look good in-sample
    but fail out-of-sample. No temporal leaking.
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import spearmanr

    years = sorted(df['year'].unique())
    results = []
    features = [c for c in df.columns if
                c not in ['year', 'district_no', target_col] and pd.api.types.is_numeric_dtype(df[c])]

    for y in years[min_train_years:]:
        train = df[df['year'] < y]
        test = df[df['year'] == y]

        for f in features:
            try:
                train_corr, _ = spearmanr(train[f], train[target_col], nan_policy='omit')
                test_corr, _ = spearmanr(test[f], test[target_col], nan_policy='omit')

                results.append({
                    'test_year': y, 'feature': f,
                    'train_corr': train_corr if not np.isnan(train_corr) else 0,
                    'test_corr': test_corr if not np.isnan(test_corr) else 0,
                    'stability_penalty': abs(train_corr - test_corr) if not (
                                np.isnan(train_corr) or np.isnan(test_corr)) else 1.0
                })
            except:
                continue

    res_df = pd.DataFrame(results)
    summary = res_df.groupby('feature').agg(
        mean_train_corr=('train_corr', 'mean'),
        mean_test_corr=('test_corr', 'mean'),
        volatility=('stability_penalty', 'mean')
    ).sort_values('volatility', ascending=True)

    return summary


def log_walkforward_feature_decay(df, target_col='kreisYield', min_train_years=10):
    import logging
    from scipy.stats import spearmanr
    import numpy as np

    years = sorted(df['year'].unique())
    features = [c for c in df.columns if
                pd.api.types.is_numeric_dtype(df[c]) and c not in ['year', 'district_no', target_col]]

    for f in features:
        train_corrs, test_corrs = [], []
        for y in years[min_train_years:]:
            train, test = df[df['year'] < y], df[df['year'] == y]
            if len(test) < 10 or train[f].nunique() <= 1 or test[f].nunique() <= 1:
                continue

            tr_c, _ = spearmanr(train[f], train[target_col], nan_policy='omit')
            te_c, _ = spearmanr(test[f], test[target_col], nan_policy='omit')

            if not np.isnan(tr_c) and not np.isnan(te_c):
                train_corrs.append(tr_c)
                test_corrs.append(te_c)

        if train_corrs:
            mean_tr, mean_te = np.mean(train_corrs), np.mean(test_corrs)
            decay = abs(mean_tr) - abs(mean_te)
            if decay > 0.15:
                logging.warning(
                    f"Feature {f} is rotting OOS! Train Corr: {mean_tr:.3f} -> Test Corr: {mean_te:.3f}. Decay: {decay:.3f}")
            else:
                logging.info(f"Feature {f} stable. Test Corr: {mean_te:.3f}")


def log_spatial_temporal_gaps(df):
    import logging

    expected_districts = df['district_no'].nunique()
    years = sorted(df['year'].unique())

    for y in years:
        df_year = df[df['year'] == y]
        actual_districts = df_year['district_no'].nunique()

        if actual_districts < expected_districts * 0.9:
            logging.error(f"Year {y}: Missing >10% of NUTS3 districts! Found {actual_districts}/{expected_districts}.")

        missing_ratios = df_year.isnull().mean()
        shit_features = missing_ratios[missing_ratios > 0.05]

        for feat, ratio in shit_features.items():
            logging.warning(
                f"Year {y}: Feature '{feat}' is missing in {ratio * 100:.1f}% of districts. March 1st data is compromised.")


def log_zero_variance_traps(df):
    import logging

    features = [c for c in df.columns if
                pd.api.types.is_numeric_dtype(df[c]) and c not in ['year', 'district_no', 'kreisYield']]

    for f in features:
        zeros_pct = (df[f] == 0).mean()
        if zeros_pct > 0.8:
            logging.critical(
                f"TRAP: Feature '{f}' is 0 for {zeros_pct * 100:.1f}% of the data. The model will overfit this noise.")

        unique_vals = df[f].nunique()
        if unique_vals < 5 and zeros_pct < 0.8:
            logging.info(f"Categorical/Flag identified: '{f}' has {unique_vals} unique values.")

def main():
    df = load_and_prepare_data()
    if df is None:
        return

    corr_df = analyze_correlations(df)
    imp_df = run_multivariate_importance(df)
    check_extreme_regimes(df)
    evaluate_feature_stability(df)
    log_walkforward_feature_decay(df)
    log_spatial_temporal_gaps(df)
    log_zero_variance_traps(df)
    logging.info("\nAnalysis Complete.")


if __name__ == "__main__":
    main()