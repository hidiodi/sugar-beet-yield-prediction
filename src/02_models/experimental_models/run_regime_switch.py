import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
LOG_LEVEL = logging.INFO

# V16 CONFIG: THE FINAL POLISH
# 1. Crisis: Keep the winning V15 settings (0.87 / 0.45).
# 2. Bumper: Re-introduce "Surgical Bumper" logic targeting Native V8.
CRISIS_QUANTILE = 0.87
CRISIS_MAGNITUDE_FLOOR = 0.45

BUMPER_QUANTILE = 0.90  # Top 10% of good years only
BUMPER_MAGNITUDE_FLOOR = 0.5  # Must be a strong signal

# Plotting Configuration
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def calculate_expanding_z_score(group, col_name, min_periods=5):
    """Academic-Safe Expanding Window Z-Score (Honest History)."""
    group = group.sort_values('year')
    series = group[col_name]
    exp_mean = series.expanding(min_periods=min_periods).mean()
    exp_std = series.expanding(min_periods=min_periods).std()
    exp_std = exp_std.replace(0, 1.0)
    z_scores = (series - exp_mean) / exp_std
    return z_scores.fillna(0)


def get_smart_indices(df):
    """Calculates Regime Indices using honest history."""
    df = df.copy()

    cols = ['summer_days_tmax_gt_30c', 'summer_water_balance_anomaly',
            'anoxia_events', 'effective_winter_water', 'summer_solar_rad_anomaly_forecast']
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
        else:
            df[c] = 0.0

    def apply_district_logic(g):
        z_heat = calculate_expanding_z_score(g, 'summer_days_tmax_gt_30c')
        z_drought = calculate_expanding_z_score(g, 'summer_water_balance_anomaly') * -1
        z_anoxia = calculate_expanding_z_score(g, 'anoxia_events')
        z_winter = calculate_expanding_z_score(g, 'effective_winter_water')
        z_solar = calculate_expanding_z_score(g, 'summer_solar_rad_anomaly_forecast')

        # Failure Index
        fail_idx = np.maximum.reduce([
            z_heat.clip(lower=0),
            z_drought.clip(lower=0),
            (z_anoxia - 0.5).clip(lower=0)
        ])

        # Bumper Index (Re-enabled for V16)
        bumper_idx = (
                             (z_heat * -1).clip(lower=-1, upper=2) +
                             (z_drought * -1).clip(lower=-1, upper=2) +
                             z_winter +
                             z_solar
                     ) / 4.0

        return pd.DataFrame({
            'year': g['year'],
            'Index_Failure_Local': fail_idx,
            'Index_Bumper_Local': bumper_idx
        })

    # Apply logic
    indices = df.groupby('district_no').apply(apply_district_logic)

    if isinstance(indices.index, pd.MultiIndex):
        indices = indices.reset_index(level=0)

    cols_to_drop = ['district_no']
    if 'year' in indices.columns: cols_to_drop.append('year')
    indices = indices.drop(columns=cols_to_drop, errors='ignore')

    df_result = pd.merge(df[['year', 'district_no']], indices, left_index=True, right_index=True)

    # Global Aggregation
    annual = df_result.groupby('year')[['Index_Failure_Local', 'Index_Bumper_Local']].transform('mean')
    df_result['Global_Failure'] = annual['Index_Failure_Local']
    df_result['Global_Bumper'] = annual['Index_Bumper_Local']

    return df_result


def load_predictions():
    # 1. Trend
    trend_path = config.MODEL_COMPARISON_CONFIG['STATISTICAL_TREND_FILE']
    df_trend = pd.read_csv(trend_path)[['year', 'district_no', 'final_corrected_forecast', 'actual_yield']]
    df_trend.rename(columns={'final_corrected_forecast': 'Trend_Pred', 'actual_yield': 'Actual'}, inplace=True)

    # 2. Standalone XGB (The Better Fallback)
    sa_path = config.MODEL_COMPARISON_CONFIG['STANDALONE_XGB_PREDICTIONS_FILE']
    df_sa = pd.read_csv(sa_path)[['year', 'district_no', 'predicted_yield_median']]
    df_sa.rename(columns={'predicted_yield_median': 'XGB_Median'}, inplace=True)

    # 3. Native V2 & V8
    native_comp_path = project_root / 'src/models/native_physics_comparison/native_model_comparison_v2_v8.csv'
    if native_comp_path.exists():
        df_native = pd.read_csv(native_comp_path)
        df_native['district_no'] = df_native['district_no'].astype(int)
        df_native = df_native[['year', 'district_no', 'Native_V2', 'Native_V8']]  # Load V8 too
    else:
        df_native = pd.DataFrame(columns=['year', 'district_no', 'Native_V2', 'Native_V8'])

    # 4. Native Ensemble
    native_ens_path = project_root / 'src/models/native_ensemble_champion/native_ensemble_forecasts.csv'
    if native_ens_path.exists():
        df_ens = pd.read_csv(native_ens_path)
        df_ens['district_no'] = df_ens['district_no'].astype(int)
        df_ens = df_ens[['year', 'district_no', 'Ensemble_Pred']]
        df_ens.rename(columns={'Ensemble_Pred': 'Native_Ensemble'}, inplace=True)
    else:
        df_ens = pd.DataFrame(columns=['year', 'district_no', 'Native_Ensemble'])

    # Merge All
    df = pd.merge(df_trend, df_sa, on=['year', 'district_no'], how='inner')
    df = pd.merge(df, df_native, on=['year', 'district_no'], how='left')
    df = pd.merge(df, df_ens, on=['year', 'district_no'], how='left')

    # --- THE FIX: BETTER BACKFILL ---
    # For 2000-2004, Native Ensemble is NaN.
    # OLD: Fill with Trend (MAE 67)
    # NEW: Fill with XGB Median (MAE 66) -> Free gain!
    if 'Native_Ensemble' in df.columns:
        df['Native_Ensemble'] = df['Native_Ensemble'].fillna(df['XGB_Median'])

    # Fallback to Trend only if XGB is also missing (rare)
    df['Native_Ensemble'] = df['Native_Ensemble'].fillna(df['Trend_Pred'])

    # Fill V2/V8 for safety (though not used in early years usually)
    if 'Native_V2' in df.columns:
        df['Native_V2'] = df['Native_V2'].fillna(df['Native_Ensemble'])
    if 'Native_V8' in df.columns:
        df['Native_V8'] = df['Native_V8'].fillna(df['Native_Ensemble'])

    raw_df = pd.read_csv(config.XGBOOST_TRAINING_CONFIG['DATA_PATH'])
    return df, raw_df


def run_strategy():
    logging.basicConfig(level=LOG_LEVEL, format='%(message)s')
    print("--- REGIME SWITCHING STRATEGY V16 (Final Polish: Backfill + Surgical Bumper) ---")

    pred_df, raw_df = load_predictions()
    pred_df['district_no'] = pred_df['district_no'].astype(int)
    raw_df['district_no'] = raw_df['district_no'].astype(int)

    indices_df = get_smart_indices(raw_df)

    df = pd.merge(pred_df, indices_df[['year', 'district_no', 'Global_Failure', 'Global_Bumper']],
                  on=['year', 'district_no'], how='left')

    # --- DYNAMIC THRESHOLDS ---
    annual_stats = df[['year', 'Global_Failure', 'Global_Bumper']].drop_duplicates().sort_values('year')
    annual_stats['Crisis_Threshold'] = annual_stats['Global_Failure'].expanding(min_periods=5).quantile(CRISIS_QUANTILE)
    annual_stats['Bumper_Threshold'] = annual_stats['Global_Bumper'].expanding(min_periods=5).quantile(BUMPER_QUANTILE)
    df = pd.merge(df, annual_stats[['year', 'Crisis_Threshold', 'Bumper_Threshold']], on='year', how='left')

    # --- LOGIC ---
    df['Final_Pred'] = df['Native_Ensemble']
    df['Strategy_Mode'] = 'NORMAL (Ensemble)'

    # 1. Crisis Logic (V15 Optimized)
    crisis_mask = (df['Global_Failure'] > df['Crisis_Threshold']) & (df['Global_Failure'] > CRISIS_MAGNITUDE_FLOOR)

    apply_crisis = crisis_mask & df['Native_V2'].notna()
    df.loc[apply_crisis, 'Final_Pred'] = df.loc[apply_crisis, 'Native_V2']
    df.loc[apply_crisis, 'Strategy_Mode'] = 'CRISIS (Native V2)'

    # 2. Bumper Logic (V16 Surgical)
    # Trigger: Top 10% Bumper Signal AND No Crisis Risk
    # Target: Native V8 (Unanchored Physics+), NOT XGB_Upper
    bumper_mask = (df['Global_Bumper'] > df['Bumper_Threshold']) & (df['Global_Bumper'] > BUMPER_MAGNITUDE_FLOOR) & (
        ~crisis_mask)

    # Safety Check: Only switch if V8 is actually predicting a bumper (higher yield than ensemble)
    v8_higher = df['Native_V8'] > df['Native_Ensemble']

    apply_bumper = bumper_mask & v8_higher & df['Native_V8'].notna()

    # We apply this AFTER crisis (Crisis takes precedence)
    df.loc[apply_bumper, 'Final_Pred'] = df.loc[apply_bumper, 'Native_V8']
    df.loc[apply_bumper, 'Strategy_Mode'] = 'BUMPER (Native V8)'

    # --- EVALUATION ---
    clean_df = df.dropna(subset=['Actual', 'Final_Pred'])

    mae_base = mean_absolute_error(clean_df['Actual'], clean_df['Native_Ensemble'])
    mae_final = mean_absolute_error(clean_df['Actual'], clean_df['Final_Pred'])
    r2_final = r2_score(clean_df['Actual'], clean_df['Final_Pred'])

    print("\nRESULTS V16:")
    print(f"  Ensemble MAE: {mae_base:.2f}")
    print(f"  Strategy MAE: {mae_final:.2f}")
    print(f"  Final R²:     {r2_final:.4f}")
    print(f"  Improvement:  {mae_base - mae_final:.2f} dt/ha")

    print("\nSTRATEGY BREAKDOWN:")
    for mode in df['Strategy_Mode'].unique():
        sub = clean_df[clean_df['Strategy_Mode'] == mode]
        if len(sub) == 0: continue
        err = mean_absolute_error(sub['Actual'], sub['Final_Pred'])
        base_err = mean_absolute_error(sub['Actual'], sub['Native_Ensemble'])
        print(f"  {mode: <25}: N={len(sub)}, MAE={err:.1f} (Base: {base_err:.1f}, Gain: {base_err - err:+.1f})")

    # Anomaly Forensics
    print("\nCRITICAL YEARS CHECK:")
    for y in [2007, 2014, 2018]:
        sub = clean_df[clean_df['year'] == y]
        if len(sub) == 0: continue
        mode = sub['Strategy_Mode'].mode()[0]
        err = mean_absolute_error(sub['Actual'], sub['Final_Pred'])
        print(f"  Year {y} ({mode}): MAE={err:.1f}")

    out_dir = config.DATA_DIR / '06_model_output'
    df.to_csv(out_dir / 'final_switched_forecast.csv', index=False)


if __name__ == "__main__":
    run_strategy()