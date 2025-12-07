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

# V20 CONFIG: SCENARIO SOFT SWITCH
# We use the Observed Heat (Scenario) to drive the Soft Switch.
# This validates the model's ability to react to KNOWN extreme conditions.
SIGMOID_STEEPNESS = 5.0
SIGMOID_CENTER = 0.5

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def calculate_expanding_z_score(group, col_name, min_periods=5):
    """Academic-Safe Expanding Window Z-Score."""
    group = group.sort_values('year')
    series = group[col_name]
    exp_mean = series.expanding(min_periods=min_periods).mean()
    exp_std = series.expanding(min_periods=min_periods).std()
    exp_std = exp_std.replace(0, 1.0)
    z_scores = (series - exp_mean) / exp_std
    return z_scores.fillna(0)


def sigmoid_weight(z_score, k=SIGMOID_STEEPNESS, x0=SIGMOID_CENTER):
    """Converts a Z-score into a weight between 0.0 and 1.0."""
    return 1 / (1 + np.exp(-k * (z_score - x0)))


def get_smart_indices(df):
    """
    Calculates Risk Indices using SCENARIO DATA (Observed Heat).
    """
    df = df.copy()

    # Check for the Scenario Variable
    if 'summer_days_tmax_gt_30c' not in df.columns:
        logging.warning("Scenario Variable (Heat) not found! Switcher will be blind.")
        df['summer_days_tmax_gt_30c'] = 0.0

    cols = ['summer_days_tmax_gt_30c', 'summer_water_balance_anomaly', 'effective_winter_water']
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
        else:
            df[c] = 0.0

    def apply_district_logic(g):
        # 1. SCENARIO DRIVER: Observed Heat
        z_heat = calculate_expanding_z_score(g, 'summer_days_tmax_gt_30c')

        # 2. Water Drivers
        z_drought = calculate_expanding_z_score(g, 'summer_water_balance_anomaly') * -1
        z_winter = calculate_expanding_z_score(g, 'effective_winter_water') * -1

        # Composite Failure Index (Heat + Drought)
        # In Scenario Mode, we trust the Heat Signal implicitly.
        fail_idx = np.maximum(z_heat, (z_drought + z_winter) / 2.0)

        return pd.DataFrame({
            'year': g['year'],
            'Index_Failure_Local': fail_idx
        })

    indices = df.groupby('district_no').apply(apply_district_logic)

    if isinstance(indices.index, pd.MultiIndex):
        indices = indices.reset_index(level=0)

    cols_to_drop = ['district_no']
    if 'year' in indices.columns: cols_to_drop.append('year')
    indices = indices.drop(columns=cols_to_drop, errors='ignore')

    df_result = pd.merge(df[['year', 'district_no']], indices, left_index=True, right_index=True)

    # Global Aggregation
    annual = df_result.groupby('year')['Index_Failure_Local'].transform('mean')
    df_result['Global_Failure'] = annual

    return df_result


def load_predictions():
    # 1. Trend
    trend_path = config.MODEL_COMPARISON_CONFIG['STATISTICAL_TREND_FILE']
    df_trend = pd.read_csv(trend_path)[['year', 'district_no', 'final_corrected_forecast', 'actual_yield']]
    df_trend.rename(columns={'final_corrected_forecast': 'Trend_Pred', 'actual_yield': 'Actual'}, inplace=True)

    # 2. Native V2 & V8 (Scenario Mode Models)
    native_comp_path = project_root / 'src/models/native_physics_comparison/native_model_comparison_v2_v8.csv'
    if native_comp_path.exists():
        df_native = pd.read_csv(native_comp_path)
        df_native['district_no'] = df_native['district_no'].astype(int)
        df_native = df_native[['year', 'district_no', 'Native_V2', 'Native_V8']]
    else:
        df_native = pd.DataFrame(columns=['year', 'district_no', 'Native_V2', 'Native_V8'])

    # 3. Standalone XGB (Fallback for 2000-2004)
    sa_path = config.MODEL_COMPARISON_CONFIG['STANDALONE_XGB_PREDICTIONS_FILE']
    df_sa = pd.read_csv(sa_path)[['year', 'district_no', 'predicted_yield_median']]
    df_sa.rename(columns={'predicted_yield_median': 'XGB_Median'}, inplace=True)

    # Merge
    df = pd.merge(df_trend, df_sa, on=['year', 'district_no'], how='inner')
    df = pd.merge(df, df_native, on=['year', 'district_no'], how='left')

    # Backfills
    # For 2000-2004, use XGB Median (Best available non-physics model)
    df['Native_V2'] = df['Native_V2'].fillna(df['XGB_Median'])
    df['Native_V8'] = df['Native_V8'].fillna(df['XGB_Median'])

    raw_df = pd.read_csv(config.XGBOOST_TRAINING_CONFIG['DATA_PATH'])
    return df, raw_df


def run_strategy():
    logging.basicConfig(level=LOG_LEVEL, format='%(message)s')
    print("--- REGIME SWITCHING STRATEGY V20 (Scenario Soft Switch) ---")

    pred_df, raw_df = load_predictions()
    pred_df['district_no'] = pred_df['district_no'].astype(int)
    raw_df['district_no'] = raw_df['district_no'].astype(int)

    indices_df = get_smart_indices(raw_df)

    df = pd.merge(pred_df, indices_df[['year', 'district_no', 'Global_Failure']], on=['year', 'district_no'],
                  how='left')

    # --- SOFT VOTING ---
    df['Crisis_Weight'] = sigmoid_weight(df['Global_Failure'])

    # Blend: Weighted Average
    df['Final_Pred'] = (
            df['Crisis_Weight'] * df['Native_V2'] +
            (1.0 - df['Crisis_Weight']) * df['Native_V8']
    )

    # Strategy Mode Label
    df['Strategy_Mode'] = 'Balanced (Mix)'
    df.loc[df['Crisis_Weight'] > 0.7, 'Strategy_Mode'] = 'Crisis-Heavy (>70% V2)'
    df.loc[df['Crisis_Weight'] < 0.3, 'Strategy_Mode'] = 'Opportunity-Heavy (>70% V8)'

    # --- EVALUATION ---
    clean_df = df.dropna(subset=['Actual', 'Final_Pred'])

    # Calculate a "Naive Ensemble" for baseline comparison (50/50 mix)
    naive_ensemble = (clean_df['Native_V2'] + clean_df['Native_V8']) / 2
    mae_base = mean_absolute_error(clean_df['Actual'], naive_ensemble)

    mae_final = mean_absolute_error(clean_df['Actual'], clean_df['Final_Pred'])
    r2_final = r2_score(clean_df['Actual'], clean_df['Final_Pred'])

    print("\nRESULTS V20 (SCENARIO SWITCH):")
    print(f"  Naive Ensemble MAE:   {mae_base:.2f}")
    print(f"  Smart Scenario MAE:   {mae_final:.2f}")
    print(f"  Final R²:             {r2_final:.4f}")
    print(f"  Improvement:          {mae_base - mae_final:.2f} dt/ha")

    print("\nWEIGHT DISTRIBUTION:")
    print(df['Strategy_Mode'].value_counts())

    print("\nCRITICAL YEARS FORENSICS:")
    for y in [2007, 2014, 2018]:
        sub = clean_df[clean_df['year'] == y]
        if len(sub) == 0: continue
        avg_weight = sub['Crisis_Weight'].mean()
        err = mean_absolute_error(sub['Actual'], sub['Final_Pred'])
        print(f"  Year {y}: Avg Crisis Weight = {avg_weight:.2f} | MAE = {err:.1f}")

    out_dir = config.DATA_DIR / '06_model_output'
    df.to_csv(out_dir / 'final_switched_forecast.csv', index=False)


if __name__ == "__main__":
    run_strategy()