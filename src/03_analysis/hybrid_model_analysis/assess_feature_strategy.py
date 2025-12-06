import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config


def calculate_z_score(series):
    series = series.fillna(series.mean())
    std = series.std()
    if std == 0: return pd.Series(0, index=series.index)
    return (series - series.mean()) / std


def simulate_v14_logic(df):
    """
    V14: V13 + Flood Logic.
    """
    df = df.copy()
    groups = df.groupby('district_no')

    # 1. Component Z-Scores
    df['z_heat'] = groups['summer_days_tmax_gt_30c'].transform(calculate_z_score)
    df['z_bal'] = groups['summer_water_balance_anomaly'].transform(calculate_z_score)
    df['z_tank'] = groups['effective_winter_water'].transform(calculate_z_score)
    df['z_rain'] = groups['summer_precip_anomaly_forecast'].transform(calculate_z_score)
    df['z_anoxia'] = groups['anoxia_events'].transform(calculate_z_score)
    df['z_sow'] = groups['sowing_doy'].transform(calculate_z_score)  # High = Late (Bad)

    # 2. FAILURE INDEX (Multi-Mode)

    # Mode A: Scorch (Heat * Dry Balance)
    dryness = (df['z_bal'] * -1).clip(lower=0)
    heat = df['z_heat'].clip(lower=0)

    scorch = np.maximum(
        heat * dryness,
        (heat - 1.5).clip(lower=0) * 2.0
    )

    # Mode B: Drowning (Flood)
    # Threshold 0.8
    drown = (df['z_anoxia'] - 0.8).clip(lower=0) * 2.0

    # Mode C: Late Sowing (Mud)
    # If Sowing is > 1.5 Sigma Late, it's a risk
    late_start = (df['z_sow'] - 1.5).clip(lower=0) * 2.0

    df['Index_Failure'] = np.maximum.reduce([scorch, drown, late_start])

    # 3. BUMPER INDEX (Unchanged)
    water_supply = (df['z_tank'].clip(lower=0) + df['z_rain'].clip(lower=0)) / 2.0
    coolness = (df['z_heat'] * -1).clip(lower=0)

    df['Index_Bumper'] = water_supply * coolness

    return df


def analyze_year(df, year, global_std):
    subset = df[df['year'] == year]
    if subset.empty: return None

    avg_residual = subset['target_residual'].mean()
    yield_z = avg_residual / global_std

    fail_signal = subset['Index_Failure'].mean()
    bump_signal = subset['Index_Bumper'].mean()

    # Actual
    actual_state = "Normal"
    if yield_z > 0.7:
        actual_state = "Bumper (+)"
    elif yield_z < -0.7:
        actual_state = "Crash (-)"

    # Predicted
    predicted_state = "Normal"

    if fail_signal > 0.5:
        predicted_state = "Crash (-)"
    elif bump_signal > 0.30:
        predicted_state = "Bumper (+)"

    if predicted_state == "Bumper (+)" and fail_signal > 0.5:
        predicted_state = "Normal"

    success = (actual_state == predicted_state)
    if actual_state == "Normal" and predicted_state == "Normal": success = True

    return {
        'year': year,
        'actual': actual_state,
        'pred': predicted_state,
        'success': success,
        'Fail': fail_signal,
        'Bump': bump_signal,
        'z_anoxia': subset['z_anoxia'].mean(),
        'z_sow': subset['z_sow'].mean()
    }


def run_analysis():
    print("--- V14 FULL HISTORY AUDIT (1981-2024) ---")
    df = pd.read_csv(config.XGBOOST_TRAINING_CONFIG['DATA_PATH'])
    df['target_residual'] = df['kreisYield'] - df['stage1_forecast']

    df = simulate_v14_logic(df)
    residual_std = df['target_residual'].std()

    print(
        f"\n{'Year':<5} | {'Actual':<11} | {'Pred':<11} | {'Verdict':<4} | {'Fail':<6} | {'Bump':<6} | {'Anoxia':<6} | {'Sow(Z)'}")
    print("-" * 105)

    results = []
    years = sorted(df['year'].unique())

    for year in years:
        res = analyze_year(df, year, residual_std)
        results.append(res)

        icon = "✅" if res['success'] else "❌"
        print(
            f"{year:<5} | {res['actual']:<11} | {res['pred']:<11} | {icon:<7} | {res['Fail']:>5.2f}  | {res['Bump']:>5.2f}  | {res['z_anoxia']:>5.2f}  | {res['z_sow']:>5.2f}")

    # Summary
    df_res = pd.DataFrame(results)
    recent = df_res[df_res['year'] >= 2000]
    print("\n=======================================================")
    print(f"Total Accuracy (1981-2024): {df_res['success'].mean():.1%}")
    print(f"Recent Accuracy (2000-2024): {recent['success'].mean():.1%}")
    print("=======================================================")

    # Forensic 2013 vs 2014
    y13 = df_res[df_res['year'] == 2013].iloc[0]
    y14 = df_res[df_res['year'] == 2014].iloc[0]

    print(f"\n2013 (Crash) -> Fail: {y13['Fail']:.2f}, Anoxia: {y13['z_anoxia']:.2f}, Sowing: {y13['z_sow']:.2f}")
    print(f"2014 (Bumper)-> Fail: {y14['Fail']:.2f}, Anoxia: {y14['z_anoxia']:.2f}, Sowing: {y14['z_sow']:.2f}")


if __name__ == "__main__":
    run_analysis()