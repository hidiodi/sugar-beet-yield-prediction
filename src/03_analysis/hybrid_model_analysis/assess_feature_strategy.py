import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config


def calculate_z_score(series):
    series = series.fillna(series.mean())
    std = series.std()
    if std == 0: return pd.Series(0, index=series.index)
    return (series - series.mean()) / std


def simulate_smart_features(df):
    df = df.copy()
    groups = df.groupby('district_no')

    # 1. Base Z-Scores (Standardized Context)
    # Heat (High = Bad)
    df['z_heat'] = groups['summer_days_tmax_gt_30c'].transform(calculate_z_score)
    # Water Balance (Inverted: High = Dry = Bad)
    df['z_drought'] = groups['summer_water_balance_anomaly'].transform(calculate_z_score) * -1
    # Anoxia (High = Bad)
    df['z_anoxia'] = groups['anoxia_events'].transform(calculate_z_score)
    # Winter Tank (High = Good)
    df['z_tank'] = groups['effective_winter_water'].transform(calculate_z_score)
    # Solar (High = Good)
    df['z_solar'] = groups['summer_solar_rad_anomaly_forecast'].transform(calculate_z_score)

    # 2. Local Indices

    # FAILURE: Max Risk (Heat OR Drought OR Anoxia)
    # We clip negative Z-scores to 0 (Good weather doesn't subtract from failure risk)
    df['Index_Failure_Local'] = np.maximum.reduce([
        df['z_heat'].clip(lower=0),
        df['z_drought'].clip(lower=0),
        (df['z_anoxia'] - 0.5).clip(lower=0)  # Anoxia needs to be significant
    ])

    # BUMPER: All Systems Go (Water AND Coolness) + Solar Bonus
    # Water Supply: Tank OR Rain (Low Drought)
    water_avail = np.maximum(df['z_tank'], (df['z_drought'] * -1))
    coolness = (df['z_heat'] * -1)

    # We use Average instead of Min to be softer, but penalize negatives
    base_growth = (water_avail + coolness) / 2.0

    df['Index_Bumper_Local'] = base_growth + (df['z_solar'].clip(lower=0) * 0.5)

    # 3. Global Context (The "Classifier" Stage)
    # Calculate annual averages
    annual_stats = df.groupby('year')[['Index_Failure_Local', 'Index_Bumper_Local']].transform('mean')
    df['Global_Failure'] = annual_stats['Index_Failure_Local']
    df['Global_Bumper'] = annual_stats['Index_Bumper_Local']

    return df


def run_analysis():
    print("--- SMARTER FEATURE ASSESSMENT (Z-Scores + Global Context) ---")
    df = pd.read_csv(config.XGBOOST_TRAINING_CONFIG['DATA_PATH'])

    # Target: Residuals (Yield - Trend)
    df['target_residual'] = df['kreisYield'] - df['stage1_forecast']

    df = simulate_smart_features(df)

    print(f"\n{'Year':<5} | {'Actual':<10} | {'Global Fail':<12} | {'Global Bump':<12} | {'Verdict'}")
    print("-" * 65)

    years_of_interest = sorted(df['year'].unique())
    correct_count = 0

    for year in years_of_interest:
        subset = df[df['year'] == year]
        if subset.empty: continue

        # 1. Determine Reality
        actual_res = subset['target_residual'].mean()
        status = "Normal"
        if actual_res < -50: status = "CRASH"
        if actual_res > 50: status = "BUMPER"

        # 2. Get Signals
        g_fail = subset['Global_Failure'].mean()
        g_bump = subset['Global_Bumper'].mean()

        # 3. Apply Smarter Logic
        pred = "Normal"

        # RULE 1: The Hard Cliff (Massive Stress = Death)
        if g_fail > 1.25:
            pred = "CRASH"

        # RULE 2: The Fragile Zone (Moderate Stress + No Recovery = Death)
        elif g_fail > 0.5 and g_bump < 0.2:
            pred = "CRASH"

        # RULE 3: The Clean Win (High Growth + Low Stress = Bumper)
        elif g_bump > 0.8 and g_fail < 0.4:
            pred = "BUMPER"

        # -------------------------

        match = "✅" if status == pred else "❌"
        if match == "✅": correct_count += 1

        print(f"{year:<5} | {status:<10} | {g_fail:>6.2f} (Z)     | {g_bump:>6.2f} (Z)     | {match}")
    print("-" * 85)
    print(f"Accuracy: {correct_count}/{len(years_of_interest)} ({correct_count/len(years_of_interest):.1%})")

if __name__ == "__main__":
    run_analysis()