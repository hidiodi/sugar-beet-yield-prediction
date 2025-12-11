import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config


def calculate_z_score(series):
    series = series.fillna(series.mean())
    std = series.std()
    if std == 0: return pd.Series(0, index=series.index)
    return (series - series.mean()) / std


def ensure_features_exist(df):
    """
    Recreates missing derived features on the fly if they are not in the CSV.
    """
    # 1. Flash Drought Index
    if 'flash_drought_index' not in df.columns:
        if 'summer_temp_anomaly_forecast' in df.columns and 'summer_precip_anomaly_forecast' in df.columns:
            # Reconstruct logic: Heat * Dryness (magnitude)
            is_dry = (df['summer_precip_anomaly_forecast'] < 0).astype(int)
            df['flash_drought_index'] = df['summer_temp_anomaly_forecast'] * df[
                'summer_precip_anomaly_forecast'].abs() * is_dry
        else:
            df['flash_drought_index'] = 0

    # 2. Anoxia Events (WOFOST)
    if 'anoxia_events' not in df.columns:
        # Fallback to Summer Water Balance if WOFOST metric is missing
        if 'summer_water_balance_anomaly' in df.columns:
            # Proxy: High positive balance = Wet feet
            df['anoxia_events'] = (df['summer_water_balance_anomaly'] - 0.5).clip(lower=0) * 5.0
        else:
            df['anoxia_events'] = 0

    return df


def simulate_evap_proxy_logic(df):
    """
    V15: Using Evaporation Forecast as the primary Stress Proxy.
    Hypothesis: Low Forecasted Evap = Drought/Heat Stress (Water Limited).
    """
    df = df.copy()
    groups = df.groupby('district_no')

    # --- 1. Component Z-Scores ---

    # THE NEW HEAT/STRESS PROXY
    # We invert Evap: Low Evap (Negative Z) -> High Stress (Positive Signal)
    if 'summer_evaporation_anomaly_forecast' in df.columns:
        z_evap_raw = groups['summer_evaporation_anomaly_forecast'].transform(calculate_z_score)
        df['z_stress_proxy'] = z_evap_raw * -1.0
    else:
        df['z_stress_proxy'] = 0

    # Tank (Winter Water)
    df['z_tank'] = groups['effective_winter_water'].transform(calculate_z_score)

    # Anoxia (Flood Risk - Keep from V14)
    df['z_anoxia'] = groups['anoxia_events'].transform(calculate_z_score)

    # Flash Drought (Keep from V14)
    df['z_flash'] = groups['flash_drought_index'].transform(calculate_z_score)

    # --- 2. FAILURE INDEX (V15) ---

    # Mode A: The "Dry Burn" (Driven by Low Evap)
    risk_evap = df['z_stress_proxy'].clip(lower=0)

    # Mode B: Flash Shock
    risk_flash = df['z_flash'].clip(lower=0)

    # Mode C: Drowning (Flood)
    risk_flood = (df['z_anoxia'] - 0.8).clip(lower=0) * 1.5

    # Combined Risk
    df['Index_Failure'] = np.maximum.reduce([risk_evap, risk_flash, risk_flood])

    # --- 3. BUMPER INDEX (V15) ---
    # High Evap (Good Growing Conditions) + High Tank

    if 'summer_evaporation_anomaly_forecast' in df.columns:
        growth_conditions = z_evap_raw.clip(lower=0)
    else:
        growth_conditions = 0

    tank_reserve = df['z_tank'].clip(lower=0)

    df['Index_Bumper'] = (growth_conditions + tank_reserve) / 2.0

    return df


def analyze_year(df, year, global_std):
    subset = df[df['year'] == year]
    if subset.empty: return None

    avg_residual = subset['target_residual'].mean()
    yield_z = avg_residual / global_std

    fail_signal = subset['Index_Failure'].mean()
    bump_signal = subset['Index_Bumper'].mean()

    # Actual State
    actual_state = "Normal"
    if yield_z > 0.7:
        actual_state = "Bumper (+)"
    elif yield_z < -0.7:
        actual_state = "Crash (-)"

    # Predicted State
    predicted_state = "Normal"

    # Logic V15
    if fail_signal > 0.6:
        predicted_state = "Crash (-)"
    elif bump_signal > 0.4:
        predicted_state = "Bumper (+)"

    # Conflict
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
        'z_stress': subset['z_stress_proxy'].mean()
    }


def run_analysis():
    print("--- V15: EVAPORATION PROXY STRATEGY (Robust) ---")

    try:
        df = pd.read_csv(config.XGBOOST_TRAINING_CONFIG['DATA_PATH'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Calculate Residuals
    if 'stage1_forecast' in df.columns:
        df['target_residual'] = df['kreisYield'] - df['stage1_forecast']
    else:
        df['target_residual'] = df['kreisYield'] - df['kreisYield'].mean()  # Fallback

    # 1. Ensure Features Exist (On-the-fly calculation)
    df = ensure_features_exist(df)

    # 2. Simulate V15 Logic
    df = simulate_evap_proxy_logic(df)

    residual_std = df['target_residual'].std()

    print(f"\n{'Year':<5} | {'Actual':<11} | {'Pred':<11} | {'Verdict':<4} | {'Fail':<6} | {'Bump':<6} | {'Stress(Z)'}")
    print("-" * 90)

    results = []
    # 2000-2024
    years = [y for y in sorted(df['year'].unique()) if y >= 2000]

    for year in years:
        res = analyze_year(df, year, residual_std)
        results.append(res)
        icon = "✅" if res['success'] else "❌"
        print(
            f"{year:<5} | {res['actual']:<11} | {res['pred']:<11} | {icon:<7} | {res['Fail']:>5.2f}  | {res['Bump']:>5.2f}  | {res['z_stress']:>6.2f}")

    # Summary
    df_res = pd.DataFrame(results)
    print("\n=======================================================")
    print(f"Recent Accuracy (2000-2024): {df_res['success'].mean():.1%}")
    print("=======================================================")

    # Forensic
    if not df_res.empty:
        for y in [2003, 2014, 2018]:
            if y in df_res['year'].values:
                row = df_res[df_res['year'] == y].iloc[0]
                print(f"Year {y}: Fail={row['Fail']:.2f}, Bump={row['Bump']:.2f}, Stress_Proxy={row['z_stress']:.2f}")


if __name__ == "__main__":
    run_analysis()