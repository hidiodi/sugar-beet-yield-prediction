import pandas as pd
import numpy as np
import logging
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
FEATURES_FILE = "data/05_model_input/stage1_preseason_features.csv"
TREND_FILE = "data/05_model_input/wofost_walkforward/final_honest_forecasts.csv"

TREND_COL = 'final_corrected_forecast'
TARGET_COL = 'actual_yield'

# --- HIERARCHY CONFIGURATION ---

# 1. WATER CRASH (Top Priority)
WATER_CRASH_TRIGGER = -0.95
WATER_CRASH_BASE_PENALTY = 0.15
WATER_CRASH_SLOPE = 0.25
# NEW: Resilience Gate. If Tank & Physics are good, we survive the drought.
RESILIENCE_GATE_TANK = 0.5
RESILIENCE_GATE_WOF = -0.5

# 2. HEAT STRESS (Second Priority)
HEAT_TRIGGER = 0.85
HEAT_PENALTY_FIXED = 0.92
HEAT_BUFFER_GATE = 0.2

# 3. BUMPER (Third Priority)
WATER_BUMPER_TRIGGER = 1.0
HEAT_BUMPER_LIMIT = 0.5
# NEW: Solar Gate. Bumper requires decent light.
SOLAR_BUMPER_LIMIT = -1.0
BUMPER_BONUS_BASE = 1.05
BUMPER_SLOPE = 0.05

logging.basicConfig(level=logging.INFO, format='%(message)s')


def calculate_zscores(df):
    df = df.sort_values(['district_no', 'year'])

    def get_z(col):
        if col not in df.columns: return 0
        exp_mean = df.groupby('district_no')[col].transform(lambda x: x.expanding(min_periods=2).mean())
        exp_std = df.groupby('district_no')[col].transform(lambda x: x.expanding(min_periods=2).std())
        return ((df[col] - exp_mean) / exp_std).fillna(0)

    # 1. Water
    if 'summer_water_balance_anomaly' in df.columns:
        df['z_water'] = get_z('summer_water_balance_anomaly')
    else:
        df['z_water'] = get_z('summer_precip_anomaly_forecast')

    # 2. Heat
    df['z_heat'] = get_z('z_heat')

    # 3. Solar (New)
    df['z_solar'] = get_z('summer_solar_rad_anomaly_forecast')

    # 4. Buffers
    df['z_tank_metric'] = get_z('z_tank')
    df['z_wofost_metric'] = get_z('wofost_yield_water_limited')

    return df


def apply_hierarchical_rules(df):
    df['multiplier'] = 1.0
    df['regime'] = 'Normal'

    # --- PRIORITY 1: WATER CRASH (RESILIENCE GATED) ---
    # Logic: Water < -0.95 AND NOT (Tank > 0.5 AND Wofost > -0.5)
    # If buffers are strong, we downgrade to Normal (Trend)
    is_dry = df['z_water'] < WATER_CRASH_TRIGGER
    is_resilient = (df['z_tank_metric'] > RESILIENCE_GATE_TANK) & \
                   (df['z_wofost_metric'] > RESILIENCE_GATE_WOF)

    is_water_crash = is_dry & (~is_resilient)

    def calc_water_penalty(z):
        excess = abs(z - WATER_CRASH_TRIGGER)
        penalty = WATER_CRASH_BASE_PENALTY + (excess * WATER_CRASH_SLOPE)
        return max(0.60, 1.0 - penalty)

    df.loc[is_water_crash, 'multiplier'] = df.loc[is_water_crash, 'z_water'].apply(calc_water_penalty)
    df.loc[is_water_crash, 'regime'] = 'Water Crash'

    # --- PRIORITY 2: HEAT STRESS (BUFFERED) ---
    is_hot = df['z_heat'] > HEAT_TRIGGER
    is_unbuffered = (df['z_water'] < HEAT_BUFFER_GATE) & (df['z_tank_metric'] < HEAT_BUFFER_GATE)

    # Only if not already crashed
    is_heat_stress = is_hot & is_unbuffered & (df['regime'] == 'Normal')

    df.loc[is_heat_stress, 'multiplier'] = HEAT_PENALTY_FIXED
    df.loc[is_heat_stress, 'regime'] = 'Heat Stress'

    # --- PRIORITY 3: BUMPER (SOLAR GATED) ---
    # Logic: Wet AND Cool AND Sunny
    is_wet = df['z_water'] > WATER_BUMPER_TRIGGER
    is_cool = df['z_heat'] < HEAT_BUMPER_LIMIT
    is_sunny = df['z_solar'] > SOLAR_BUMPER_LIMIT  # New Check

    is_bumper = is_wet & is_cool & is_sunny & (df['regime'] == 'Normal')

    def calc_bumper(z):
        excess = z - WATER_BUMPER_TRIGGER
        bonus = (BUMPER_BONUS_BASE - 1.0) + (excess * BUMPER_SLOPE)
        return min(1.15, 1.0 + bonus)

    df.loc[is_bumper, 'multiplier'] = df.loc[is_bumper, 'z_water'].apply(calc_bumper)
    df.loc[is_bumper, 'regime'] = 'Bumper'

    return df


def run_final_pipeline():
    logging.info("--- 🚀 Starting V31 'Solar-Gated' Final Pipeline ---")

    feat_df = pd.read_csv(FEATURES_FILE)
    feat_df['district_no'] = feat_df['district_no'].astype(str).str.zfill(5)

    trend_df = pd.read_csv(TREND_FILE)
    trend_df['district_no'] = trend_df['district_no'].astype(str).str.zfill(5)

    if TARGET_COL not in trend_df.columns:
        df = pd.merge(feat_df, trend_df[['district_no', 'year', TREND_COL]], on=['district_no', 'year'], how='inner')
    else:
        df = pd.merge(feat_df, trend_df[['district_no', 'year', TREND_COL, TARGET_COL]], on=['district_no', 'year'],
                      how='inner')

    df = calculate_zscores(df)
    df = apply_hierarchical_rules(df)

    df['final_pred'] = df[TREND_COL] * df['multiplier']

    results = []
    test_years = [y for y in sorted(df['year'].unique()) if y >= 2014]

    for year in test_years:
        yr_data = df[df['year'] == year]
        trend_mae = (yr_data[TREND_COL] - yr_data[TARGET_COL]).abs().mean()
        hybrid_mae = (yr_data['final_pred'] - yr_data[TARGET_COL]).abs().mean()
        avg_mult = yr_data['multiplier'].mean()
        regime_counts = yr_data['regime'].value_counts().to_dict()

        flag = ""
        if year == 2018:
            flag = "🎯 2018"
        elif year == 2016:
            z_sol = yr_data['z_solar'].mean()
            flag = f"🌥️ 2016 (Solar: {z_sol:.2f})"

        logging.info(
            f"Year {year}: Trend {trend_mae:.1f} | Hybrid {hybrid_mae:.1f} | Imp {trend_mae - hybrid_mae:+.1f} | {regime_counts} {flag}")
        results.append({'trend_mae': trend_mae, 'hybrid_mae': hybrid_mae})

    res_df = pd.DataFrame(results)
    logging.info("\n--- 🏆 V31 FINAL SCORECARD ---")
    logging.info(f"Base Trend MAE: {res_df['trend_mae'].mean():.2f}")
    logging.info(f"Hybrid MAE: {res_df['hybrid_mae'].mean():.2f}")

    # --- NEW CODE START: SAVE TO DISK ---
    output_path = "data/06_model_output/v31_solar_gated_forecast.csv"

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Select key columns and save
    output_cols = ['district_no', 'year', 'final_pred', 'regime']
    df[output_cols].to_csv(output_path, index=False)
    logging.info(f"✅ Saved V31 forecasts to: {output_path}")

if __name__ == "__main__":
    run_final_pipeline()
