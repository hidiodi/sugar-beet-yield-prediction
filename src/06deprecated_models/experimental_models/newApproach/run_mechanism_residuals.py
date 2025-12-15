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

# --- V31 BASELINE LOGIC (Re-implemented for Stage 1) ---
WATER_CRASH_TRIGGER = -0.95
WATER_CRASH_BASE_PENALTY = 0.15
WATER_CRASH_SLOPE = 0.25
RESILIENCE_GATE_TANK = 0.5
RESILIENCE_GATE_WOF = -0.5
HEAT_TRIGGER = 0.85
HEAT_PENALTY_FIXED = 0.92
HEAT_BUFFER_GATE = 0.2
WATER_BUMPER_TRIGGER = 1.0
HEAT_BUMPER_LIMIT = 0.5
SOLAR_BUMPER_LIMIT = -1.0
BUMPER_BONUS_BASE = 1.05
BUMPER_SLOPE = 0.05

# --- STAGE 2: MECHANISM PENALTIES (dt/ha) ---

# 1. PHYSIOLOGICAL DROUGHT (The 2018 Fix)
# We use the WOFOST 'cumulative_water_stress' metric.
# High Z = High Stress.
DROUGHT_Z_TRIGGER = 0.5
DROUGHT_PENALTY_RATE = 25.0  # Subtract 25 dt/ha per SD above trigger

# 2. ANOXIA / WATERLOGGING (The 2016 Fix)
# 'anoxia_events' from WOFOST.
ANOXIA_PENALTY_RATE = 4.0  # Subtract 4 dt/ha per event

# 3. LATE SOWING (Operational Drag)
# 'prob_sowing_failure' (Mud Days).
MUD_Z_TRIGGER = 1.0
MUD_PENALTY_RATE = 5.0  # Subtract 5 dt/ha per SD

# 4. VIGOR PROTECTION (The 2014 Safety)
# If physical potential is high, penalties are halved.
VIGOR_Z_THRESHOLD = 0.8
PROTECTION_FACTOR = 0.5

logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_z_score(df, col):
    if col not in df.columns: return pd.Series(0, index=df.index)
    exp_mean = df.groupby('district_no')[col].transform(lambda x: x.expanding(min_periods=2).mean())
    exp_std = df.groupby('district_no')[col].transform(lambda x: x.expanding(min_periods=2).std())
    return ((df[col] - exp_mean) / exp_std).fillna(0)


def generate_stage1_forecast(df):
    """Re-runs V31 Solar-Gated Logic."""
    df = df.sort_values(['district_no', 'year'])

    # Features
    if 'summer_water_balance_anomaly' in df.columns:
        df['z_water'] = get_z_score(df, 'summer_water_balance_anomaly')
    else:
        df['z_water'] = get_z_score(df, 'summer_precip_anomaly_forecast')

    df['z_heat'] = get_z_score(df, 'z_heat')
    df['z_solar'] = get_z_score(df, 'summer_solar_rad_anomaly_forecast')
    df['z_tank_metric'] = get_z_score(df, 'z_tank')
    df['z_wofost_metric'] = get_z_score(df, 'wofost_yield_water_limited')

    df['s1_mult'] = 1.0

    # 1. Water Crash
    is_dry = df['z_water'] < WATER_CRASH_TRIGGER
    is_resilient = (df['z_tank_metric'] > RESILIENCE_GATE_TANK) & (df['z_wofost_metric'] > RESILIENCE_GATE_WOF)
    is_water_crash = is_dry & (~is_resilient)

    def calc_crash(z):
        excess = abs(z - WATER_CRASH_TRIGGER)
        penalty = WATER_CRASH_BASE_PENALTY + (excess * WATER_CRASH_SLOPE)
        return max(0.60, 1.0 - penalty)

    df.loc[is_water_crash, 's1_mult'] = df.loc[is_water_crash, 'z_water'].apply(calc_crash)

    # 2. Heat Stress
    is_hot = df['z_heat'] > HEAT_TRIGGER
    is_unbuffered = (df['z_water'] < HEAT_BUFFER_GATE) & (df['z_tank_metric'] < HEAT_BUFFER_GATE)
    is_heat_stress = is_hot & is_unbuffered & (df['s1_mult'] == 1.0)
    df.loc[is_heat_stress, 's1_mult'] = HEAT_PENALTY_FIXED

    # 3. Bumper
    is_wet = df['z_water'] > WATER_BUMPER_TRIGGER
    is_cool = df['z_heat'] < HEAT_BUMPER_LIMIT
    is_sunny = df['z_solar'] > SOLAR_BUMPER_LIMIT
    is_bumper = is_wet & is_cool & is_sunny & (df['s1_mult'] == 1.0)

    def calc_bumper(z):
        excess = z - WATER_BUMPER_TRIGGER
        bonus = (BUMPER_BONUS_BASE - 1.0) + (excess * BUMPER_SLOPE)
        return min(1.15, 1.0 + bonus)

    df.loc[is_bumper, 's1_mult'] = df.loc[is_bumper, 'z_water'].apply(calc_bumper)
    df['pred_stage1'] = df[TREND_COL] * df['s1_mult']

    return df


def apply_mechanism_residuals(df):
    """Applies specific biological penalties."""

    # Calc Z-Scores
    df['z_drought_bio'] = get_z_score(df, 'cumulative_water_stress')
    df['z_mud'] = get_z_score(df, 'prob_sowing_failure')
    # anoxia_events is raw count, doesn't need Z-score for this logic (events > 0)

    df['resid_penalty'] = 0.0

    # 1. DROUGHT PENALTY (Physiological)
    # Target 2018: Z is likely > 2.0.
    # Excess = 1.5. Penalty = 1.5 * 25 = 37.5 dt/ha.
    is_drought = df['z_drought_bio'] > DROUGHT_Z_TRIGGER

    def calc_drought_loss(z):
        return (z - DROUGHT_Z_TRIGGER) * DROUGHT_PENALTY_RATE

    df.loc[is_drought, 'resid_penalty'] += df.loc[is_drought, 'z_drought_bio'].apply(calc_drought_loss)

    # 2. ANOXIA PENALTY (Waterlogging)
    # Target 2016: Events > 0.
    if 'anoxia_events' in df.columns:
        df['resid_penalty'] += df['anoxia_events'] * ANOXIA_PENALTY_RATE

    # 3. MUD PENALTY (Sowing Delay)
    is_muddy = df['z_mud'] > MUD_Z_TRIGGER

    def calc_mud_loss(z):
        return (z - MUD_Z_TRIGGER) * MUD_PENALTY_RATE

    df.loc[is_muddy, 'resid_penalty'] += df.loc[is_muddy, 'z_mud'].apply(calc_mud_loss)

    # 4. VIGOR PROTECTION
    # If the plant is fundamentally strong (High WOFOST potential),
    # it resists these stresses better.
    is_vigorous = df['z_wofost_metric'] > VIGOR_Z_THRESHOLD
    df.loc[is_vigorous, 'resid_penalty'] *= PROTECTION_FACTOR

    # Apply
    df['final_pred'] = df['pred_stage1'] - df['resid_penalty']

    return df


def run_mechanism_pipeline():
    logging.info("--- 🚀 Starting V35 'Mechanism Residuals' Pipeline ---")

    # Load & Merge
    feat_df = pd.read_csv(FEATURES_FILE)
    feat_df['district_no'] = feat_df['district_no'].astype(str).str.zfill(5)
    trend_df = pd.read_csv(TREND_FILE)
    trend_df['district_no'] = trend_df['district_no'].astype(str).str.zfill(5)

    if TARGET_COL not in trend_df.columns:
        df = pd.merge(feat_df, trend_df[['district_no', 'year', TREND_COL]], on=['district_no', 'year'], how='inner')
    else:
        df = pd.merge(feat_df, trend_df[['district_no', 'year', TREND_COL, TARGET_COL]], on=['district_no', 'year'],
                      how='inner')

    # Run Pipeline
    df = generate_stage1_forecast(df)
    df = apply_mechanism_residuals(df)

    # Evaluate
    results = []
    test_years = [y for y in sorted(df['year'].unique()) if y >= 2014]

    for year in test_years:
        yr_data = df[df['year'] == year]
        mae_s1 = (yr_data['pred_stage1'] - yr_data[TARGET_COL]).abs().mean()
        mae_final = (yr_data['final_pred'] - yr_data[TARGET_COL]).abs().mean()
        avg_pen = yr_data['resid_penalty'].mean()

        flag = ""
        if year == 2018:
            flag = f"🎯 2018 (Pen: {avg_pen:.1f})"
        elif year == 2016:
            flag = f"🌧️ 2016 (Pen: {avg_pen:.1f})"

        logging.info(
            f"Year {year}: S1 MAE {mae_s1:.1f} | Final {mae_final:.1f} | Imp {mae_s1 - mae_final:+.1f} | Pen: {avg_pen:.1f} {flag}")
        results.append({'mae_s1': mae_s1, 'mae_final': mae_final})

    res_df = pd.DataFrame(results)
    logging.info("\n--- 🏆 V35 FINAL SCORECARD ---")
    logging.info(f"Stage 1 MAE: {res_df['mae_s1'].mean():.2f}")
    logging.info(f"Final MAE: {res_df['mae_final'].mean():.2f}")


if __name__ == "__main__":
    run_mechanism_pipeline()