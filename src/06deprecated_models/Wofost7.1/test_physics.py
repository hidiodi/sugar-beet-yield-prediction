import pandas as pd
import sys
from pathlib import Path

# --- Setup Paths ---
# This matches the path defined in config.WOFOST_CONFIG['FILE_PATHS']['OUTPUT_DIR']
output_path = Path("data/06_model_output/multi_year_final/forecast_extreme_weather_metrics.csv")


def main():
    print(f"--- Verifying WOFOST Output at: {output_path} ---")

    if not output_path.exists():
        print("❌ FATAL: File not found! The run might have failed or saved elsewhere.")
        return

    df = pd.read_csv(output_path)
    print(f"✅ File loaded. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # --- 1. Check for New Physics Columns ---
    new_cols = ['anoxia_events', 'prob_sowing_failure', 'prob_terminal_freeze', 'harvest_respiration_risk']
    missing = [c for c in new_cols if c not in df.columns]

    if missing:
        print(f"❌ FAILED: Missing new columns: {missing}")
        return
    else:
        print("✅ Column Structure: OK (New metrics found)")

    # --- 2. The Physics "Sanity Check" (2014 vs 2018) ---
    print("\n--- PHYSICS REALITY CHECK ---")

    # 2014: The Wet Year (Bumper Crop, but some waterlogging)
    # We expect Anoxia > 0, Drought near 0
    df_2014 = df[df['year'] == 2014]
    avg_anoxia_14 = df_2014['anoxia_events'].mean()
    avg_drought_14 = df_2014[
        'cumulative_water_stress_mean'].mean() if 'cumulative_water_stress_mean' in df.columns else 0

    # 2018: The Dry Year (Crash)
    # We expect Anoxia near 0, Drought High
    df_2018 = df[df['year'] == 2018]
    avg_anoxia_18 = df_2018['anoxia_events'].mean()
    avg_drought_18 = df_2018[
        'cumulative_water_stress_mean'].mean() if 'cumulative_water_stress_mean' in df.columns else 0

    print(f"YEAR 2014 (Wet):")
    print(f"  -> Avg Anoxia Events: {avg_anoxia_14:.4f} days (Expect > 0)")
    print(f"  -> Avg Drought Stress: {avg_drought_14:.4f}")

    print(f"YEAR 2018 (Dry):")
    print(f"  -> Avg Anoxia Events: {avg_anoxia_18:.4f} days (Expect near 0)")
    print(f"  -> Avg Drought Stress: {avg_drought_18:.4f}")

    # --- 3. Sowing Check ---
    sowing_fails = df['prob_sowing_failure'].sum()
    print(f"\nTotal Sowing Failures Detected (All Years): {sowing_fails}")
    if sowing_fails == 0:
        print("⚠️ WARNING: Sowing failure prob is 0 everywhere. Thresholds might be too loose.")
    else:
        print("✅ Sowing Logic: Active")

    # --- 4. Final Verdict ---
    if avg_anoxia_14 > avg_anoxia_18:
        print("\n✅ SUCCESS: The model correctly identified 2014 as wetter/riskier for Anoxia than 2018.")
        print("You are ready to proceed to Feature Engineering.")
    else:
        print("\n❌ FAILURE: Physics inverted or flat. Check logs.")


if __name__ == "__main__":
    main()