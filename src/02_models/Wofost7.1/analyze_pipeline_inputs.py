# File: scripts/analyze_pipeline_inputs.py
# Description: A comprehensive diagnostic tool to validate the key data assets
# for the WOFOST pipeline.
# VERSION 5: Now includes a sanity check of the generated forecast weather data
# to detect common unit errors (e.g., cm vs mm) that cause silent failures.

import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# --- Define paths to all data assets that need checking ---
PATHS = {
    'yield': 'data/02_intermediate/sugarbeet_yield.csv',
    'soil': 'data/03_processed/static_features_districts.csv',
    'geo': 'data/01_raw/districts_official.geojson',
    'initial_conditions': 'data/03_processed/InitialConditions.csv',
    'forecast_weather_dir': 'data/03_processed/forecast_weather_parts'  # NEW
}


def run_diagnostics():
    """
    Loads all necessary data assets and runs a series of validation checks.
    """
    logging.info("--- Starting Comprehensive WOFOST Pipeline Input Diagnosis (V5) ---")
    error_messages = []
    data_sources = {}  # To store loaded dataframes

    # --- Phase 1 & 2: Load Data and Check for Mismatches (Combined for efficiency) ---
    logging.info("\n--- [Phase 1/2] Loading Core Inputs & Checking ID Consistency ---")
    # ... (This section is the same as your V4 script for loading yield, soil, geo) ...
    # For brevity, this part is condensed. Your existing loading logic is fine.
    try:
        # Load yield
        df_yield = pd.read_csv(PATHS['yield'], dtype={'district_no': str})
        df_yield['district_no'] = df_yield['district_no'].str.zfill(5)
        yield_ids = set(df_yield['district_no'].unique())
        # Load soil
        df_soil = pd.read_csv(PATHS['soil'], dtype={'district_no': str})
        df_soil['district_no'] = df_soil['district_no'].str.zfill(5)
        soil_ids = set(df_soil['district_no'].unique())
        # Load geo
        gdf_geo = gpd.read_file(PATHS['geo'])
        gdf_geo.rename(columns={'id': 'district_no'}, inplace=True, errors='ignore')
        gdf_geo['district_no'] = gdf_geo['district_no'].astype(str).str.zfill(5)
        geo_ids = set(gdf_geo['district_no'].unique())

        print(f"[SUMMARY] Unique District Counts: Yield={len(yield_ids)}, Soil={len(soil_ids)}, Geo={len(geo_ids)}")

        missing_from_soil = sorted(list(yield_ids.difference(soil_ids)))
        if missing_from_soil:
            msg = f"ID Mismatch: {len(missing_from_soil)} districts are in yield data but MISSING from soil data. List: {missing_from_soil}"
            error_messages.append({'type': 'mismatch', 'text': msg})

        missing_from_geo = sorted(list(yield_ids.difference(geo_ids)))
        if missing_from_geo:
            msg = f"ID Mismatch: {len(missing_from_geo)} districts are in yield data but MISSING from geo data. List: {missing_from_geo}"
            error_messages.append({'type': 'mismatch', 'text': msg})

    except FileNotFoundError as e:
        msg = f"FATAL: Core input file not found: {e}. Cannot continue."
        error_messages.append({'type': 'fatal', 'text': msg})
        sys.exit(msg)  # Exit early if core files are missing

    # --- Phase 3: Sanity-Checking 'InitialConditions.csv' ---
    logging.info("\n--- [Phase 3] Sanity-Checking 'InitialConditions.csv' ---")
    try:
        ic_df = pd.read_csv(PATHS['initial_conditions'], dtype={'district_no': str}, parse_dates=['sowing_date'])
        mismatched_sowing_dates = ic_df[ic_df['sowing_date'].dt.year != ic_df['year']]
        if not mismatched_sowing_dates.empty:
            msg = (
                f"Logic Error: Found {len(mismatched_sowing_dates)} records in 'InitialConditions.csv' where the sowing year "
                "does not match the simulation year. This will cause zero-yield errors.")
            error_messages.append({'type': 'logic', 'text': msg})
        else:
            logging.info("  [OK] All sowing dates have the correct year.")
    except FileNotFoundError:
        msg = "File 'InitialConditions.csv' not found. Please run 'build_initial_conditions.py'."
        error_messages.append({'type': 'missing_file', 'text': msg})

    # ==========================================================================
    # === NEW: PHASE 4 - THE CRITICAL CHECK FOR SILENT FAILURES ===
    # ==========================================================================
    logging.info("\n--- [Phase 4] Sanity-Checking Generated Forecast Weather ---")
    forecast_dir = Path(PATHS['forecast_weather_dir'])
    if not forecast_dir.exists():
        msg = f"Directory '{forecast_dir}' not found. Please run 'build_forecast_weather.py' first."
        error_messages.append({'type': 'missing_file', 'text': msg})
    else:
        # Find one sample file to analyze
        sample_file = next(forecast_dir.glob('*.parquet'), None)
        if not sample_file:
            msg = f"No forecast weather (.parquet) files found in '{forecast_dir}'. The build script may have failed."
            error_messages.append({'type': 'missing_file', 'text': msg})
        else:
            logging.info(f"  Analyzing sample file: {sample_file.name}")
            df_weather = pd.read_parquet(sample_file)

            # Check for plausible ranges. These thresholds are key to catching errors.
            stats = df_weather[['tmin', 'tmax', 'precip', 'srad']].mean()

            # PRECIPITATION CHECK (cm vs mm error)
            precip_mean = stats.get('precip', 0)
            if precip_mean < 0.5:  # Mean daily rain < 0.5mm is a huge red flag
                msg = (f"CRITICAL: Mean daily precipitation in forecast is {precip_mean:.4f} mm. "
                       "This value is extremely low and will cause zero-yield simulations.")
                error_messages.append({'type': 'unit_error', 'text': msg})

            # TEMPERATURE CHECK (outliers)
            tmax_mean = stats.get('tmax', 0)
            if not (-10 < tmax_mean < 45):
                msg = (f"CRITICAL: Mean Tmax in forecast is {tmax_mean:.1f}°C. "
                       "This value is outside a plausible range and indicates a severe data error.")
                error_messages.append({'type': 'data_error', 'text': msg})

            if not error_messages or 'unit_error' not in [e['type'] for e in error_messages]:
                logging.info("  [OK] Forecast weather statistics appear plausible.")

    # --- Final Report and Conclusion ---
    print("\n" + "=" * 70)
    print("---               Diagnosis Conclusion                ---")
    print("=" * 70)

    if not error_messages:
        print("\n[SUCCESS] All checks passed! Your data assets appear consistent and correctly formatted.")
    else:
        print(f"\n[ACTION REQUIRED] Found {len(error_messages)} issue(s). Review the following:\n")
        for i, error in enumerate(error_messages):
            print(f"--- Problem #{i + 1} ({error['type']}) ---")
            print(f"  ISSUE: {error['text']}")
            if error['type'] == 'unit_error':
                print(
                    "  LIKELY CAUSE: The historical weather data used to train the WeatherGenerator has precipitation in 'cm', not 'mm'.")
                print(
                    "  SOLUTION: Apply the fix to 'build_forecast_weather.py' to multiply the historical precipitation by 10 before fitting.")
            elif error['type'] == 'logic':
                print("  SOLUTION: Fix the fallback logic in 'build_initial_conditions.py' to use the correct year.")
            elif error['type'] == 'mismatch':
                print(
                    "  SOLUTION: Remove the mismatched districts from 'sugarbeet_yield.csv' and re-run 'build_site_data.py'.")
            elif error['type'] in ['missing_file', 'fatal', 'data_error']:
                print(
                    "  SOLUTION: Please run the prerequisite build scripts or check the source data for severe errors.")
            print("-" * (len(error['type']) + 16))


if __name__ == "__main__":
    run_diagnostics()