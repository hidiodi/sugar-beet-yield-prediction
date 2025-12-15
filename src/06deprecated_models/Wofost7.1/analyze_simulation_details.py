# File: scripts/analyze_soil_parameters.py
# Description: A targeted diagnostic script to analyze the soil physics parameters
# in the final StaticSiteData.csv. It checks for physical plausibility and common
# unit errors that lead to unrealistic model behavior (e.g., no water stress).

import pandas as pd
from pathlib import Path
import logging
import sys

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/03_processed")
SITE_DATA_PATH = PROCESSED_DATA_DIR / "StaticSiteData.csv"

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def analyze_soil_physics(file_path: Path):
    """
    Analyzes the soil parameters within the final site data file.
    """
    logging.info("=" * 80)
    logging.info(f"--- Analyzing Soil Physics in: {file_path.name} ---")
    logging.info("=" * 80)

    if not file_path.exists():
        logging.error(f"[FATAL] The site data file was not found at: {file_path}")
        logging.error("Please ensure 'build_site_data.py' has been run successfully.")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
        # Analyze just one year for a clean summary
        df_sample = df[df['year'] == df['year'].min()].copy()
    except Exception as e:
        logging.error(f"[FATAL] Could not read or parse the CSV file. Error: {e}")
        sys.exit(1)

    soil_cols = ['SMW', 'SMFCF', 'SM0', 'RDMSOL']
    if not all(col in df_sample.columns for col in soil_cols):
        logging.error(f"FATAL: One or more required soil columns are missing from the file. Expected: {soil_cols}")
        sys.exit(1)

    # --- Phase 1: Check Physical Laws ---
    # The fundamental law of soil physics is that the wilting point moisture must be
    # less than the field capacity, which must be less than the saturation point.
    # All values should be unitless fractions (m3/m3) between 0 and 1.
    print("\n--- [1] Checking Physical Soil Laws (SMW < SMFCF < SM0) ---")

    violations = df_sample[~(df_sample['SMW'] < df_sample['SMFCF']) | ~(df_sample['SMFCF'] < df_sample['SM0'])]

    if not violations.empty:
        logging.error(
            f"\n[CRITICAL FAILURE] Found {len(violations)} records where the physical laws of soil moisture are violated.")
        print("This is a definitive bug. The model cannot produce realistic results with this data.")
        print("Example of a broken record:")
        print(violations.head(1)[soil_cols].to_string())

        recommendation = (
            "The input file `static_features_districts.csv` has physically impossible soil moisture values. "
            "This is often caused by a unit error in the script that generated it (e.g., build_static_features.py). "
            "The values must be corrected at the source.")
    else:
        logging.info("  [OK] All records adhere to the physical law SMW < SMFCF < SM0.")
        recommendation = None  # No immediate recommendation if this passes

    # --- Phase 2: Check for Unit Errors ---
    # Soil moisture fractions should be between 0 and 1. If they are > 1, they are not fractions.
    print("\n--- [2] Checking for Common Unit Errors (Values should be < 1) ---")

    stats = df_sample[soil_cols].describe().T
    print("\nStatistical Summary of Soil Parameters:")
    print(stats.to_string())

    unit_error_found = False
    if stats.loc['SMW', 'mean'] > 1.0 or stats.loc['SMFCF', 'mean'] > 1.0 or stats.loc['SM0', 'mean'] > 1.0:
        logging.error("\n[CRITICAL FAILURE] The mean values for SMW, SMFCF, or SM0 are greater than 1.0.")
        print("This indicates they are NOT unitless fractions (m3/m3) as required by the model.")
        print("They are likely percentages or permilles and MUST be converted.")
        unit_error_found = True
        recommendation = ("The soil moisture values in `static_features_districts.csv` are not fractions. "
                          "You MUST divide them by 100 (if they are percentages) or 1000 in the script that generates them.")
    else:
        logging.info("\n  [OK] Soil moisture values appear to be in the correct fractional range (0-1).")

    # --- Phase 3: Calculate and Check Plant Available Water ---
    # This tells us how much water the plant can actually use.
    print("\n--- [3] Analyzing Plant Available Water Capacity (PAWC) ---")
    df_sample['PAWC'] = (df_sample['SMFCF'] - df_sample['SMW'])
    df_sample['Total_PAW_cm'] = df_sample['PAWC'] * df_sample['RDMSOL']

    paw_stats = df_sample[['PAWC', 'Total_PAW_cm']].describe().T
    print("\nStatistical Summary of Plant Available Water:")
    print(paw_stats.to_string())

    if paw_stats.loc['Total_PAW_cm', 'mean'] > 30:
        logging.warning(
            f"\n[WARNING] The mean Total Plant Available Water is {paw_stats.loc['Total_PAW_cm', 'mean']:.1f} cm.")
        print("This is a very high value, suggesting a 'bottomless well' scenario where water stress is unlikely.")
        print("This could be caused by either a unit error in SMW/SMFCF or an overly large RDMSOL value.")
        if not unit_error_found:  # Only add this recommendation if we haven't found the unit error yet
            recommendation = "The Plant Available Water is extremely high. Review the units and values of SMW, SMFCF, and RDMSOL in `static_features_districts.csv`."

    # --- Final Conclusion ---
    print("\n" + "=" * 80)
    print("---                      SOIL ANALYSIS CONCLUSION                     ---")
    print("=" * 80)

    if recommendation:
        print(f"[DIAGNOSIS] A critical problem was found in the soil physics data.")
        print(f"[ACTION]    {recommendation}")
    else:
        print("[SUCCESS] The soil physics data appears physically plausible and correctly formatted.")
        print(
            "If the model results are still poor, the issue likely lies elsewhere (e.g., weather data anomalies or crop parameters).")


if __name__ == "__main__":
    analyze_soil_physics(SITE_DATA_PATH)