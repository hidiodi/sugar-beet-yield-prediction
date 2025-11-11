# 01_process_dwd_phenology_data.py

def prepare_phenology_data():
    """
    USER IMPLEMENTATION REQUIRED.

    This script processes phenological data to generate sowing and harvest
    dates, which are critical for calibrating the VSM 1 RPP "Sensor".

    Data Source:
    ------------
    - Deutscher Wetterdienst (DWD) Climate Data Center (CDC).
    - Product: "Annual grids of several phenological plant stages in Germany".
    - These are 1km raster grids.

    Processing Steps:
    -----------------
    1.  Download the relevant raster grids for sowing and harvesting dates.
    2.  For each NUTS 3 (Landkreis) polygon, perform a zonal statistics
        operation to calculate the mean "Day of Year" for sowing and harvesting.
    3.  This will produce a precise, interpolated baseline for each district.

    Expected Output Columns in `FOUNDATIONAL_FEATURES_HUMAN.csv`:
    ----------------------------------------------------------
    - sowing_date_doy_nuts3: int
    - harvest_date_doy_nuts3: int
    """
    print("--- Preparing VSM System 2 (Phenology) Data ---")
    print("NOTE: This is a placeholder. User must implement DWD raster processing.")
    print("--- Finished Phenology Data Preparation ---")

if __name__ == "__main__":
    prepare_phenology_data()
