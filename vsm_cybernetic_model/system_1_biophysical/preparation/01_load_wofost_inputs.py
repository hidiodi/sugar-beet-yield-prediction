# 01_load_wofost_inputs.py

def prepare_wofost_inputs():
    """
    Prepares the static biophysical inputs and defines the VSM 1 calibration
    data requirements.

    This script's primary role is to prepare static data. However, it also
    documents the critical dependency of the VSM 1 RPP simulation on the
    dynamic VSM 2 data.

    ---
    Part 1: Static Feature Preparation (User Implementation Required)
    ---
    This section should load and process static soil and environmental data.

    Expected Output:
        - A pandas DataFrame saved to `cfg.FOUNDATIONAL_FEATURES_STATIC_BIO`.
        - Schema:
            - district_no: int
            - Soil_Water_Battery: float
            - ... (other static features)

    ---
    Part 2: Dynamic Calibration Data (Documentation)
    ---
    The `system_1_biophysical/model/01_run_rpp_simulations.py` script requires
    foundational features from VSM 2 for calibration. The preparation for
    these features is handled in the `system_2_coordination` module.

    Required Foundational Feature for RPP Calibration:
        - sowing_date_doy_nuts3: int (from DWD raster grids)
    """
    print("--- Preparing VSM System 1 (Biophysical) Static Inputs ---")
    print("NOTE: This is a placeholder. User must implement static data loading.")
    print("This module has a dependency on 'sowing_date_doy_nuts3' from VSM 2.")
    print("--- Finished VSM System 1 Preparation ---")

if __name__ == "__main__":
    prepare_wofost_inputs()
