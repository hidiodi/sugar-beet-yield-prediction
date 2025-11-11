# 00_define_human_system_schema.py

def define_schema():
    """
    This script serves as the master schema definition for the
    `FOUNDATIONAL_FEATURES_HUMAN.csv` file.

    All of the individual preparation scripts in the `system_2_coordination`
    through `system_5_policy` directories are responsible for producing
    the columns defined here. This file provides a single, consolidated
    view of the data the expert engine models expect.

    Expected Output File:
    ---------------------
    - Path: `vsm_cybernetic_model.configs.main_config.FOUNDATIONAL_FEATURES_HUMAN`
    - Format: CSV

    DataFrame Schema:
    -----------------
    - year: int
        The observation year.
    - district_no: int
        The NUTS 3 district code.

    --- VSM System 2 (Coordination / Management) ---
    - sowing_date_doy_nuts3: int
        Mean Day of Year for sowing, from DWD raster data.
    - harvest_date_doy_nuts3: int
        Mean Day of Year for harvesting, from DWD raster data.
    - irrigation_pct_nuts2: float
        Percentage of irrigable area, applied from NUTS 2 level.
    - crop_area_variance_nuts3: float
        Year-over-year variance in sugar beet hectares as a proxy for rotation.

    --- VSM System 3 (Control / Economic Battery) ---
    - avg_farm_size_n3: float
        Average farm size in hectares (from Destatis 41251).
    - so_per_ha_n3: float
        Disaggregated Standard Output per hectare (income proxy).
    - input_cost_index_n1: float
        National-level index for input costs (e.g., fertilizer).
    - producer_price_index_n1: float
        National-level index for producer prices (e.g., root crops).
    - land_tenure_ratio_n2: float
        Ratio of rented to owned land, applied from NUTS 2 level.

    --- VSM System 4 (Strategy / Market) ---
    - distance_to_processor_km_nuts3: float
        Road network distance to the nearest sugar processing plant.

    --- VSM System 5 (Policy) ---
    - total_cap_subsidy_nuts3: float
        Total aggregated CAP subsidy payments for the district.
    - pct_area_rote_gebiete_nuts3: float
        Percentage of UAA within a "Rote Gebiete" (Nitrate Vulnerable Zone).
    """
    pass

if __name__ == "__main__":
    print("This script is for documentation purposes only.")
    print("Please see the docstring for the expected DataFrame schema.")
