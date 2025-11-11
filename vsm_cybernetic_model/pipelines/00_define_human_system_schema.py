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

    --- VSM System 2 (Coordination) ---
    - avg_sowing_date_doy: int
        Average day-of-year for sugar beet sowing.
    - avg_harvest_date_doy: int
        Average day-of-year for sugar beet harvesting.
    - crop_competition_ratio: float
        Ratio of sugar beet area to major competing crops.
    - sugar_beet_specialization: float
        Ratio of sugar beet area to total arable land.

    --- VSM System 3 (Control / Economic Battery) ---
    - avg_farm_size_ha: float
        Average farm size in hectares.
    - land_tenure_ratio: float
        Ratio of rented land to owned land.
    - avg_land_price_eur_ha: float
        Average price of agricultural land in EUR per hectare.
    - total_SO_NUTS3: float
        Total Standard Output for the district (monetary value).
    - cost_pressure_index: float
        Engineered proxy for regional economic pressure.
    - family_labor_ratio: float
        Ratio of family workers to total farm labor.

    --- VSM System 4 (Strategy / Market) ---
    - dist_to_processor_km: float
        Road network distance to the nearest sugar processing plant.
    - national_price_sugar_beet: float
        National average producer price for sugar beet.
    - national_price_wheat: float
        National average producer price for wheat (competing crop).
    - national_price_maize: float
        National average producer price for maize (competing crop).

    --- VSM System 5 (Policy) ---
    - percent_UAA_in_NVZ: float
        Percentage of utilized agricultural area within a Nitrate Vulnerable Zone.
    - CAP_Euros_per_Hectare_UAA: float
        Average CAP subsidy payment in EUR per hectare of UAA.
    """
    pass

if __name__ == "__main__":
    print("This script is for documentation purposes only.")
    print("Please see the docstring for the expected DataFrame schema.")
