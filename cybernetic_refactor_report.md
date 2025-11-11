# The VSM-CPS Diagnostic Model: Final Technical Documentation & Implementation Guide

---
## 1. Introduction: The Diagnostic Model

### 1.1. Purpose

This document provides the complete technical documentation for the **VSM-CPS (Viable System Model - Cyber-Physical System) Diagnostic Model**. Its purpose is to serve as a single source of truth for understanding, implementing, and maintaining the system.

The model's primary function is not just to *predict* German sugar beet yield, but to **diagnose the systemic drivers** behind yield deviations. It is a "why" model, designed to differentiate between failures in the biophysical environment (VSM System 1) and failures in the human systems of management, economics, and policy (VSM Systems 2-5).

### 1.2. High-Level Architecture

The system is a hierarchical, multi-stage machine learning pipeline:

1.  **Data Integration:** A user-driven process where raw, multi-format data is loaded and transformed into a standardized set of "foundational features."
2.  **Expert Engine Training:** A series of unsupervised PCA models—our "Expert Engines"—are trained to distill these foundational features into high-level indices representing the state of each VSM subsystem.
3.  **Final Diagnosis:** A final XGBoost "regulator" model is trained on these VSM indices to predict the gap between a realistic potential yield and the final observed yield, allowing for diagnostic attribution.

---
## 2. Data Integration Workflow (User "To-Do" List)

This section is the primary guide for the data engineer. The `vsm_cybernetic_model` is fully implemented, but it relies on a set of "foundational feature" tables that must be created from raw data. Your task is to implement the scripts in the `**/preparation/` directories to produce these tables according to the schemas defined below.

### 2.1. VSM System 1: Calibrating the "Sensor"

The biophysical "sensor" requires data to create a **Realistic Physical Potential (RPP)** baseline.

*   **Script to Implement:** `system_1_biophysical/preparation/01_load_wofoost_inputs.py`
*   **Raw Data Source:** Deutscher Wetterdienst (DWD) "Annual grids of several phenological plant stages in Germany" (1km raster grids).
*   **Task:** Perform a zonal statistics operation on the raster grids to calculate the mean day-of-year for sowing for each NUTS 3 Landkreis.
*   **Required Output Feature:** `sowing_date_doy_nuts3`. This is used by the implemented `01_run_rpp_simulations.py` script.

### 2.2. VSM System 2: Management & Coordination

*   **Scripts to Implement:** `system_2_coordination/preparation/*.py`
*   **Raw Data Sources:**
    *   **Irrigation:** Eurostat tables (`aei_ef_ir`, `tai03`, `ef_fsi_irri`) (NUTS 2).
    *   **Crop Rotation Proxy:** Destatis GENESIS Table `41241` (NUTS 3).
*   **Tasks:**
    1.  Disaggregate the NUTS 2 irrigation data to the NUTS 3 level.
    2.  Analyze the time-series of crop area in Table 41241 to calculate year-over-year variance.
*   **Required Output Features:** `irrigation_pct_nuts2`, `crop_area_variance_nuts3`.

### 2.3. VSM System 3: The "Economic Battery"

This is the most complex data integration task, requiring multi-scale disaggregation.

*   **Script to Implement:** `system_3_control/preparation/01_prepare_economic_battery_features.py`
*   **Raw Data Sources:**
    *   **NUTS 3 Structure:** Destatis `41251` (Farm Size), `41241` (UAA by Crop).
    *   **NUTS 2 Economics:** Eurostat `ef_kvaareg` (Standard Output), `aact_eaa01` (Consumption).
    *   **NUTS 1/0 Prices:** Eurostat `apri_pi_out` & `apri_pi_in` (Price Indices).
*   **Task:** Implement the disaggregation logic. For example: `N3_SO_per_ha = N2_SO (Eurostat) / N3_UAA_by_Crop (Destatis)`.
*   **Required Output Features:** `avg_farm_size_n3`, `so_per_ha_n3`, `input_cost_index_n1`, etc.

### 2.4. VSM Systems 4 & 5: Strategy & Policy

*   **Scripts to Implement:** `system_4_strategy/preparation/*.py`, `system_5_policy/preparation/*.py`
*   **Raw Data Sources (Federated):**
    *   **Subsidies:** `Gesamtliste der Begünstigten` CSV from `agrarzahlungen.de`.
    *   **Nitrate Zones:** "Rote Gebiete" GIS shapefiles from German state portals.
    *   **Market Access:** Manual list of Südzucker/Nordzucker factory addresses.
*   **Tasks:**
    1.  Aggregate the CAP subsidies CSV to the NUTS 3 level.
    2.  Perform a spatial join of the "Rote Gebiete" shapefiles with NUTS 3 polygons.
    3.  Geocode the factory addresses and calculate the road network distance for each NUTS 3 centroid.
*   **Required Output Features:** `total_cap_subsidy_nuts3`, `pct_area_rote_gebiete_nuts3`, `distance_to_processor_km_nuts3`.

---
## 3. Implemented Pipeline & Execution Guide

The `vsm_cybernetic_model` module contains the fully implemented core logic.

### 3.1. How the Code Works

*   **Configuration (`/configs`):** All file paths and model parameters are defined here. This is the central location for configuration.
*   **Expert Engine Training (`/system_*/model/`):** Each `train_*_engine.py` script loads the foundational features you prepared, scales them, trains a PCA model, and saves the fitted `scaler` and `pca` model artifacts to `/models/stage_1_experts/`.
*   **Pipeline Orchestration (`/pipelines`):**
    *   `01_run_feature_engineering_pipeline.py`: Has two modes:
        *   **`train` mode:** Runs all the `train_*_engine.py` scripts to create the model artifacts. **This must be run once.**
        *   **`transform` mode:** Loads the saved artifacts and uses them to convert your foundational features into the final VSM indices for the main model.
    *   `02_run_model_training_pipeline.py`: Loads the final VSM indices and trains the XGBoost regulator model.

### 3.2. How to Run the System

**Step 1: Implement `preparation` Scripts**
-   Complete your data integration tasks as defined in Section 2.

**Step 2: Train the Expert Engines**
-   From the repository root, run:
    ```bash
    python -m vsm_cybernetic_model.pipelines.01_run_feature_engineering_pipeline train
    ```

**Step 3: Train the Final Regulator Model**
-   First, generate the final feature matrix using the trained engines:
    ```bash
    python -m vsm_cybernetic_model.pipelines.01_run_feature_engineering_pipeline transform
    ```
-   Then, train the final XGBoost model:
    ```bash
    python -m vsm_cybernetic_model.pipelines.02_run_model_training_pipeline
    ```

---
## 4. Validation and Interpretation Strategy

The system is designed for robust validation.

### 4.1. How the Code Works

*   The `**/verification/` directories contain a full suite of implemented scripts.
*   The `03_run_verification_pipeline.py` script automatically discovers and runs all of these scripts, generating a series of plots and analyses in the `/reports/verification/` directory.

### 4.2. How to Use

**Step 1: Run the Full Verification Suite**
-   After running the full training process, execute:
    ```bash
    python -m vsm_cybernetic_model.pipelines.03_run_verification_pipeline
    ```

**Step 2: Interpret the Outputs**

1.  **Foundational Feature Plots:** Check the plots for the foundational features (e.g., `vsm2_sowing_date_distribution.png`). Do they make sense? Are there outliers?
2.  **Component Loading Heatmaps:** This is the most important check. Look at the heatmaps for each VSM engine (e.g., `vsm3_component_loadings.png`). Do the PCA components align with your domain knowledge? For the "Economic Battery," are features like land price and farm size heavily weighted on the main component? If not, the engine is not interpretable and must be re-evaluated.
3.  **Final SHAP Analysis:** Examine the `final_model_shap_summary.png`. This plot shows which of your high-level VSM indices were the most important predictors for the final model. This is the ultimate validation that the VSM architecture is successfully diagnosing the drivers of yield.
