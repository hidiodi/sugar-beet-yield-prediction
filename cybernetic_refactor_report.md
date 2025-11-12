# The VSM-CPS Diagnostic Model: Documentation for the Baseline Model (v6.0)

---
## 1. Introduction: The Baseline Forecast Diagnostic Model

### 1.1. Purpose

This document provides the complete technical documentation for the **Baseline VSM-CPS Diagnostic Model**. This version of the model is designed to be **runnable end-to-end using only the existing, curated feature set**.

Its purpose is to establish a strong baseline performance for our VSM architecture. It provides a pre-season forecast and a diagnostic explanation for that forecast, based on a mapping of our rich existing features to the VSM framework.

This document details the feature mapping, the implemented pipeline, and provides a clear roadmap for future improvements by integrating new data sources.

---
## 2. VSM Mapping of Existing Features

The core of this baseline model is a new mapping of our extensive existing feature set to the five VSM subsystems. This allows us to build and test the VSM architecture without sourcing any new data.

| VSM System | Feature Category | Example Existing Features |
| :--- | :--- | :--- |
| **System 1 (Biophysical)** | Environment & Forecast | `avg_sand_0_30cm`, `antecedent_gdd_sum_anomaly`, `winter_cropland_snow_cover_days`, `summer_temp_anomaly_forecast` |
| **System 2 (Coordination)** | Management Proxy | `zuckerrben` (area of sugar beet cultivation) |
| **System 3 (Control)** | Economic Battery | `dngemittel` (fertilizer costs), `profit_margin_proxy_lag1`, `cost_of_inputs_momentum` |
| **System 4 (Strategy)** | Market Signals | `national_avg_yield_lag1` (national supply proxy) |
| **System 5 (Policy)** | **DATA GAP** | *(No features in the current dataset directly map to this system)* |

---
## 3. Implemented Pipeline & Execution Guide

The `vsm_cybernetic_model` module is a fully implemented, runnable system that works with the existing `master_dataset.csv`.

### 3.1. How the Code Works

*   **Unified Data Preparation:** The placeholder `preparation` scripts have been replaced by a single, functional script: `pipelines/00_prepare_foundational_features.py`. This script loads your `data/04_master/master_dataset.csv`, handles any necessary preprocessing, and saves the clean data in the format the rest of the pipeline expects.
*   **Data-Driven Configurations:** The `configs/*.py` files have been refactored. The `INPUT_FEATURES` lists now contain the actual feature names from your master dataset, mapped to the correct VSM system.
*   **End-to-End Execution:** The system is fully runnable. The pipelines will load your data, train the five "Expert Engines" (PCA models), generate the VSM indices, and train the final XGBoost "Regulator" model.

### 3.2. How to Run the System

**Step 1: Verify Your Data**
*   Ensure your complete, curated feature set is located at `data/04_master/master_dataset.csv`.

**Step 2: Train the Expert Engines**
*   From the repository root, run:
    ```bash
    python -m vsm_cybernetic_model.pipelines.01_run_feature_engineering_pipeline train
    ```

**Step 3: Train the Final Regulator Model**
*   Generate the final feature matrix:
    ```bash
    python -m vsm_cybernetic_model.pipelines.01_run_feature_engineering_pipeline transform
    ```
*   Train the final XGBoost model:
    ```bash
    python -m vsm_cybernetic_model.pipelines.02_run_model_training_pipeline
    ```

---
## 4. Future Work: A Roadmap for New Data Integration

This baseline model is powerful, but its diagnostic capabilities can be significantly enhanced by integrating new, targeted data sources. The research and planning for this next phase is complete and is summarized here. This roadmap replaces the previous, more complex "to-do list."

**Priority 1: Calibrate the VSM 1 "Sensor"**
*   **Task:** Implement the `system_2_coordination/preparation/01_process_dwd_phenology_data.py` placeholder.
*   **Data Source:** DWD "Annual grids of phenological plant stages."
*   **Goal:** Replace the simple weather forecast features with a true, calibrated "Realistic Physical Potential" (RPP) baseline from WOFOST. This will provide a much cleaner signal for the final regulator.

**Priority 2: Enhance the VSM 3 "Economic Battery"**
*   **Task:** Implement the disaggregation logic in `system_3_control/preparation/01_prepare_economic_battery_features.py`.
*   **Data Sources:** Destatis structural data (e.g., `41251`) and Eurostat NUTS 2 economic data (e.g., `ef_kvaareg`).
*   **Goal:** Move beyond national price indices to a more granular, regional measure of economic health, which will dramatically improve the model's ability to diagnose economic failures.

**Priority 3: Fill the VSM 5 "Policy" Gap**
*   **Task:** Implement the `system_5_policy/preparation/01_prepare_policy_features.py` placeholder.
*   **Data Sources:** `agrarzahlungen.de` for CAP subsidies and Länder GIS portals for "Rote Gebiete."
*   **Goal:** Introduce the policy and regulatory dimension into the model, allowing it to diagnose failures driven by these external constraints for the first time.
