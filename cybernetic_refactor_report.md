# The VSM-CPS Diagnostic Model: Final Documentation & Implementation Guide

---
## 1. Introduction: The Diagnostic Model
*(This section is unchanged)*

---
## 2. Implemented Pipeline & Execution Guide

The `vsm_cybernetic_model` module contains the fully implemented core logic for the diagnostic model.

### 2.1. How the Code Works

*   **Configuration (`/configs`):** Centralizes all file paths and model parameters.
*   **Expert Engine Training (`/system_*/model/`):** Each `train_*_engine.py` script trains a PCA model on a specific subset of foundational features and saves the fitted `scaler` and `pca` artifacts to `/models/stage_1_experts/`.
*   **Pipeline Orchestration (`/pipelines`):**
    *   `01_run_feature_engineering_pipeline.py` operates in two modes:
        *   **`train` mode:** Creates the expert engine artifacts.
        *   **`transform` mode:** Loads the artifacts to generate the final VSM indices.
    *   `02_run_model_training_pipeline.py`: Trains the final XGBoost regulator model.

### 2.2. How to Run the System

**Step 1: Implement the Data Preparation Scripts (See Section 4)**
*   Your primary task is to complete the scripts in the `preparation` directories.

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
## 3. Validation and Interpretation: A Practical Guide

The system includes a comprehensive, multi-layered validation suite, executed via a single command. This section details what each script does and how to interpret its output.

### 3.1. How to Run the Verification Suite

After successfully training the full system (Steps 2 and 3 above), run the following command from the repository root:
```bash
python -m vsm_cybernetic_model.pipelines.03_run_verification_pipeline
```
This will generate a series of plots and analyses in the `/reports/verification/` directory.

### 3.2. Interpreting the Outputs: A Layer-by-Layer Guide

**Layer 1: Foundational Feature Validation**
*   **What it does:** The `01_validate_*_features.py` scripts generate plots (histograms, scatter plots) of the key foundational features you created.
*   **What to look for:** Check these plots for sanity. Do the distributions make sense? Are there extreme outliers that might indicate data processing errors? For example, the `vsm2_sowing_date_distribution.png` plot should show a plausible range of sowing dates for Germany.

**Layer 2: RPP Baseline Validation**
*   **What it does:** The `system_1_biophysical/verification/01_validate_rpp_baseline.py` script generates the `vsm1_rpp_plausibility_check.png` plot.
*   **What to look for:** This is a critical check. The distribution for `RPP_mean_yield` should be higher on average than the distribution for the real yield, confirming that our baseline represents a reasonable *potential* yield.

**Layer 3: Expert Engine Validation (Interpretability)**
*   **What it does:** The `02_analyze_*_engine.py` scripts are the most important validation step. They generate heatmaps of the PCA component loadings (e.g., `vsm3_component_loadings.png`).
*   **What to look for:** These heatmaps tell you *what the expert engines have learned*. For the **VSM 3 "Economic Battery,"** you must confirm that the first principal component (`PC1`) has high positive or negative loadings for features that represent economic strength (e.g., `avg_land_price`, `so_per_ha_n3`) and low loadings for unrelated features. If the loadings do not make intuitive sense, the engine is an uninterpretable "black box," and its input features must be re-evaluated.

**Layer 4: Post-Hoc Model Validation (Final Check)**
*   **What it does:** The top-level `verification/01_run_post_hoc_shap_analysis.py` script generates the `final_model_shap_summary.png`.
*   **What to look for:** This plot shows the global feature importance for the final XGBoost regulator. You must confirm that the VSM indices (e.g., `VSM3_PC1`, `VSM4_PC1`) rank among the most important features. If they do not, it indicates that our high-level VSM architecture is not effectively capturing the variance in the yield gap.

---
## 4. Your Implementation Checklist: The Path to a Live Model

This section provides the explicit, actionable "to-do list" required to finish the model. Your task is to implement the placeholder scripts in the `**/preparation/` directories.

**✅ Checklist:**

1.  **Implement VSM 1 & 2 Preparation (Calibration Data):**
    *   **File:** `system_2_coordination/preparation/01_process_dwd_phenology_data.py`
    *   **Task:** Implement the zonal statistics logic to process the DWD raster grids and generate the `sowing_date_doy_nuts3` feature. This is the **highest priority** as it is required to calibrate the VSM 1 sensor.

2.  **Implement VSM 2 Preparation (Management Features):**
    *   **File:** `system_2_coordination/preparation/02_process_crop_rotation_data.py`
    *   **Tasks:**
        *   Source and disaggregate the NUTS 2 Eurostat irrigation data (`irrigation_pct_nuts2`).
        *   Process the time-series from Destatis Table `41241` to create the `crop_area_variance_nuts3` feature.

3.  **Implement VSM 3 Preparation (The Economic Battery):**
    *   **File:** `system_3_control/preparation/01_prepare_economic_battery_features.py`
    *   **Task:** This script already contains the working base for national price indices. Your task is to implement the **disaggregation logic** described in the docstring, sourcing the required NUTS 3 structural data (Destatis) and NUTS 2 economic data (Eurostat) to create features like `so_per_ha_n3`.

4.  **Implement VSM 4 Preparation (Market Strategy):**
    *   **File:** `system_4_strategy/preparation/01_prepare_strategy_features.py`
    *   **Task:** Implement the geospatial logic to geocode the 13 processor addresses and calculate the road network `distance_to_processor_km_nuts3` for each NUTS 3 district.

5.  **Implement VSM 5 Preparation (Policy):**
    *   **File:** `system_5_policy/preparation/01_prepare_policy_features.py`
    *   **Tasks:**
        *   Implement the aggregation logic to process the `agrarzahlungen.de` CAP subsidies CSV to generate `total_cap_subsidy_nuts3`.
        *   Implement the geospatial workflow to process the federated "Rote Gebiete" shapefiles and calculate `pct_area_rote_gebiete_nuts3`.

Once these five tasks are complete, the system will be fully data-connected and the `train` and `transform` pipelines can be run to produce the final, diagnostic model.
