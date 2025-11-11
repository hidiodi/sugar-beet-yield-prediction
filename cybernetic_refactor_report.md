# The VSM-CPS Diagnostic Model: Final Documentation (v4.1 Aligned)

---
## 1. Introduction: The Forecast Diagnostic Model

### 1.1. Purpose

This document provides the complete technical documentation for the **VSM-CPS (Viable System Model - Cyber-Physical System) Diagnostic Model**.

The model is a **single, unified Forecast Diagnostic system**. It is designed to operate pre-season (e.g., in March) to predict the upcoming harvest's yield and, crucially, to diagnose the likely systemic drivers of that outcome. It achieves this by training on historical data, where it learns the relationships between past systemic conditions and yield outcomes.

### 1.2. High-Level Architecture

The system is a hierarchical pipeline that learns from historical data to make future predictions:

1.  **Data Integration:** User-implemented scripts transform raw data into "foundational features."
2.  **Expert Engine Training:** Unsupervised PCA models—"Expert Engines"—are trained on the full historical dataset to learn the latent structure of each VSM subsystem.
3.  **Final Diagnosis:** A final XGBoost "regulator" model is trained on the historical VSM indices to predict the gap between a realistic potential yield and the final observed yield.

---
## 2. Implemented Pipeline & Execution Guide

### 2.1. How the Code Works

*   **The VSM 1 "Sensor":** The `system_1_biophysical/model/01_run_rpp_simulations.py` script is designed to be driven by an **ensemble weather forecast**. Its placeholder logic correctly generates distributional outputs, including `RPP_ensemble_mean_yield` and `RPP_ensemble_std_dev_yield`, which are used as inputs for the VSM 1 expert engine.
*   **Feature Lagging for Forecasting:** The `pipelines/01_run_feature_engineering_pipeline.py` contains a critical function, `_apply_feature_lags`. This function is called in `transform` mode and automatically applies a one-year lag to all "Post-Season" features (e.g., `so_per_ha_n3`, `total_cap_subsidy_nuts3`). This ensures the model is always trained on data that mimics the information available at the time of a pre-season forecast.
*   **Expert Engine Training (`/system_*/model/`):** Each `train_*_engine.py` script trains a PCA model and saves the fitted `scaler` and `pca` artifacts.
*   **Pipeline Orchestration (`/pipelines`):**
    *   The main pipeline has two modes: **`train`** (to create the expert engines) and **`transform`** (to apply the lagging and generate the final VSM indices).

### 2.2. How to Run the System

**(This section is unchanged)**

---
## 3. Validation and Interpretation: A Practical Guide
**(This section is unchanged)**

---
## 4. Implementation Checklist: Data Timing

This checklist details the foundational features required and clarifies their availability for a pre-season forecast.

**✅ Checklist:**

| VSM System | Feature to Create | Data Source | Data Timing | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1 & 2** | `sowing_date_doy_nuts3` | DWD Raster Grids | **Pre-Season** | **Highest Priority.** Needed for both modes. Assumed to be known or estimated by March. |
| **2** | `crop_area_variance_nuts3` | Destatis `41241` | Post-Season | **Will be lagged by the pipeline.** |
| **2** | `irrigation_pct_nuts2` | Eurostat `aei_ef_ir` | Static | Structural feature, changes slowly. Available for any forecast. |
| **3** | `avg_farm_size_n3` | Destatis `41251` | Static | Structural feature (decennial census). Available for any forecast. |
| **3** | `so_per_ha_n3` | Eurostat `ef_kvaareg` | Post-Season | **Will be lagged by the pipeline.** |
| **3** | `input_cost_index_n1` | Eurostat `apri_pi_in` | **Pre-Season (Lagged)** | For a March forecast, use the index values from the previous year. |
| **3** | `producer_price_index_n1` | Eurostat `apri_pi_out`| **Pre-Season (Lagged)** | For a March forecast, use the previous year's prices. |
| **4** | `distance_to_processor_km_nuts3`| Manual Geocoding | Static | Static geographical feature. Available for any forecast. |
| **5** | `total_cap_subsidy_nuts3` | `agrarzahlungen.de` | Post-Season | **Will be lagged by the pipeline.** |
| **5** | `pct_area_rote_gebiete_nuts3`| Länder GIS Portals | Static | Regulatory feature, changes slowly. Available for any forecast. |

This confirms the system uses a single, unified regulator model, trained on a dataset that correctly simulates the data available at the time of a pre-season forecast.
