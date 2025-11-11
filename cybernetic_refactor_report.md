# The VSM-CPS Diagnostic Model: Final Documentation & Implementation Guide (v4.1: Unified Forecast Architecture)

---
## 1. Introduction: The Forecast Diagnostic Model

### 1.1. Purpose

This document provides the complete technical documentation for the **VSM-CPS (Viable System Model - Cyber-Physical System) Diagnostic Model**. Its purpose is to serve as a single source of truth for understanding, implementing, and maintaining the system.

The model is a **single, unified Forecast Diagnostic system**. It is designed to operate pre-season (e.g., in March) to predict the upcoming harvest's yield and, crucially, to diagnose the likely systemic drivers of that outcome. It achieves this by training on historical data, where it learns the relationships between past systemic conditions and yield outcomes.

### 1.2. High-Level Architecture

The system is a hierarchical pipeline that learns from historical data to make future predictions:

1.  **Data Integration:** Raw data is loaded and transformed into a standardized set of "foundational features." (See Section 4).
2.  **Expert Engine Training:** Unsupervised PCA models—"Expert Engines"—are trained on the full historical dataset to learn the latent structure of each VSM subsystem.
3.  **Final Diagnosis:** A final XGBoost "regulator" model is trained on the historical VSM indices to predict the gap between a realistic potential yield and the final observed yield.

When run in a live forecast, the pre-trained expert engines and regulator model are used to generate a diagnosis based on the latest available pre-season data and weather forecasts.

---
## 2. The VSM 1 "Sensor" in a Forecast Context

The VSM 1 biophysical sensor is architected specifically for a forecast environment.

*   **Weather Input:** The WOFOST simulation is **always** driven by an **ensemble of forecasted weather scenarios** (e.g., the 51 members of a SEAS5 forecast). This is true for both historical training runs (where historical forecasts are used) and live forecast runs.
*   **RPP Output:** The result is always a *distribution* of Realistic Physical Potential (RPP) outcomes. The foundational features extracted are therefore distributional, such as:
    *   `RPP_ensemble_mean_yield`
    *   `RPP_ensemble_std_dev_yield` (a key measure of biophysical uncertainty/risk)
    *   `prob_rpp_failure` (the percentage of ensemble members that predict a crop failure).

---
## 3. Implemented Pipeline & Execution Guide
*(Execution commands are unchanged)*

---
## 4. Implementation Checklist: Data Timing

This checklist details the foundational features required and clarifies their availability for a pre-season forecast. The model is trained on historical data where all features are available, but it is *designed to predict* using only the subset available pre-season.

**✅ Checklist:**

| VSM System | Feature to Create | Data Source | Data Timing | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1 & 2** | `sowing_date_doy_nuts3` | DWD Raster Grids | **Pre-Season** | **Highest Priority.** Assumed to be known or estimated by March. |
| **2** | `crop_area_variance_nuts3` | Destatis `41241` | Post-Season | Available for historical training data only. Cannot be known pre-season. |
| **2** | `irrigation_pct_nuts2` | Eurostat `aei_ef_ir` | Static | Structural feature, changes slowly. Available for any forecast. |
| **3** | `avg_farm_size_n3` | Destatis `41251` | Static | Structural feature (decennial census). Available for any forecast. |
| **3** | `so_per_ha_n3` | Eurostat `ef_kvaareg` | Post-Season | Based on annual economic accounts. Available for historical training data only. |
| **3** | `input_cost_index_n1` | Eurostat `apri_pi_in` | **Pre-Season (Lagged)** | For a March forecast, use the index values from the previous year. |
| **3** | `producer_price_index_n1` | Eurostat `apri_pi_out`| **Pre-Season (Lagged)** | For a March forecast, use the previous year's prices. |
| **4** | `distance_to_processor_km_nuts3`| Manual Geocoding | Static | Static geographical feature. Available for any forecast. |
| **5** | `total_cap_subsidy_nuts3` | `agrarzahlungen.de` | Post-Season | Based on annual payment disclosures. Available for historical training data only. |
| **5** | `pct_area_rote_gebiete_nuts3`| Länder GIS Portals | Static | Regulatory feature, changes slowly. Available for any forecast. |

### Architectural Implication: One Unified Regulator Model

The system is designed to train **one unified regulator model**. This model is trained on the complete historical dataset. When constructing the training data, features are lagged appropriately to mimic the information that would have been available at forecast time. For example, the `so_per_ha_n3` feature for a 2020 forecast would be the value from 2019. This ensures the model learns from the same data structure it will see during a live forecast, eliminating the need for two separate models.

---
## 5. Validation and Interpretation: A Practical Guide
*(This section is unchanged, as the validation scripts and their interpretation are still correct for the unified forecast model)*
