# Technical Documentation for the VSM-CPS Diagnostic Model (v3.0)

## 1. Executive Summary

This document provides the official technical and architectural documentation for the implemented VSM-CPS (Viable System Model - Cyber-Physical System) diagnostic model (v3.0). The system is a modular, multi-stage machine learning pipeline designed to diagnose the systemic drivers of German sugar beet yield.

The core architecture is built around a series of unsupervised "Expert Engines" (PCA models) that distill low-level data into high-level indices representing each of the five VSM subsystems. A key innovation is the VSM System 1 "Sensor," which uses a WOFOST crop model calibrated with real-world management data to generate a "Realistic Physical Potential" (RPP) baseline.

This document details the model's rationale, the final software architecture, the data integration plan, and the embedded strategies for validation and risk management.

---

## 2. Model Rationale
*(... section content unchanged ...)*

---

## 3. Strategic Data Foundation & Integration Plan

### 3.1. Strategic Data Foundation (Recap)

The project's feasibility rests on a foundation of confirmed, high-resolution public data. The analysis has identified the specific datasets required to model the "human system" gaps (VSM 2-5) at the NUTS 3 (Landkreise) resolution.

| VSM System | Core Data Asset | Source | Resolution |
| :--- | :--- | :--- | :--- |
| **System 1 & 2** | Phenological Grids (Sowing Date) | DWD CDC | 1km Raster |
| **System 2** | Crop Area, Irrigation | Destatis (41241), Eurostat | NUTS 3, NUTS 2 |
| **System 3** | Farm Structure, Crop Area, Econ. Accounts | Destatis (41251, 41241), Eurostat | NUTS 3, NUTS 2 |
| **System 4** | Processor Locations, Price Indices | Corporate Websites, Eurostat/Destatis | Point, NUTS 0/1 |
| **System 5** | CAP Subsidies, Nitrate Zones ("Rote Gebiete") | agrarzahlungen.de, German Länder Portals| CSV, GIS |

### 3.2. Data Integration Plan

This section summarizes the specific data sources and processing logic that must be implemented in the `preparation/` scripts of the `vsm_cybernetic_model` module.

**VSM System 1 (Biophysical "Sensor") Calibration:**
*   **Raw Data:** DWD Climate Data Center "Annual grids of several phenological plant stages in Germany".
*   **`preparation` Task:** Perform a zonal statistics operation on the 1km raster grids to calculate the mean `sowing_date_doy_nuts3` for each NUTS 3 Landkreis. This feature is the critical input for calibrating the WOFOST RPP simulation.

**VSM System 2 (Coordination / Management):**
*   **Raw Data:**
    *   Irrigation: Eurostat tables (`aei_ef_ir`, `tai03`, `ef_fsi_irri`).
    *   Crop Rotation Proxy: Destatis GENESIS Table `41241`.
*   **`preparation` Task:**
    *   Disaggregate the NUTS 2 irrigation data to the NUTS 3 level to create `irrigation_pct_nuts2`.
    *   Analyze the time-series of Table 41241 for each NUTS 3 region to calculate the year-over-year `crop_area_variance_nuts3`.

**VSM System 3 (Control / "Economic Battery"):**
*   **Raw Data (Multi-Scale Bundle):**
    *   **NUTS 3:** Destatis `41251` (Farm Size), `41241` (UAA by Crop), `41231` (Farm Types).
    *   **NUTS 2:** Eurostat `ef_kvaareg` (Standard Output), `aact_eaa01` (Consumption), `ef_mp_tenure` (Land Tenure).
    *   **NUTS 1/0:** Eurostat `apri_pi_out` & `apri_pi_in` (Price Indices).
*   **`preparation` Task:** Perform the complex disaggregation of NUTS 2 economic signals to the NUTS 3 level, using the NUTS 3 structural data as weighting keys. (e.g., `N3_SO_per_ha = N2_SO / N3_UAA`).

**VSM System 4 & 5 (Strategy & Policy):**
*   **Raw Data (Federated):**
    *   CAP Subsidies: `Gesamtliste der Begünstigten` CSV from `agrarzahlungen.de`.
    *   Nitrate Zones: "Rote Gebiete" GIS shapefiles from individual German Länder (state) portals.
    *   Market Access: Manual list of Südzucker and Nordzucker factory addresses.
*   **`preparation` Task:**
    *   Aggregate the CAP subsidies CSV by municipality and then to NUTS 3 to create `total_cap_subsidy_nuts3`.
    *   Perform a spatial join of the "Rote Gebiete" shapefiles with NUTS 3 polygons to calculate `pct_area_rote_gebiete_nuts3`.
    *   Geocode the factory addresses and calculate the road network `distance_to_processor_km_nuts3` for each NUTS 3 centroid.

---

## 4. Implemented Software Architecture
*(... section content unchanged ...)*

---

## 5. Validation and Verification Strategy
*(... section content unchanged ...)*

---

## 6. Risk Analysis and Mitigation
*(... section content unchanged ...)*
