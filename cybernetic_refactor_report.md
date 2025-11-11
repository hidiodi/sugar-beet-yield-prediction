# Technical Documentation for the VSM-CPS Diagnostic Model (v3.0)

## 1. Executive Summary

This document provides the official technical and architectural documentation for the implemented VSM-CPS (Viable System Model - Cyber-Physical System) diagnostic model (v3.0). The system is a modular, multi-stage machine learning pipeline designed to diagnose the systemic drivers of German sugar beet yield.

The core architecture is built around a series of unsupervised "Expert Engines" (PCA models) that distill low-level data into high-level indices representing each of the five VSM subsystems. A key innovation is the VSM System 1 "Sensor," which uses a WOFOST crop model calibrated with real-world management data to generate a "Realistic Physical Potential" (RPP) baseline.

This document details the model's rationale, the final software architecture, the data processing workflow, and the embedded strategies for validation and risk management.

---

## 2. Model Rationale

### 2.1. How the Model Works

The VSM-CPS model is a hierarchical pipeline that transforms a wide array of raw data into a final diagnostic prediction.

1.  **Foundational Feature Engineering:** User-implemented scripts in the `preparation/` directories load raw data and generate a set of clean "foundational features."
2.  **VSM 1 "Sensor" Calibration & Simulation:** The WOFOST model is calibrated using VSM 2 (Coordination) data (e.g., observed sowing dates) to produce a "Realistic Physical Potential" (RPP) baseline. This is handled by `system_1_biophysical/model/01_run_rpp_simulations.py`.
3.  **Expert Engine Training:** In "train" mode, a series of unsupervised PCA models are trained to find the latent structure within the foundational features for each VSM system. These fitted `scaler` and `pca` model artifacts are saved to disk. This is handled by the `system_*/model/0*_train_*_engine.py` scripts.
4.  **VSM Index Transformation:** In "transform" mode, the saved expert engine artifacts are loaded and used to convert the foundational features into the final VSM indices (e.g., `VSM3_PC1` becomes the `Economic_Battery_Index`).
5.  **System Diagnosis:** A final XGBoost "regulator" model is trained on these VSM indices to predict the gap between the RPP and the final observed yield, allowing for diagnostic attribution.

### 2.2. What the Model Does: Prediction vs. Diagnosis

The primary function of the model is **diagnostic attribution**. Its core purpose is to answer: "**Why** did the yield deviate from its potential?" By attributing the final prediction to features that represent specific VSM systems, the model can differentiate between a **VSM 1 (Biophysical) failure** and a **VSM 3 (Economic) failure**.

### 2.3. Architectural Superiority

This architecture is superior to a standard correlational model because it:
*   **Creates a Cleaner Signal:** By using a realistic RPP baseline, the "gap" that the final model predicts is a much more precise measure of failures caused by economic, market, and policy factors (VSM 3-5).
*   **Enables Causal Diagnosis:** The model's features are not just variables; they are proxies for systemic functions, which allows for a more causal and interpretable diagnosis.
*   **Is Data-Driven:** The use of unsupervised "Expert Engines" allows the model to learn the complex, non-linear relationships that define a system's state, rather than relying on brittle, hand-coded formulas.

---

## 3. Implemented Software Architecture

The system is implemented as a self-contained Python module, `vsm_cybernetic_model/`, separate from the project's legacy `src/` directory.

### 3.1. Directory Structure & Data Flow

*   **`/configs`:** Centralized configuration for all file paths and model parameters. This is the primary location for user adjustments.
*   **`/preparation` (within each system):** These user-implemented scripts are the entry point for raw data. They must be modified to load data and produce the specific foundational feature tables required by the system.
*   **`/model` (within each system):** These scripts contain the fully implemented logic for training the unsupervised expert engines.
*   **`/models/stage_1_experts`:** This directory is the designated storage location for all the trained `.joblib` artifacts (scalers and PCA models).
*   **`/pipelines`:** These scripts orchestrate the end-to-end workflow.
    *   `01_run_feature_engineering_pipeline.py`: The main entry point, with a `train` mode to create the expert engines and a `transform` mode to generate the final feature matrix.
    *   `02_run_model_training_pipeline.py`: Trains the final XGBoost regulator model.
    *   `03_run_verification_pipeline.py`: Executes all `verification` scripts to provide a system health check.

### 3.2. Data Workflow

The data processing follows a clear, multi-stage sequence:
1.  **Raw Data** -> `preparation` scripts
2.  -> **Foundational Features** (`.csv` files in `data/02_intermediate/`) -> `model` scripts (in `train` mode)
3.  -> **Expert Engine Artifacts** (`.joblib` files in `vsm_cybernetic_model/models/`)
4.  **Foundational Features** + **Expert Engine Artifacts** -> `pipelines` script (in `transform` mode)
5.  -> **Final VSM Feature Matrix** (`.csv` file in `data/05_model_input/`) -> `02_run_model_training_pipeline.py`
6.  -> **Final Regulator Model** (`.joblib` file).

---

## 4. Validation and Verification Strategy

The system includes a robust, multi-layered validation strategy.

*   **Unit Testing:** Core functions should be unit-tested to ensure technical correctness.
*   **Foundational Feature Validation:** The `verification` scripts are designed for "sense-making" of the foundational features through geospatial plots and correlation analyses.
*   **RPP Baseline Validation:** The `RPP_mean_yield` must be checked for plausibility against the final real yield and known agronomic conditions.
*   **Unsupervised Model Validation:** The expert engines must be validated for interpretability and statistical significance by analyzing their **explained variance** and **component loadings**.
*   **Post-Hoc Model Validation:** The final VSM indices should be validated by confirming they have high importance (e.g., via SHAP) in the final regulator model.

---

## 5. Risk Analysis and Mitigation

*   **Data Availability:** Changes in external data sources are managed by encapsulating data loading in the `preparation` scripts and caching downloaded raw data.
*   **Proxy Accuracy:** The risk of uninterpretable "black box" indices is the most significant. This is mitigated by the mandatory **Component Loading Analysis** as part of the validation strategy. If an expert engine is not interpretable, it must be re-configured.
*   **Model Stability:** The risk of unstable unsupervised models is mitigated by version controlling all trained `.joblib` artifacts and using fixed `random_state` parameters.
*   **Temporal Mismatch:** The use of decennial census data for annual predictions is a known limitation and must be documented in any analysis. The impact is lessened by the presence of other, more dynamic annual features.