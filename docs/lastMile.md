# Final Mile Plan: converting Draft to Valid Scientific Journal Paper

This document outlines the remaining steps required to finalize the paper "Hybrid Yield Forecasting Model: Bridging Process-Based Simulation and Machine Learning" for submission to a scientific journal.

## 1. Ablation Study (Impact Analysis)

To rigorously validate the contribution of specific components, we must perform an ablation study. This involves removing parts of the system and measuring the degradation in performance (MAE/Skill Score).

*   **Feature Importance Validation**:
    *   **Task**: Retrain the model without the **"Scorch Index"** and **"Anoxia"** features.
    *   **Hypothesis**: Performance during extreme years (2003, 2018) should drop significantly.
    *   **Output**: Add a "Feature Ablation" table to the Results section.

*   **Meta-Learner Validation**:
    *   **Task**: Compare the "Super Ensemble" (Soft Voting) against a simple **"Equal Weight Ensemble"** (Average of all components).
    *   **Hypothesis**: The Meta-Learner should outperform the simple average, proving that the dynamic switching logic is valuable.
    *   **Output**: Add this comparison to Table 1 or a new subsection in Results.

*   **Data Quality Control Validation**:
    *   **Task**: Retrain the Meta-Learner *without* the "Garbage Filter" (i.e., include rows with Oracle Error > 200 dt/ha).
    *   **Hypothesis**: The model should be less robust and have higher error, proving the necessity of filtering data entry errors.
    *   **Output**: Mention the quantitative impact in the Discussion (Signal-to-Noise Ratio).

## 2. Documentation & Transparency (Reproducibility)

Scientific journals require high reproducibility. We need to fill in missing details.

*   **Data Provenance**:
    *   **Action**: Explicitly state the source of "interpolated weather grids" (e.g., DWD HYRAS, ERA5, or other).
    *   **Action**: Specify the version/resolution of SoilGrids used.
    *   **Action**: Specify the exact product IDs for MODIS/Sentinel NDVI data.

*   **Hyperparameters**:
    *   **Action**: Document the ARIMA order `(1, 0, 0)` in the "Statistical Trend Baseline" section.
    *   **Action**: Document the GAM Splines parameter (`n_splines=10`) in the same section.
    *   **Action**: Ensure all XGBoost hyperparameters (learning rate, depth, etc.) are consistent between code and text.

*   **Feature Dictionary**:
    *   **Action**: Create a **Supplementary Material** section or Appendix listing *all* engineered features.
    *   **Content**: Must include formulas for `z_rain`, `z_sow`, `Index_Bumper`, `drown`, `late_start`, `VegetationVigorIndex`, and `RootZoneDepletion`.

## 3. Visualization

*   **Generate Figure 2**:
    *   **Task**: Create the "Time Series of Actual vs. Predicted Yields (2000–2024)" plot.
    *   **Details**: It must clearly show the "Actual", "Super Ensemble", and "Trend" lines, highlighting the 2003 and 2018 drought events to visually demonstrate the skill improvement.
    *   **Action**: Write a Python script (similar to `chartGen.py`) to generate this plot from the results data.

## 4. Paper Finalization

*   **Terminology**:
    *   **Action**: Finalize the name "Super Ensemble". If "Safe Ensemble" or "Hybrid Yield Model" is preferred, standardize it across the Abstract, Introduction, and Methods.
    *   **Action**: Ensure consistency in naming "Stage 1" vs "Stage 2" features.

*   **Consistency Check**:
    *   **Action**: Verify that the numbers in the Abstract match the numbers in the Results tables exactly.
    *   **Action**: Ensure the "Introduction" and "Discussion" tell a coherent story about the "Interpolation vs. Extrapolation" dilemma.

*   **References**:
    *   **Action**: Verify that all citations (`\cite{...}`) in `main.tex` have corresponding valid entries in `references.bib`.

## 5. Clean Up & Repository Hygiene

To ensure the repository is clean and only contains necessary files for reproduction, we have identified the core pipeline files. Any file not listed below should be reviewed for removal or archiving.

### 5.1 Core Pipeline Files (Do Not Remove)

These files are explicitly called in `src/02_models/execute_hybrid_pipeline.py` or are dependencies.
**Simulation Prep & Execution (WOFOST):**
*    `src/02_models/01_simulation/Wofost7.2/prepare_genetic_parameters.py`,
*    `src/02_models/01_simulation/Wofost7.2/prepare_site_data.py`,
*    `src/02_models/01_simulation/Wofost7.2/prepare_forecast_weather.py`,
*    `src/02_models/01_simulation/Wofost7.2/prepare_initial_conditions.py`,
*    `src/02_models/01_simulation/Wofost7.2/execute_wofost_simulation.py`,


**Simulation & Heat Signal:**
*   `src/02_models/01_simulation/Wofost7.2/*.py` (Prepare & Execute scripts)
*   `src/02_models/01_simulation/multivariate_heat_signal.py`

**Feature Engineering:**
*   `src/02_models/02_features/generate_stage1_features.py`
*   `src/02_models/02_features/generate_stage2_features.py`

**Component Models:**
*   `src/02_models/03_components/statistical/estimate_yield_trend.py`
*   `src/02_models/03_components/native_ensemble/train_physics_informed_model.py`
*   `src/02_models/03_components/native_ensemble/train_physics_ensemble.py`
*   `src/02_models/03_components/hybrid_xgb/train_yield_ratio_xgb.py`
*   `src/02_models/03_components/hybrid_xgb/backtest_yield_ratio_xgb.py`
*   `src/02_models/03_components/robust_linear/train_robust_integrator.py`

**Super Ensemble:**
*   `src/02_models/04_super_ensemble/prepare_ensemble_data.py`
*   `src/02_models/04_super_ensemble/train_meta_regressor.py`
*   `src/02_models/04_super_ensemble/execute_ensemble_forecast.py`

**Analysis & Diagnostics:**
*   `src/02_models/05_analysis/check_data_leakage.py`
*   `src/02_models/05_analysis/analyze_error_distribution.py`
*   `src/02_models/05_analysis/analyze_super_ensemble.py`
*   `src/02_models/05_analysis/compare_model_versions.py`

**Configuration & Orchestration:**
*   `src/02_models/config.py`
*   `src/02_models/execute_hybrid_pipeline.py`
*   `src/utils/pipeline_runner.py`

### 5.2 Cleanup Status

The repository has been cleaned.
*   **Deprecated Models:** Removed `src/02_models/06deprecated_models/` and `src/02_models/03_components/v31_solar/`.
*   **Archived Scripts:** Unused analysis scripts and experimental components have been moved to `archive/`.
