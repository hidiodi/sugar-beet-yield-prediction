# Journal Paper Plan

This document outlines the final steps to prepare the paper "Hybrid Yield Forecasting Model: Bridging Process-Based Simulation and Machine Learning" for submission.

## 1. Repository Status
The repository has been cleaned:
- Deprecated models (`src/06deprecated_models/`, `src/02_models/03_components/v31_solar/`) have been removed.
- Unused analysis scripts have been archived in `archive/`.
- The core pipeline configuration (`src/02_models/execute_hybrid_pipeline.py`) has been updated to point to the correct WOFOST version (7.2).

## 2. Action Items

### 2.1 Ablation Study
To validate the model components, we need to run specific experiments.

**Action:** Create a script `src/03_analysis/planning/run_ablation_study.py` that:
1.  **Baseline:** Runs the full pipeline (or ensures baseline results exist).
2.  **Experiment 1 (Feature Importance):**
    - Modifies `src/02_models/config.py` (or feature generation scripts) to exclude "Scorch Index" and "Anoxia".
    - Retrains the Hybrid XGBoost and Super Ensemble.
    - Compares MAE/Skill Score for 2003 and 2018 against baseline.
3.  **Experiment 2 (Meta-Learner):**
    - Retrains the Super Ensemble using a simple "Equal Weight" strategy (average of components) instead of the Meta-Regressor.
    - Compares overall MAE.
4.  **Experiment 3 (Data Quality):**
    - Retrains without the "Garbage Filter" (Oracle Error > 200 dt/ha included).
    - Checks for degradation in robustness.

### 2.2 Visualization
We need to generate the high-quality figures for the paper.

**Action:** Update `src/03_analysis/visualization/generate_paper_plots.py` to include:
-   **Figure 2: Time Series of Actual vs. Predicted Yields (2000–2024).**
    -   Must show "Actual", "Super Ensemble", and "Trend" lines.
    -   Highlight 2003 and 2018.
-   **Figure 3: Ablation Results (Optional).**
    -   Bar chart comparing MAE of Baseline vs Ablation experiments.

### 2.3 Documentation
Fill in the specific details required for reproducibility.

**Action:** Update the paper draft (or a `SUPPLEMENTARY_METHODS.md` file) with:
-   **Data Provenance:** Source of weather grids (DWD HYRAS/ERA5), SoilGrids version, MODIS/Sentinel product IDs.
-   **Hyperparameters:** ARIMA order (1,0,0), GAM parameters, XGBoost settings.
-   **Feature Formulas:** Explicit math for `z_rain`, `z_sow`, `Index_Bumper`, `drown`, `late_start`.

## 3. Execution Order
1.  **Implement `run_ablation_study.py`** (or manually run the experiments).
2.  **Update `generate_paper_plots.py`** and generate the final figures.
3.  **Compile the Latex/Word document** with the new figures and numbers.
4.  **Final Review** of the repository structure (verify `README.md` instructions work).
