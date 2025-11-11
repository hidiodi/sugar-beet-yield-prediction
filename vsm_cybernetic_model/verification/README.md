# VSM-CPS Verification

This directory contains the top-level, post-hoc verification scripts for the VSM-CPS model.

## Scripts

-   `01_run_post_hoc_shap_analysis.py`: This is the final and most important validation step. It runs after the final XGBoost "regulator" model has been trained and generates a SHAP summary plot. This plot shows the global importance of the high-level VSM indices (e.g., `VSM3_PC1`), allowing you to verify that the engineered features are indeed the primary drivers of the final prediction.

## Note on Unit Testing

This directory focuses on model and feature validation. **Unit tests** for the core data processing functions are a critical part of the overall validation strategy but are not included here.

It is strongly recommended that you create a `tests/` directory at the top level of the `vsm_cybernetic_model` and implement unit tests for the complex logic in the `preparation` and `model` scripts to ensure their technical correctness.
