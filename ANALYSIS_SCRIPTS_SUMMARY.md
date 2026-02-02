# Analysis Scripts Summary

This document summarizes the analysis scripts found in the repository, categorizing them by their purpose and recommending their status (Keep/Archive).

## Core Pipeline Analysis (`src/02_models/05_analysis/`)
These scripts are part of the core pipeline as defined in `docs/lastMile.md`.

| Script | Description | Status |
| :--- | :--- | :--- |
| `analyze_error_distribution.py` | Analyzes error distribution, potential for confusion matrices/classification reports. | **Keep** |
| `analyze_super_ensemble.py` | Analyzes the Super Ensemble model performance. | **Keep** |
| `check_data_leakage.py` | Checks for data leakage using an adversarial classifier. | **Keep** |
| `compare_model_versions.py` | Compares different model versions (e.g., R2, MAE). | **Keep** |
| `visualize_error_map.py` | visualizes error maps (Geopandas/Matplotlib). | **Keep** |

## Planning & Value Story (`src/03_analysis/planning/`)
Scripts that generate the "Value Story" and simulations for the final paper.

| Script | Description | Status |
| :--- | :--- | :--- |
| `generate_forecast_dashboard.py` | Generates an interactive Folium map comparing model forecasts to reality. | **Keep** |
| `simulate_yield_optimization.py` | Runs a 40-year simulation to quantify yield improvement (Value Proof). | **Keep** |

## Visualization (`src/03_analysis/visualization/`)
Scripts dedicated to generating plots for the paper.

| Script | Description | Status |
| :--- | :--- | :--- |
| `generate_paper_plots.py` | Generates various plots for the paper. | **Keep** |
| `visualize_error_map.py` | (Duplicate name/functionality?) Visualizes error map. | **Keep** |
| `visualize_model_predictions.py` | Visualizes model predictions (Validation vs Test). | **Keep** |
| `01_visualize_static_features.py` | Visualizes static features. | **Keep** |
| `visualize_data_transformations.py` | Header says `evaluate_model_robustly.py`. Checks model performance. | **Review** |

## Basic Analysis (`src/03_analysis/basic_analysis/`)
General analysis and diagnostic scripts.

| Script | Description | Status |
| :--- | :--- | :--- |
| `analyze_final_nn_model.py` | Analyzes a Neural Network model. **Note:** Pipeline currently focuses on XGBoost/SuperEnsemble. | **Archive** |
| `analyze_outliers.py` | Analyzes outliers in the data/predictions. | **Keep** |
| `analyze_stage1_features.py` | detailed diagnostics of Stage 1 features. | **Keep** |
| `check_forecast_quality.py` | Checks forecast quality (Pearson/Spearman correlations). | **Keep** |
| `cluster_regime_discovery.py` | Exploratory clustering. Likely experimental. | **Archive** |
| `compare_model_versions.py` | Compares model versions. | **Keep** |
| `comprehensive_model_comparison.py` | Compare multiple models. | **Keep** |
| `deep_dive_errors.py` | Deep dive into errors (confusion matrix etc). | **Keep** |
| `diagnose_hybrid_model.py` | "Professional" diagnostic script with subgroup discovery. | **Keep** |
| `explain_stage1_model.py` | SHAP analysis for Stage 1 model. | **Keep** |
| `run_counterfactual_analysis.py` | Runs counterfactual analysis. | **Keep** |
| `run_hybrid_analysis_pipeline.py` | Orchestrates the analysis pipeline. | **Keep** |
| `run_outlier_analysis.py` | Runs outlier analysis. | **Keep** |
| `sanity_check_leakage.py` | Checks for leakage. | **Keep** |
| `shap_analysis_xgb.py` | SHAP analysis for XGBoost. | **Keep** |
| `super_analyzer.py` | Meta-analysis script. | **Keep** |
| `verification_script.py` | Verifies data integrity/GeoJSON. | **Keep** |

## Hybrid Model Analysis (`src/03_analysis/hybrid_model_analysis/`)
Specific analysis for the hybrid model components.

| Script | Description | Status |
| :--- | :--- | :--- |
| `analyze_hybrid_model.py` | Diagnostics for Hybrid & Standalone models. | **Keep** |
| `analyze_input_features.py` | Analyzes input features. | **Keep** |
| `analyze_wofost_pipeline.py` | Comprehensive diagnostic of the agricultural forecasting pipeline. | **Keep** |
| `assess_feature_strategy.py` | Assesses feature strategy. | **Keep** |
| `feature_correlation_analysis.py` | Analyzes feature correlations. | **Keep** |
