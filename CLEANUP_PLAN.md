# Cleanup Plan

This document lists the files and directories identified for removal or archiving to clean up the repository.

## Delete
These directories contain deprecated or experimental models that are no longer needed.
- `src/06deprecated_models/`
- `src/02_models/03_components/v31_solar/`

## Archive
These scripts are either unused, redundant, or not part of the core pipeline. They will be moved to the `archive/` directory.

### Analysis Scripts
- `src/03_analysis/basic_analysis/analyze_final_nn_model.py` (Neural Network analysis, pipeline uses XGBoost/SuperEnsemble)
- `src/03_analysis/basic_analysis/cluster_regime_discovery.py` (Experimental clustering)
- `src/03_analysis/visualization/visualize_data_transformations.py` (Appears misnamed "evaluate_model_robustly.py", likely redundant)

### Model Scripts
- `src/02_models/02_features/analyze_stage1_features.py` (Duplicate/Unused, superseded by `src/03_analysis/basic_analysis/analyze_stage1_features.py`)

## Notes
- `src/02_models/01_simulation/Wofost7.2/` is the active WOFOST directory (docs mention 7.1, but files match).
- `src/02_models/04_super_ensemble/` files are all core.
- `src/02_models/03_components/` subdirectories (hybrid_xgb, native_ensemble, robust_linear, statistical) contain core files.
