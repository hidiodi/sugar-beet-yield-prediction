# VSM-Cybernetic-Physical System (VSM-CPS) Model

This module contains the implementation blueprint for the Viable System Model (VSM) - Cyber-Physical System (CPS) diagnostic model for German sugar beet production. It is designed as a modular, runnable scaffold that can be progressively populated with data and logic as they are developed.

## Architecture Overview

The architecture is explicitly designed to mirror the VSM framework, separating the different logical systems into their own Python modules. This allows for clear ownership and targeted development.

-   **`/system_1_biophysical`**: Models the core biophysical operations (VSM System 1). This is where the WOFOST-as-a-sensor logic resides, measuring the plant's internal homeostatic responses.
-   **`/system_2_coordination`**: Models the coordination and management activities of farmers (VSM System 2). This includes proxies for planting/harvesting timing and crop rotation choices.
-   **`/system_3_control`**: Models the internal audit, control, and economic realities of the farm (VSM System 3). This is where the crucial `Economic_Battery_Index` is constructed.
-   **`/system_4_strategy`**: Models the farm's response to the external market environment (VSM System 4). This includes features like market access (distance to processors).
-   **`/system_5_policy`**: Models the overarching policy, identity, and regulatory constraints of the system (VSM System 5). This includes features derived from CAP subsidies and NVZ regulations.

Each system directory is further divided into a clear engineering pipeline:
-   **`/preparation`**: Scripts for loading, cleaning, and preparing raw data.
-   **`/model`**: Scripts for building the high-level VSM features and composite indices.
-   **`/verification`**: Scripts and notebooks for validating the engineered proxies (e.g., plotting maps, checking correlations).

## Configuration

All configuration for the modules is centralized in the `/configs` directory. Each system has its own configuration file (`system_*.py`) for specific parameters, and `main_config.py` holds global parameters like file paths and model settings.

## Pipelines

The main entry points for running the model are located in the `/pipelines` directory. These scripts orchestrate the calls to the individual system modules in the correct order.

-   `01_run_feature_engineering_pipeline.py`: Executes the full `preparation` and `model` steps for all systems to build the final feature matrix.
-   `02_run_model_training_pipeline.py`: Loads the final feature matrix and trains the XGBoost "regulator" model.
-   `03_run_verification_pipeline.py`: Runs all `verification` scripts to generate validation plots and reports.

## How to Use

1.  **Populate Data**: Place the required raw data into the appropriate directories (to be created, e.g., `data/01_raw/`).
2.  **Configure**: Update the parameters in the `/configs` files.
3.  **Implement Logic**: Fill in the placeholder scripts in each module with the data processing and modeling logic.
4.  **Run Pipeline**: Execute the main pipeline scripts from the root of the repository to generate features, train the model, and run verifications.
