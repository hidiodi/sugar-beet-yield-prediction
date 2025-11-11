# VSM-Cybernetic-Physical System (VSM-CPS) Model (v1.0 - Implemented)

This module contains the fully implemented blueprint for the Viable System Model (VSM) - Cyber-Physical System (CPS) diagnostic model. The core logic for training the expert engines and the final regulator model is complete.

The system is now ready for data integration. The user's primary task is to implement the data loading and processing logic within the `preparation` subdirectories of each system module.

## Architecture Overview (Implemented)

-   **`/system_*`**: Each VSM system module contains the logic for training its "Expert Engine."
-   **`/configs`**: Contains all file paths and model parameters. **This is the main place for user configuration.**
-   **`/models`**: The target directory where all trained model artifacts (`.joblib` files for scalers and engines) will be saved.
-   **`/pipelines`**: Contains the main entry points for running the system.

## How to Use: Your Data Integration Task

Your task is to connect your raw data sources to the system. The implemented model scripts expect specific foundational feature files as inputs. You must implement the scripts in the `preparation` directories to generate these files.

**Step 1: Place Your Raw Data**
-   Place all your raw data files (CSVs, shapefiles, etc.) into the `data/01_raw/` directory at the root of the repository.

**Step 2: Define Your Schemas**
-   Open the `pipelines/00_define_human_system_schema.py` and `system_1_biophysical/preparation/01_load_wofoost_inputs.py` files.
-   The docstrings in these files contain the **exact DataFrame schemas** that your preparation scripts need to produce.

**Step 3: Implement the `preparation` Scripts**
-   Go through each `preparation` script in the `system_*` directories.
-   Replace the placeholder logic with your own Python code (using `pandas`, `geopandas`, etc.) to load your raw data and transform it into the required foundational feature tables.
-   Ensure your scripts save their final outputs to the paths defined in `configs/main_config.py` (e.g., `FOUNDATIONAL_FEATURES_HUMAN`).

**Step 4: Run the Training Pipeline**
-   Once your preparation scripts are implemented and can successfully generate the foundational feature files, you can run the main training pipeline.
-   From your terminal, at the root of the repository, execute the following command:
    ```bash
    python -m vsm_cybernetic_model.pipelines.01_run_feature_engineering_pipeline train
    ```
-   This will train all the Stage 1 "Expert Engine" models and save them to the `vsm_cybernetic_model/models/stage_1_experts/` directory.

**Step 5: Run the Transformation and Final Model Training**
-   After the expert engines are trained, you can generate the final feature matrix and train the main XGBoost regulator.
    ```bash
    # Generate the final features using the trained engines
    python -m vsm_cybernetic_model.pipelines.01_run_feature_engineering_pipeline transform

    # Train the final XGBoost model
    python -m vsm_cybernetic_model.pipelines.02_run_model_training_pipeline
    ```

Your system is now fully trained and ready for analysis.
