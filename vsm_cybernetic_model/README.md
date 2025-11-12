# VSM-Cybernetic-Physical System (VSM-CPS) Model (v1.1 - Baseline Model)

This module contains a runnable baseline implementation of the Viable System Model (VSM) diagnostic model. It has been refactored to work **end-to-end** using a single, curated feature set: `data/04_master/master_dataset.csv`.

The complex data sourcing and preparation pipelines have been temporarily removed to create a stable, verifiable baseline. The focus of this version is on the core VSM architecture and the training of the "Expert Engines" and the final "Regulator" model.

## Architecture Overview (Baseline)

-   **`/system_*`**: Each VSM system module contains the logic for its "Expert Engine".
-   **`/configs`**: Contains all file paths and model parameters. The feature lists in `configs/vsm*.py` have been mapped to the columns in the `master_dataset.csv`.
-   **`/pipelines`**: Contains the main entry points for running the system.
-   **`/verification`**: Contains scripts to validate the model's outputs and internal logic.

## How to Run the Baseline Model

This model is designed to be run from the root of the repository. The pipeline is broken into sequential steps.

**Prerequisites:**
-   Ensure you have the required Python packages installed. A `requirements.txt` is not provided, but you will need `pandas`, `xgboost`, `scikit-learn`, `joblib`, `matplotlib`, and `seaborn`.

**Step 1: Prepare Foundational Data**
-   This step is a simple passthrough that copies the master dataset to the location the pipeline expects.
    ```bash
    python -m vsm_cybernetic_model.pipelines.prepare_foundational_features
    ```

**Step 2: Train the Stage 1 "Expert Engines"**
-   This command trains the five VSM expert engines (PCA models) and saves them to the `vsm_cybernetic_model/models/stage_1_experts/` directory.
    ```bash
    python -m vsm_cybernetic_model.pipelines.01_run_feature_engineering_pipeline train
    ```

**Step 3: Transform Data and Create Final Features**
-   This command uses the trained expert engines to transform the foundational data into the final feature matrix for the regulator model.
    ```bash
    python -m vsm_cybernetic_model.pipelines.01_run_feature_engineering_pipeline transform
    ```

**Step 4: Train the Final Stage 2 "Regulator" Model**
-   Finally, this command trains the XGBoost model on the final features.
    ```bash
    python -m vsm_cybernetic_model.pipelines.02_run_model_training_pipeline
    ```

**Step 5: Run Verification**
-   After training, you can run the verification scripts to perform plausibility checks and analysis.
    ```bash
    python -m vsm_cybernetic_model.pipelines.03_run_verification_pipeline
    ```

The system is now fully trained and verified. You can find model artifacts in `vsm_cybernetic_model/models/` and verification plots in `vsm_cybernetic_model/verification/`.
