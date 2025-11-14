# WOFOST Simulation Pipeline Report

This document provides a comprehensive overview of the refactored WOFOST simulation pipeline, detailing its architecture, functionality, and potential areas for future improvement.

## How the WOFOST Pipeline Works: A Data-First Architecture

The refactored pipeline operates on a **"data-first"** or **"builder-consumer"** model. This architecture strictly separates the slow, complex process of data preparation from the actual model simulation. This makes the entire workflow more modular, efficient, and transparent.

The process is divided into two main stages:

### Stage 1: The "Builders" - Pre-calculating All Simulation Inputs

Three dedicated Python scripts are responsible for generating all the data needed for the simulations. Each script is a "builder" that creates a specific, well-defined data asset. **Note: These scripts have been optimized and now run their main processing loops in parallel to significantly reduce execution time.**

1.  **`build_initial_conditions.py`**: This script calculates the starting state for every simulation. It reads historical weather data and soil parameters to determine the dynamic sowing date and the initial available soil moisture (WAV) for each district and year.
    *   **Output:** `InitialConditions.csv`

2.  **`build_genetic_parameters.py`**: This script accounts for improvements in crop genetics over time. It loads a base set of crop parameters from a YAML file and applies a "genetic gain" formula for each year, creating a unique set of physiological parameters (like radiation-use efficiency and heat sum requirements) for every year in the simulation period.
    *   **Output:** `SugarbeetGenes.json`

3.  **`build_forecast_weather.py`**: This is the most computationally intensive builder. It first learns the statistical weather patterns from historical data. Then, using seasonal forecast anomalies, it generates a full 51-member ensemble of synthetic daily weather scenarios for every district and year.
    *   **Output:** `ForecastedWeather.csv`

### Stage 2: The "Consumer" - Running the Simulations

The main **`run_wofost_pipeline.py`** script is now a lean "consumer." Its sole responsibility is to run the WOFOST model. It no longer contains any data preparation logic. Instead, it simply loads the pre-calculated data assets created by the builders and feeds them into the simulation engine. This makes the script faster to execute and much easier to debug.

### Verification: The "Validation Dashboard"

To ensure the integrity of this new pipeline, a dedicated analysis script, **`validation_dashboard.py`**, has been created. This script acts as a quality control dashboard, generating plots and reports to:
-   Verify that the input data assets (e.g., `InitialConditions.csv`) are sane and physically realistic.
-   Confirm that the WOFOST output is within a plausible range.
-   Validate that the model is behaving logically (e.g., more initial water leads to less drought stress).
-   **New:** Generate spatial maps of key outputs (e.g., mean yield, drought stress) to identify regional patterns.

## Key Enhancements Implemented

The pipeline has been upgraded with several significant enhancements to improve its accuracy and capabilities:

### 1. Dynamic Initial Conditions Using Satellite Data
The `build_initial_conditions.py` script now incorporates pre-season satellite data to create a more dynamic and accurate starting point for the simulations. Specifically, it uses the **winter NDVI anomaly** (a measure of vegetation health) to adjust the calculated initial soil moisture (WAV). A healthier-than-average pre-season vegetation (positive anomaly) results in a slight increase in the starting soil moisture, better reflecting real-world conditions.

### 2. Regional Parameter Calibration
A new script, `calibrate_regional_parameters.py`, provides the capability to tune the WOFOST crop parameters for different geographical regions of Germany. It uses a scientific optimization algorithm to find the best-fitting parameters (e.g., AMAX, TSUM1) by minimizing the error between simulated yields and historical data for each region. This allows the model to account for regional differences in farming practices and local crop varieties.
*   **Output:** `SugarbeetGenes_Regional.json`

## What the Pipeline Does: Simulating Crop Yield

The primary purpose of the WOFOST pipeline is to **simulate the growth and final yield of sugarbeet** across all German districts for a specified range of years (e.g., 1982-2024). It serves as a physics-based "first guess" for the pre-season yield forecast.

The pipeline performs two distinct types of simulations:

1.  **Historical Simulation:** For each year in the past, the model is run using the **observed historical weather**. This provides a baseline, "perfect weather" simulation that shows what the yield would have been under real-world weather conditions. This is crucial for model calibration and for understanding the model's performance against actual reported yields.

2.  **Forecast Simulation:** For each year, the model is run **51 times** using a full ensemble of synthetically generated weather scenarios. These scenarios are based on seasonal climate forecasts. This ensemble approach allows us to capture the uncertainty in the weather forecast and produce a probabilistic yield forecast, not just a single number.

### Key Outputs

The main output of the pipeline is a detailed CSV file (`forecast_ensemble.csv`) containing the results of every single simulation run. The key columns include:

-   **`yield_potential_dry_kgha`**: The maximum possible yield, assuming no water stress.
-   **`yield_water_limited_dry_kgha`**: The yield when the only limiting factor is water availability. This is the primary output used in downstream models.
-   **`drought_stress_index`**: A diagnostic metric that quantifies the impact of water stress on the crop during the simulation.

## How We Could Improve It: Future Enhancements

The pipeline is now significantly more robust and accurate. The following proposals outline the next steps for further development:

1.  **Integrate Nutrient-Limited Simulations:**
    *   **Proposal:** The current model simulates yield under optimal (potential) and water-limited conditions. The next major step is to implement **nutrient-limited simulations**, primarily focusing on Nitrogen (N). This would involve adding soil nutrient data to the `StaticSiteData.csv` and configuring the WOFOST model to run in a nutrient-limited mode.
    *   **Benefit:** This would provide a more realistic yield forecast by accounting for one of the most significant factors in crop growth, allowing us to differentiate between yield losses from drought versus poor soil fertility.

2.  **Expand and Automate the Calibration Routine:**
    *   **Proposal:** The current calibration script is powerful but could be expanded. A more advanced version could be built to optimize a wider range of crop parameters and run automatically as new yield data becomes available.
    *   **Benefit:** An automated, more comprehensive calibration system would ensure the model is always using the most up-to-date and accurate parameters, improving the reliability of the entire forecast pipeline.

3.  **Integrate More Advanced Satellite Products:**
    *   **Proposal:** Move beyond the simple NDVI anomaly and integrate more sophisticated satellite data products. This could include using Sentinel-1 radar data for a direct measure of soil moisture or using Sentinel-2 data to estimate in-season Leaf Area Index (LAI) for data assimilation.
    *   **Benefit:** Directly assimilating satellite measurements into the live model run could help "course-correct" the simulation, leading to a significant improvement in in-season forecast accuracy.

4.  **Develop a Scenario Analysis Module:**
    *   **Proposal:** Create a new script or module that allows users to easily define and run "what-if" scenarios. For example, a user could define a scenario with a hypothetical +2°C temperature anomaly or a 20% reduction in rainfall to see how the model responds.
    *   **Benefit:** This would transform the pipeline from a simple forecast tool into a powerful climate-risk analysis engine, allowing for the assessment of potential climate change impacts on agriculture.
