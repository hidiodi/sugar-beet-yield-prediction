from src.utils.pipeline_runner import run_pipeline

def main():
    """
    Defines and executes the main data processing pipeline.
    This script is just a definition file; all execution logic
    is handled by the pipeline_runner.
    """
    PIPELINE_NAME = "Main Data Processing Pipeline"

    # Define the sequence of scripts to execute.
    # Paths must be relative to this file's location (the project root).
    SCRIPTS_TO_RUN = [
        "src/01_data/Weather_Data/AGERA_Weatherdata/process_agera5_data.py",
        "src/01_data/Sugarbeetdata/process_agronomic_data.py",
        "src/01_data/Weather_Data/SEASForecastData/ECMWF51/build_forecast_features_by_member.py",
        "src/01_data/Weather_Data/04_create_crop_dataset_with_weather.py",
        "src/02_models/Wofost/04_create_daily_weather_file.py",
    ]

    # Execute the pipeline
    run_pipeline(pipeline_name=PIPELINE_NAME, script_paths=SCRIPTS_TO_RUN)

if __name__ == "__main__":
    main()