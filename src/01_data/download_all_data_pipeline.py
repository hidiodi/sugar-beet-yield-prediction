from src.utils.pipeline_runner import run_pipeline

def main():

    PIPELINE_NAME = "Main Data Downloading Pipeline"

    # Define the sequence of scripts to execute.
    # Paths must be relative to this file's location (the project root).
    SCRIPTS_TO_RUN = [
        "src/01_data/StaticSoil_data/build_static_features.py",
        "src/01_data/WinterSatellite_data/build_satellite_features.py",
        "src/01_data/Weather_Data/SEASForecastData/ECMWF51/download_ECMWF51_forecast.py",
        "src/01_data/Weather_Data/AGERA_Weatherdata/download_agera5_data.py"
    ]

    # Execute the pipeline
    run_pipeline(pipeline_name=PIPELINE_NAME, script_paths=SCRIPTS_TO_RUN)

if __name__ == "__main__":
    main()