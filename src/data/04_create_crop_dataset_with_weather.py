import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import logging
from pathlib import Path
from tqdm import tqdm

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_data(path_yield_data, path_districts_geo, path_weather_data):
    """
    Loads and prepares all necessary input files for processing.
    """
    logging.info("Loading and preparing input files...")

    # --- Load Agronomic (Yield) Data ---
    df_yield = pd.read_csv(path_yield_data)
    df_yield = df_yield[df_yield['year'].between(2017, 2024)].copy()

    # --- CRITICAL: Filter out missing yields ---
    initial_rows = len(df_yield)
    df_yield.dropna(subset=['yield'], inplace=True)
    rows_dropped = initial_rows - len(df_yield)
    if rows_dropped > 0:
        logging.info(f"Dropped {rows_dropped} rows with missing 'yield' values.")
    else:
        logging.info("No missing 'yield' values found in the agronomic data.")

    df_yield['district_no'] = df_yield['district_no'].astype(str).str.zfill(5)

    # --- Load Geospatial and Weather Data ---
    gdf_districts = gpd.read_file(path_districts_geo)
    gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)

    ds_weather = xr.open_dataset(path_weather_data)
    ds_weather.rio.write_crs("EPSG:4326", inplace=True)

    # Convert temperatures to Celsius for all calculations
    ds_weather['Temperature_Air_2m_Mean_24h'] -= 273.15
    ds_weather['Temperature_Air_2m_Max_24h'] -= 273.15
    ds_weather['Temperature_Air_2m_Min_24h'] -= 273.15

    return df_yield, gdf_districts, ds_weather


def calculate_weather_features_for_districts(gdf_districts, ds_weather):
    """
    Performs zonal statistics to calculate weather features for each district.
    """
    logging.info("Starting advanced zonal statistics for weather features...")

    results = []
    for district in tqdm(gdf_districts.itertuples(), total=len(gdf_districts), desc="Processing Districts"):
        try:
            if district.geometry is None or district.geometry.is_empty:
                logging.warning(f"Skipping district {district.district_no} due to empty geometry.")
                continue

            clipped_ds = ds_weather.rio.clip([district.geometry], gdf_districts.crs, drop=True)

            if clipped_ds.rio.width == 0 or clipped_ds.rio.height == 0:
                raise ValueError("Clipping resulted in an empty dataset.")

            daily_mean_weather = clipped_ds.mean(dim=['lat', 'lon'])

            for year in range(2017, 2025):
                year_data = daily_mean_weather.sel(time=str(year))
                peak_growth = year_data.sel(time=year_data.time.dt.month.isin([7, 8, 9]))

                results.append({
                    'district_no': district.district_no,
                    'year': year,
                    'precip_total_peak_growth': peak_growth['Precipitation_Flux'].sum().values.item(),
                    'temp_mean_peak_growth': peak_growth['Temperature_Air_2m_Mean_24h'].mean().values.item(),
                    'heat_stress_days_peak_growth': (
                                peak_growth['Temperature_Air_2m_Max_24h'] > 30).sum().values.item(),
                    'solar_rad_peak_growth': peak_growth['Solar_Radiation_Flux'].mean().values.item(),
                })

        except Exception as e:
            logging.warning(f"Could not process district {district.district_no}. Error: {e}")
            continue

    return pd.DataFrame(results)


def main():
    """Main orchestrator for creating the crop dataset with weather features."""
    logging.info("--- Starting Final Dataset Creation with Weather Features ---")

    # --- 1. Define Paths ---
    path_yield_data = Path("data/02_intermediate/sugarbeet_yield.csv")
    path_districts_geo = Path("data/01_raw/districts_official.geojson")
    path_weather_data = Path("data/02_intermediate/agera5_germany_2017_2024_merged.nc")
    output_dir = Path("data/03_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / "final_dataset_with_advanced_features.csv"

    if output_filepath.exists():
        logging.info(f"Dataset at '{output_filepath}' already exists. Skipping process.")
        return

    # --- 2. Load and Pre-filter Data ---
    df_yield, gdf_districts, ds_weather = load_and_prepare_data(
        path_yield_data, path_districts_geo, path_weather_data
    )

    # --- 3. Perform Zonal Statistics ---
    df_weather_features = calculate_weather_features_for_districts(gdf_districts, ds_weather)

    # --- 4. Assemble, Merge, and Impute Missing Weather Data ---
    logging.info("Assembling final dataset...")
    final_df = pd.merge(df_yield, df_weather_features, on=['district_no', 'year'], how='left')

    # --- 4a. NEW: State-Level Imputation ---
    weather_cols = [col for col in final_df.columns if '_growth' in col]
    missing_before = final_df[weather_cols].isnull().sum().sum()

    if missing_before > 0:
        logging.info(f"Found {missing_before} missing weather data points. Starting state-level imputation...")
        # Add state information for grouping
        districts_to_state = gdf_districts[['district_no', 'state']]
        final_df = pd.merge(final_df, districts_to_state, on='district_no', how='left')

        # Calculate state-level means for each weather column, grouped by year
        state_yearly_means = final_df.groupby(['state', 'year'])[weather_cols].transform('mean')

        # Fill missing values with the calculated state-level means
        final_df[weather_cols] = final_df[weather_cols].fillna(state_yearly_means)

        # Clean up the temporary state column
        final_df.drop(columns=['state'], inplace=True)

        missing_after = final_df[weather_cols].isnull().sum().sum()
        logging.info(f"Imputed {missing_before - missing_after} values. {missing_after} remain.")

    # Drop any rows that still have missing data (e.g., if a whole state failed)
    final_df.dropna(subset=weather_cols, inplace=True)

    # --- 4b. Format Columns ---
    logging.info("Formatting numeric columns...")
    final_df['solar_rad_peak_growth'] = pd.to_numeric(final_df['solar_rad_peak_growth'], errors='coerce').round(5)

    # --- 5. Save Final Dataset ---
    logging.info(f"Saving final dataset with {len(final_df)} rows to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False)

    logging.info("--- SUCCESS: Dataset created! ---")
    print("\n--- Final Dataset Preview ---")
    print(final_df.head())


if __name__ == "__main__":
    main()

