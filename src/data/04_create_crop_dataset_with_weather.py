# src/data/04_create_crop_dataset_with_weather.py (ADVANCED VERSION)

import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_advanced_final_dataset():
    logging.info("--- Starting ADVANCED Final Dataset Creation ---")

    # --- 1. Define Paths ---
    path_yield_data = Path("data/02_intermediate/sugarbeet_yield_area.csv")
    path_districts_geo = Path("data/01_raw/districts_official.geojson")
    path_weather_data = Path("data/02_intermediate/agera5_germany_2017_2024_merged.nc")
    output_dir = Path("data/03_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / "final_dataset_with_advanced_features.csv"

    if output_filepath.exists():
        logging.info(f"Advanced dataset at '{output_filepath}' already exists. Skipping.")
        return

    # --- 2. Load Data ---
    logging.info("Loading and preparing input files...")
    df_yield = pd.read_csv(path_yield_data)
    df_yield = df_yield[df_yield['year'].between(2017, 2024)].copy()
    df_yield['district_no'] = df_yield['district_no'].astype(str).str.zfill(5)

    gdf_districts = gpd.read_file(path_districts_geo)
    gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)

    ds_weather = xr.open_dataset(path_weather_data)
    ds_weather.rio.write_crs("EPSG:4326", inplace=True)
    # Convert temperatures to Celsius for all calculations
    ds_weather['Temperature_Air_2m_Mean_24h'] -= 273.15
    ds_weather['Temperature_Air_2m_Max_24h'] -= 273.15
    ds_weather['Temperature_Air_2m_Min_24h'] -= 273.15

    # --- 3. Perform Advanced Zonal Statistics ---
    logging.info("Starting advanced zonal statistics...")

    results = []
    for district in tqdm(gdf_districts.itertuples(), total=len(gdf_districts), desc="Processing Districts"):
        try:
            clipped_ds = ds_weather.rio.clip([district.geometry], gdf_districts.crs, drop=True)
            daily_mean_weather = clipped_ds.mean(dim=['lat', 'lon'])

            for year in range(2017, 2025):
                year_data = daily_mean_weather.sel(time=str(year))

                # Define growth periods
                early_growth = year_data.sel(time=year_data.time.dt.month.isin([5, 6]))
                peak_growth = year_data.sel(time=year_data.time.dt.month.isin([7, 8, 9]))

                # Calculate features
                precip_total_peak = peak_growth['Precipitation_Flux'].sum().values
                temp_mean_peak = peak_growth['Temperature_Air_2m_Mean_24h'].mean().values
                heat_stress_days = (peak_growth['Temperature_Air_2m_Max_24h'] > 30).sum().values

                results.append({
                    'district_no': district.district_no,
                    'year': year,
                    'precip_total_peak_growth': precip_total_peak,
                    'temp_mean_peak_growth': temp_mean_peak,
                    'heat_stress_days_peak_growth': heat_stress_days,
                    'temp_mean_early_growth': early_growth['Temperature_Air_2m_Mean_24h'].mean().values,
                    'solar_rad_peak_growth': peak_growth['Solar_Radiation_Flux'].mean().values,
                })

        except Exception as e:
            logging.warning(f"Could not process district {district.district_no}. Error: {e}")
            continue

    # --- 4. Assemble and Merge ---
    logging.info("Assembling final dataset...")
    df_weather_advanced = pd.DataFrame(results)

    final_df = pd.merge(df_yield, df_weather_advanced, on=['district_no', 'year'], how='left')
    final_df.dropna(subset=[col for col in final_df.columns if '_growth' in col], inplace=True)

    # --- 5. Save Final Dataset ---
    logging.info(f"Saving final dataset with {len(final_df)} rows to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False)

    logging.info("--- SUCCESS: Advanced dataset created! ---")
    print("\n--- Final Advanced Dataset Preview ---")
    print(final_df.head())


if __name__ == "__main__":
    create_advanced_final_dataset()