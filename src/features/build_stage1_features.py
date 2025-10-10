import pandas as pd
import xarray as xr
import logging
from pathlib import Path
from tqdm import tqdm
import time
import geopandas as gpd

# --- CONFIGURATION: SELECT THE ECONOMIC FEATURES TO INCLUDE ---
# To add or remove a feature from the final dataset, simply add or
# comment out its ID from this list.

ECONOMIC_FEATURES_TO_INCLUDE = [
    # --- Producer Price ---
    'LWPR-1',    # Pflanzliche Erzeugung
    'LWPR-132',  # Zuckerrüben (Sugar Beets)

    #'LWBM',  # Landwirtschaftliche Betriebsmittel (Total Agricultural Inputs)

    # --- Current Consumption ---
    #'LWBM-1',
    # Waren und Dienstleist. des lfd. landw. Verbrauchs (Goods and services for current agricultural consumption)
    'LWBM-11',  # Saat- und Pflanzgut (Seeds and Planting Material)
    'LWBM-12',  # Energie und Schmierstoffe (Energy and Lubricants)
    'LWBM-13',  # Düngemittel (Fertilizer)
    'LWBM-14',  # Pflanzenschutzmittel (Plant Protection Products)
]

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_economic_features(feature_ids_to_include):
    """
    Loads and prepares all economic data based on a provided list of feature IDs.
    - Dynamically extracts features based on the input list.
    - Preserves QUARTERLY granularity for all input prices.
    """
    logging.info(f"Loading and processing {len(feature_ids_to_include)} selected economic features...")

    # --- Producer Prices (Annual Data) ---
    df_producer_prices = pd.DataFrame()  # Start with an empty dataframe
    try:
        producer_price_file = 'data/01_raw/61211-0002_de/61211-0001_de.csv'
        df_prod = pd.read_csv(producer_price_file)

        # Filter for only the producer price IDs specified in the config list
        producer_prices_filtered = df_prod[df_prod['ID'].isin(feature_ids_to_include)].copy()

        if not producer_prices_filtered.empty:
            df_producer_prices = producer_prices_filtered.melt(
                id_vars=['ID', 'Description'],
                var_name='year',
                value_name='producer_price_index'  # Generic name, will be renamed
            )
            # Create a clean feature name and pivot
            df_producer_prices['feature_name'] = df_producer_prices['Description'].str.lower().str.replace(' ',
                                                                                                           '_').str.replace(
                '[^a-zA-Z0-9_]', '', regex=True)
            df_producer_prices = df_producer_prices.pivot_table(index='year', columns='feature_name',
                                                                values='producer_price_index')
            df_producer_prices.index = pd.to_numeric(df_producer_prices.index)

        logging.info(" -> Producer prices processed.")
    except Exception as e:
        logging.error(f"Could not process producer prices. Error: {e}")
        return None

    # --- Input Prices (Quarterly Data) ---
    df_input_prices = pd.DataFrame()  # Start with an empty dataframe
    try:
        input_price_file = 'data/01_raw/61211-0002_de/61221-0003_de.csv'
        df_in = pd.read_csv(input_price_file)

        # Filter for only the input price IDs specified in the config list
        df_inputs_filtered = df_in[df_in['ID'].isin(feature_ids_to_include)].copy()

        if not df_inputs_filtered.empty:
            df_melted = df_inputs_filtered.melt(
                id_vars=['ID', 'Description'],
                var_name='period',
                value_name='price_index'
            )
            df_melted['year'] = pd.to_numeric(df_melted['period'].str.split('/').str[1])
            df_melted['quarter'] = pd.to_numeric(df_melted['period'].str.split('/').str[0]).apply(
                lambda m: f'Q{(m - 1) // 3 + 1}')
            df_melted['feature_name'] = df_melted['Description'].str.lower().str.replace(' ', '_').str.replace(
                '[^a-zA-Z0-9_]', '', regex=True)

            df_quarterly = df_melted.pivot_table(
                index=['year', 'quarter'],
                columns='feature_name',
                values='price_index'
            ).reset_index()

            df_input_prices = df_quarterly.pivot(index='year', columns='quarter')
            df_input_prices.columns = [f'{val}_{quarter}' for val, quarter in df_input_prices.columns]

        logging.info(f" -> Input prices processed. Created {len(df_input_prices.columns)} quarterly features.")

    except Exception as e:
        logging.error(f"Could not process input prices. Error: {e}")
        return None

    # --- Combine all economic data ---
    df_economic = df_producer_prices.join(df_input_prices, how='outer')
    logging.info("All selected economic features processed successfully.")
    return df_economic

def engineer_antecedent_weather(df_base, agera5_path):
    """Calculates and merges antecedent winter weather anomalies."""
    logging.info("Starting antecedent winter weather feature engineering...")
    try:
        agera5_ds = xr.open_dataset(agera5_path)
        temp_var_name = 'Temperature_Air_2m_Mean_24h'
        precip_var_name = 'Precipitation_Flux'
    except Exception as e:
        logging.error(f"Could not load or validate AgERA5 dataset. Error: {e}")
        return None

    available_years = pd.to_datetime(agera5_ds.time.values).year.unique()
    climate_normal_period = (available_years.min(), available_years.max() - 1)
    logging.info(f"Using recent climate normal based on: {climate_normal_period[0]}-{climate_normal_period[1]}")

    logging.info("Step 1: Subsetting and loading climate normal data into memory...")
    start_time = time.time()
    normals_ds_subset = agera5_ds.sel(time=slice(str(climate_normal_period[0]), str(climate_normal_period[1])))
    normals_ds_subset.load()
    end_time = time.time()
    logging.info(f"Step 1 COMPLETE. Data loaded in {end_time - start_time:.2f} seconds.")

    logging.info("Step 2: Calculating normals...")
    normals_winter_filter = normals_ds_subset['time'].dt.month.isin([10, 11, 12, 1, 2, 3])
    normals_winter_ds = normals_ds_subset.where(normals_winter_filter, drop=True)
    seasonal_temp_mean = normals_winter_ds[temp_var_name].resample(time='YE-SEP').mean(dim='time')
    winter_temp_normal = seasonal_temp_mean.mean(dim='time')
    seasonal_precip_sum = normals_winter_ds[precip_var_name].resample(time='YE-SEP').sum(dim='time')
    winter_precip_normal = seasonal_precip_sum.mean(dim='time')
    logging.info("Step 2 COMPLETE. Climate normals calculated.")

    logging.info("Loading district coordinates from GeoJSON file...")
    try:
        geojson_path = 'data/01_raw/districts_official.geojson'
        gdf = gpd.read_file(geojson_path)
        gdf['geometry'] = gdf['geometry'].to_crs('+proj=cea')
        gdf['latitude'] = gdf['geometry'].centroid.to_crs(epsg=4326).y
        gdf['longitude'] = gdf['geometry'].centroid.to_crs(epsg=4326).x
        districts_df = gdf[['id', 'latitude', 'longitude']].rename(columns={'id': 'district_no'})
        districts_df['district_no'] = districts_df['district_no'].astype(int)
    except Exception as e:
        logging.error(f"FATAL: Could not process GeoJSON file. Error: {e}")
        return None

    logging.info("Step 3: Calculating yearly anomalies...")
    all_years_features = []
    for year in tqdm(df_base['year'].unique(), desc="Processing Years"):
        start_date, end_date = f"{year - 1}-10-01", f"{year}-03-31"
        try:
            winter_period_ds = agera5_ds.sel(time=slice(start_date, end_date))
            if winter_period_ds.time.size == 0: continue
        except KeyError:
            continue
        actual_winter_temp = winter_period_ds[temp_var_name].mean(dim='time')
        actual_winter_precip = winter_period_ds[precip_var_name].sum(dim='time')
        temp_anomaly = actual_winter_temp - winter_temp_normal
        precip_anomaly = (actual_winter_precip / winter_precip_normal) - 1.0
        for _, row in districts_df.iterrows():
            district_no, lat, lon = row['district_no'], row['latitude'], row['longitude']
            all_years_features.append({
                'year': year, 'district_no': district_no,
                'winter_temp_anomaly': temp_anomaly.interp(lat=lat, lon=lon, method="linear").item(),
                'winter_precip_anomaly': precip_anomaly.interp(lat=lat, lon=lon, method="linear").item()
            })
    df_weather_features = pd.DataFrame(all_years_features)
    logging.info("Antecedent weather features calculated.")
    df_merged = pd.merge(df_base, df_weather_features, on=['year', 'district_no'], how='left')
    return df_merged

def engineer_seasonal_forecasts(df_base, districts_df, hindcast_dir):
    """
    Calculates seasonal forecast anomalies from hindcast and climatology files.
    DEFINITIVE VERSION: Correctly handles all dimensions ('number', 'forecastMonth',
    and 'forecast_reference_time') to produce a single ensemble-mean anomaly.
    """
    logging.info("Starting seasonal forecast feature engineering...")

    try:
        temp_climatology = xr.open_dataset(hindcast_dir / "seas5_climatology_germany_1993-2016_2m_temperature.nc")
        precip_climatology = xr.open_dataset(
            hindcast_dir / "seas5_climatology_germany_1993-2016_total_precipitation.nc")
    except FileNotFoundError as e:
        logging.error(f"FATAL: Climatology file not found. {e}")
        return None

    all_forecast_features = []

    for year in tqdm(df_base['year'].unique(), desc="Processing SEAS5 Forecasts"):
        try:
            temp_hindcast = xr.open_dataset(hindcast_dir / f"seas5_hindcast_germany_{year}_2m_temperature.nc")
            precip_hindcast = xr.open_dataset(hindcast_dir / f"seas5_hindcast_germany_{year}_total_precipitation.nc")
        except FileNotFoundError:
            logging.warning(f"  -> Hindcast files for year {year} not found. Skipping.")
            continue

        days_in_season = 6 * 30.4

        # --- DEFINITIVE FIX APPLIED HERE ---
        # Reduce both hindcast and climatology to clean 2D (lat, lon) grids before subtracting.

        # For Hindcast: average over season and ensemble members, then remove the single-entry time dimension.
        avg_temp_hindcast = temp_hindcast['t2m'].mean(dim=['forecastMonth', 'number']).squeeze(drop=True)
        total_precip_hindcast = (precip_hindcast['tprate'].sum(dim='forecastMonth') * days_in_season).mean(
            dim='number').squeeze(drop=True)

        # For Climatology: average over season, ensemble members, AND all reference years.
        avg_temp_climatology = temp_climatology['t2m'].mean(dim=['forecastMonth', 'number', 'forecast_reference_time'])
        total_precip_climatology = (precip_climatology['tprate'].sum(dim='forecastMonth') * days_in_season).mean(
            dim=['number', 'forecast_reference_time'])

        # Now, the anomalies are guaranteed to be simple 2D grids (latitude, longitude).
        forecasted_temp_anomaly = avg_temp_hindcast - avg_temp_climatology
        forecasted_precip_anomaly = total_precip_hindcast - total_precip_climatology

        # Spatially aggregate for each district
        for _, row in districts_df.iterrows():
            district_no, lat, lon = row['district_no'], row['latitude'], row['longitude']

            # The .item() call will now work correctly.
            temp_val = forecasted_temp_anomaly.interp(latitude=lat, longitude=lon, method="linear").item()
            precip_val = forecasted_precip_anomaly.interp(latitude=lat, longitude=lon, method="linear").item()

            all_forecast_features.append({
                'year': year,
                'district_no': district_no,
                'forecasted_temp_anomaly': temp_val,
                'forecasted_precip_anomaly': precip_val
            })

    return pd.merge(df_base, pd.DataFrame(all_forecast_features), on=['year', 'district_no'], how='left')

def build_stage1_features():
    """Main orchestrator function."""
    logging.info("--- Starting Final Stage 1 Feature Build Process ---")

    # --- Define Paths ---
    model_input_file = 'data/05_model_input/model_input.csv'
    agera5_file = 'data/02_intermediate/agera5_germany_2017_2024_merged.nc'
    hindcast_dir = Path("data/01_raw/SEAS5_hindcasts")
    output_path = Path('data/05_model_input/')
    output_file = output_path / 'stage1_final_training_data.csv'  # New final name
    output_path.mkdir(exist_ok=True)

    # --- Step 1: Base Data + Lagged Economics ---
    logging.info("\n--- Step 1: Loading base data and lagging economic features ---")
    df_base = pd.read_csv(model_input_file)[['district_no', 'year', 'yield', 'avg_elevation', 'avg_soil_pawc']]
    df_economic = get_economic_features(ECONOMIC_FEATURES_TO_INCLUDE)
    df_economic_lagged = df_economic.shift(1).rename(columns=lambda c: f"{c}_lag1")
    df_with_eco = pd.merge(df_base, df_economic_lagged, on='year', how='left')
    logging.info(f" -> Shape after adding economic features: {df_with_eco.shape}")

    # --- Step 2: Antecedent Winter Weather ---
    logging.info("\n--- Step 2: Engineering antecedent winter weather features ---")
    df_with_winter = engineer_antecedent_weather(df_with_eco, agera5_file)
    if df_with_winter is None: return
    logging.info(f" -> Shape after adding winter features: {df_with_winter.shape}")

    # --- Step 3: Seasonal Forecasts (NEW BLOCK) ---
    logging.info("\n--- Step 3: Engineering seasonal forecast features ---")
    # We need the districts_df with coordinates for this step, so we load it here.
    try:
        geojson_path = 'data/01_raw/districts_official.geojson'
        gdf = gpd.read_file(geojson_path)
        gdf['geometry'] = gdf['geometry'].to_crs('+proj=cea')
        gdf['latitude'] = gdf['geometry'].centroid.to_crs(epsg=4326).y
        gdf['longitude'] = gdf['geometry'].centroid.to_crs(epsg=4326).x
        districts_df = gdf[['id', 'latitude', 'longitude']].rename(columns={'id': 'district_no'})
        districts_df['district_no'] = districts_df['district_no'].astype(int)
    except Exception as e:
        logging.error(f"FATAL: Could not process GeoJSON for Step 3. Error: {e}")
        return

    df_final = engineer_seasonal_forecasts(df_with_winter, districts_df, hindcast_dir)
    if df_final is None: return
    logging.info(f" -> Shape after adding forecast features: {df_final.shape}")

    # --- Final Save and Report ---
    df_final.to_csv(output_file, index=False)
    logging.info(f"\n--- SUCCESS: Final Stage 1 training data created! ---")
    logging.info(f"Dataset saved to '{output_file}'")
    logging.info(f"Final dataset columns: {df_final.columns.tolist()}")
    final_missing = df_final.isnull().sum()
    logging.info(f"\nMissing values in final dataset:\n{final_missing[final_missing > 0]}")

if __name__ == '__main__':
    build_stage1_features()