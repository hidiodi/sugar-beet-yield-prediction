import pandas as pd
import geopandas as gpd
import os
import logging
from pathlib import Path

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Refined the list to be highly specific to sugar beets, removing the
# broad 'Pflanzliche Erzeugung' (LWPR-1) to reduce noise.
ECONOMIC_FEATURES_TO_INCLUDE = [
    # --- Producer Price (Specific to Sugar Beets) ---
    'LWPR-132',  # Zuckerrüben (Sugar Beets)

    # --- Key Input Costs (The "Four Pillars") ---
    'LWBM-11',  # Saat- und Pflanzgut (Seeds and Planting Material)
    'LWBM-12',  # Energie und Schmierstoffe (Energy and Lubricants)
    'LWBM-13',  # Düngemittel (Fertilizer)
    'LWBM-14',  # Pflanzenschutzmittel (Plant Protection Products)
]


# =======================================================================


def process_economic_features(producer_price_file, input_price_file, feature_ids_to_include):
    """
    Loads, processes, and prepares all specified economic data from raw sources,
    calculating annual averages for input prices.
    """
    logging.info(f"Processing {len(feature_ids_to_include)} selected economic features from raw files...")

    # --- Producer Prices (Annual Data) ---
    df_producer_prices = pd.DataFrame()
    try:
        df_prod = pd.read_csv(producer_price_file)
        # Filter for only the producer price IDs specified in the config list
        producer_prices_filtered = df_prod[df_prod['ID'].isin(feature_ids_to_include)].copy()

        if not producer_prices_filtered.empty:
            df_melted_prod = producer_prices_filtered.melt(
                id_vars=['ID', 'Description'],
                var_name='year',
                value_name='price_index'
            )
            # Create a clean feature name (e.g., 'zuckerrben')
            df_melted_prod['feature_name'] = df_melted_prod['Description'].str.lower().str.replace(' ',
                                                                                                   '_').str.replace(
                '[^a-zA-Z0-9_]', '', regex=True)
            df_producer_prices = df_melted_prod.pivot_table(index='year', columns='feature_name',
                                                            values='price_index')
            df_producer_prices.index = pd.to_numeric(df_producer_prices.index)
        logging.info(" -> Producer prices processed.")
    except Exception as e:
        logging.error(f"Could not process producer prices. Error: {e}")
        # Return an empty dataframe with a 'year' index to prevent merge errors
        return pd.DataFrame(index=pd.Index([], name='year'))

    # --- Input Prices (to be averaged annually) ---
    df_input_prices_final = pd.DataFrame()
    try:
        df_in = pd.read_csv(input_price_file)
        df_inputs_filtered = df_in[df_in['ID'].isin(feature_ids_to_include)].copy()

        if not df_inputs_filtered.empty:
            df_melted = df_inputs_filtered.melt(
                id_vars=['ID', 'Description'],
                var_name='period',
                value_name='price_index'
            )
            df_melted['feature_name'] = df_melted['Description'].str.lower().str.replace(' ', '_').str.replace(
                '[^a-zA-Z0-9_]', '', regex=True)

            df_melted['price_index'] = pd.to_numeric(df_melted['price_index'], errors='coerce')

            # Extract year from 'MM/YYYY' format
            df_melted['year'] = pd.to_numeric(df_melted['period'].str.split('/').str[1], errors='coerce')
            df_melted.dropna(subset=['year', 'price_index'], inplace=True)
            df_melted['year'] = df_melted['year'].astype(int)
            # Calculate the annual average from quarterly data
            df_annual_avg = df_melted.groupby(['year', 'feature_name'])['price_index'].mean().reset_index()
            df_input_prices_final = df_annual_avg.pivot(index='year', columns='feature_name', values='price_index')
        logging.info(" -> Input prices processed and averaged annually.")
    except Exception as e:
        logging.error(f"Could not process input prices. Error: {e}")
        # Return an empty dataframe with a 'year' index
        return pd.DataFrame(index=pd.Index([], name='year'))

    # --- Combine all economic data ---
    df_economic = df_producer_prices.join(df_input_prices_final, how='outer')
    logging.info("All selected economic features processed successfully.")
    return df_economic.reset_index()


def process_district_geography(geojson_path):
    """
    Loads the districts GeoJSON and creates a simple lookup table with
    ID, state, and centroid coordinates.
    """
    logging.info("Processing district geography data from GeoJSON...")
    try:
        gdf = gpd.read_file(geojson_path)
        gdf.rename(columns={'id': 'district_no', 'state': 'state_name'}, inplace=True)

        # Ensure CRS is set, then calculate centroids
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf_proj = gdf.to_crs('+proj=cea')  # Use an equal-area projection for accurate centroids
        centroids = gdf_proj['geometry'].centroid.to_crs(epsg=4326)  # Convert back to lat/lon

        gdf['latitude'] = centroids.y
        gdf['longitude'] = centroids.x

        # Select and clean final columns
        districts_df = gdf[['district_no', 'state_name', 'latitude', 'longitude']].copy()
        districts_df['district_no'] = districts_df['district_no'].astype(str).str.zfill(5)

        logging.info(" -> District geography lookup created successfully.")
        return districts_df
    except Exception as e:
        logging.error(f"FATAL: Could not process GeoJSON file. Error: {e}")
        return None


def main():
    """Main function to orchestrate the data merging and saving."""
    logging.info("--- Starting Comprehensive Master Dataset Creation ---")

    # --- Define File Paths ---
    base_data_file = Path('data/03_processed/final_dataset_with_advanced_features.csv')
    static_features_file = Path('data/03_processed/static_features_districts.csv')
    producer_price_file = Path('data/01_raw/Bundesdatenbank/61211-0001_de.csv')
    input_price_file = Path('data/01_raw/Bundesdatenbank/61221-0003_de.csv')
    geojson_path = Path('data/01_raw/districts_official.geojson')

    output_path = Path('data/04_master/')
    output_file = output_path / 'master_dataset.csv'
    output_path.mkdir(exist_ok=True)

    # --- Step 1: Load Base and Static Data ---
    logging.info("Loading base yield/weather and static soil/elevation data...")
    try:
        base_df = pd.read_csv(base_data_file)
        base_df['district_no'] = base_df['district_no'].astype(str).str.zfill(5)

        static_features = pd.read_csv(static_features_file)
        static_features['district_no'] = static_features['district_no'].astype(str).str.zfill(5)
    except FileNotFoundError as e:
        logging.error(f"A required input file was not found. Please check paths. Error: {e}")
        return

    # --- Step 2: Process Geography and Economic Data ---
    districts_df = process_district_geography(geojson_path)
    if districts_df is None: return

    df_economic = process_economic_features(producer_price_file, input_price_file, ECONOMIC_FEATURES_TO_INCLUDE)
    if df_economic.empty:
        logging.warning("Economic features dataframe is empty. Continuing without it.")

    # --- Step 3: Merge All Datasets into Master Table ---
    logging.info("\nMerging all datasets into a master table...")
    # Start with the base yield/weather data
    master_df = base_df.copy()

    # Merge geographic data
    master_df = pd.merge(master_df, districts_df, on='district_no', how='left')

    # Merge static features (soil, elevation)
    master_df = pd.merge(master_df, static_features, on='district_no', how='left')

    # Merge economic data on 'year'
    if not df_economic.empty:
        master_df = pd.merge(master_df, df_economic, on='year', how='left')

    logging.info("-> Merging complete.")

    # --- Step 4: Save Final Dataset ---
    master_df.to_csv(output_file, index=False)
    logging.info(f"\nMaster dataset successfully created and saved to: {output_file}")
    logging.info(f"Master dataset has {master_df.shape[0]} rows and {master_df.shape[1]} columns.")
    logging.info(f"Columns: {master_df.columns.tolist()}")


if __name__ == '__main__':
    main()
