# File: src/features/create_spatial_features.py
# Description: This script generates spatial lag features by analyzing the relationships
#              between neighboring districts.

import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_FEATURES_FILE = Path('data/05_model_input/stage1_preseason_features.csv')
GEOJSON_FILE = Path('data/01_raw/districts_official.geojson')
OUTPUT_DIR = Path('data/05_model_input/')
OUTPUT_FILE = OUTPUT_DIR / 'spatial_features.csv'

# Define which features we want to create spatial lags for.
# Choosing a few powerful ones is better than choosing all of them.
FEATURES_TO_LAG = [
    'kreisYield',  # The most powerful feature: what did my neighbors yield LAST year?
    'summer_precip_anomaly_forecast',
    'summer_temp_anomaly_forecast',
    'avg_sand_0_30cm',
    'avg_clay_0_30cm',
    'profit_margin_proxy_lag1'
]


def create_neighbor_lookup(gdf: gpd.GeoDataFrame) -> dict:
    """
    Uses the GeoDataFrame to create a dictionary mapping each district to a list of its neighbors.
    """
    logging.info("Generating neighbor lookup using Queen contiguity...")
    # Queen contiguity means polygons are neighbors if they share an edge or a corner.
    weights = Queen.from_dataframe(gdf, idVariable='id')
    neighbors = weights.neighbors
    logging.info("✓ Neighbor lookup created successfully.")
    return neighbors


def generate_spatial_lag_features():
    """
    Main function to orchestrate the creation of spatial lag features.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading base feature set and geographic data...")
    df = pd.read_csv(INPUT_FEATURES_FILE)
    gdf = gpd.read_file(GEOJSON_FILE)

    # Ensure district_no types match for lookup
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    gdf['id'] = gdf['id'].astype(str).str.zfill(5)

    neighbors_dict = create_neighbor_lookup(gdf)

    # Set multi-index for fast lookups
    df.set_index(['district_no', 'year'], inplace=True)
    df.sort_index(inplace=True)

    all_lag_features = []

    logging.info("Calculating spatial lag features for each district-year...")
    # This loop is intensive, but it's a one-time process.
    for district, year in tqdm(df.index, desc="Generating Spatial Lags"):

        current_neighbors = neighbors_dict.get(district, [])
        if not current_neighbors:
            continue

        lag_record = {'district_no': district, 'year': year}

        for feature in FEATURES_TO_LAG:
            if feature == 'kreisYield':
                # For yield, we look at the neighbors' performance in the PREVIOUS year.
                # This is a causal feature.
                previous_year = year - 1
                neighbor_indices = [(n, previous_year) for n in current_neighbors if (n, previous_year) in df.index]
                if neighbor_indices:
                    neighbor_values = df.loc[neighbor_indices, feature]
                    lag_record[f'neighbor_avg_{feature}_lag1'] = neighbor_values.mean()
                    lag_record[f'neighbor_std_{feature}_lag1'] = neighbor_values.std()

            else:
                # For static or forecast features, we look at the neighbors in the CURRENT year.
                neighbor_indices = [(n, year) for n in current_neighbors if (n, year) in df.index]
                if neighbor_indices:
                    neighbor_values = df.loc[neighbor_indices, feature]
                    lag_record[f'neighbor_avg_{feature}'] = neighbor_values.mean()

        all_lag_features.append(lag_record)

    df_spatial = pd.DataFrame(all_lag_features)
    df_spatial.fillna(0, inplace=True)  # Fill std for single neighbors, etc.

    logging.info(f"Saving spatial features to {OUTPUT_FILE}...")
    df_spatial.to_csv(OUTPUT_FILE, index=False)
    logging.info("✓ Spatial feature generation complete.")


if __name__ == "__main__":
    generate_spatial_lag_features()