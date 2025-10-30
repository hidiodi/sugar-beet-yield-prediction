# File: src/models/train_final_spatial_model.py
# Description: This single, self-contained script trains a spatially-aware version of the final model.
#              It loads the standard feature set, generates spatial lag features on-the-fly,
#              and trains a new XGBoost model on the enriched dataset.

import pandas as pd
import geopandas as gpd
from xgboost import XGBRegressor
import os
import joblib
import warnings
from libpysal.weights import Queen
import logging
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Input Data ---
# This is the output from your existing, non-spatial feature pipeline
BASE_FEATURES_FILE = Path('data/05_model_input/stage1_preseason_features.csv')
GEOJSON_FILE = Path('data/01_raw/districts_official.geojson')

# --- Output Location for the new SPATIAL models ---
MODEL_OUTPUT_DIR = Path('src/models/spatial_champion')

# --- Feature Definitions ---
# The original feature set from the non-spatial model
BASE_FEATURE_COLS = [
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly',
    'antecedent_gdd_sum_anomaly', 'spring_temp_anomaly_forecast',
    'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
    'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast',
    'spring_soil_temp_l1_anomaly_forecast', 'spring_snowfall_anomaly_forecast',
    'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast',
    'summer_runoff_anomaly_forecast', 'summer_soil_temp_l1_anomaly_forecast',
    'summer_snowfall_anomaly_forecast', 'spring_temp_prob_warm_forecast',
    'spring_precip_prob_wet_forecast', 'spring_solar_rad_prob_wet_forecast',
    'spring_evaporation_prob_wet_forecast', 'spring_runoff_prob_wet_forecast',
    'spring_soil_temp_l1_prob_warm_forecast', 'spring_snowfall_prob_wet_forecast',
    'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',
    'summer_solar_rad_prob_wet_forecast', 'summer_evaporation_prob_wet_forecast',
    'summer_runoff_prob_wet_forecast', 'summer_soil_temp_l1_prob_warm_forecast',
    'summer_snowfall_prob_wet_forecast', 'lat', 'lon', 'avg_elevation',
    'avg_slope', 'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm',
    'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'avg_bdod_0_100cm',
    'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm',
    'avg_phh2o_0_100cm', 'winter_cropland_ndvi_mean',
    'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean',
    'winter_cropland_LST_anomaly', 'winter_cropland_snow_cover_days',
    'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',
    'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
    'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip',
    'antecedent_gdd_sum_anomaly_sq', 'summer_heat_x_profit_margin',
    'summer_precip_x_input_costs', 'spring_temp_prob_warm_forecast_sq',
    'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq',
    'summer_precip_prob_wet_forecast_sq', 'state6_precip_interaction',
    'is_drought_high_clay_in_state_11'
]

# The new spatial features we will generate and add
NEW_SPATIAL_FEATURES = [
    'neighbor_avg_kreisYield_lag1',
    'neighbor_std_kreisYield_lag1',
    'neighbor_avg_summer_precip_anomaly_forecast',
    'neighbor_avg_summer_temp_anomaly_forecast',
    'neighbor_avg_avg_sand_0_30cm',
    'neighbor_avg_avg_clay_0_30cm',
    'neighbor_avg_profit_margin_proxy_lag1'
]

# The full, final feature set for this spatial model
FINAL_FEATURE_COLS = BASE_FEATURE_COLS + NEW_SPATIAL_FEATURES

# Hyperparameters and Quantiles
BEST_PARAMS = {
    'n_estimators': 914, 'learning_rate': 0.026114, 'max_depth': 5,
    'subsample': 0.922850, 'colsample_bytree': 0.811573, 'gamma': 1.830853,
    'min_child_weight': 2, 'random_state': 42, 'n_jobs': -1
}
QUANTILES = {'lower': 0.025, 'median': 0.5, 'upper': 0.975}


def generate_spatial_features_on_the_fly(df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Generates and merges spatial lag features into the provided dataframe.
    """
    logging.info("--- Generating Spatial Lag Features On-the-Fly ---")

    weights = Queen.from_dataframe(gdf, idVariable='id')
    neighbors_dict = weights.neighbors

    df_indexed = df.set_index(['district_no', 'year']).sort_index()

    features_to_lag = [
        'kreisYield', 'summer_precip_anomaly_forecast', 'summer_temp_anomaly_forecast',
        'avg_sand_0_30cm', 'avg_clay_0_30cm', 'profit_margin_proxy_lag1'
    ]

    all_lag_records = []
    for district, year in tqdm(df_indexed.index, desc="Generating Spatial Lags"):
        current_neighbors = neighbors_dict.get(district, [])
        if not current_neighbors: continue

        lag_record = {'district_no': district, 'year': year}
        for feature in features_to_lag:
            if feature == 'kreisYield':
                previous_year = year - 1
                neighbor_indices = [(n, previous_year) for n in current_neighbors if
                                    (n, previous_year) in df_indexed.index]
                if neighbor_indices:
                    neighbor_values = df_indexed.loc[neighbor_indices, feature]
                    lag_record['neighbor_avg_kreisYield_lag1'] = neighbor_values.mean()
                    lag_record['neighbor_std_kreisYield_lag1'] = neighbor_values.std()
            else:
                neighbor_indices = [(n, year) for n in current_neighbors if (n, year) in df_indexed.index]
                if neighbor_indices:
                    neighbor_values = df_indexed.loc[neighbor_indices, feature]
                    lag_record[f'neighbor_avg_{feature}'] = neighbor_values.mean()

        all_lag_records.append(lag_record)

    df_spatial = pd.DataFrame(all_lag_records).fillna(0)

    # Merge the new spatial features back into the original dataframe
    df_enriched = pd.merge(df, df_spatial, on=['district_no', 'year'], how='left')
    logging.info("✓ Spatial features generated and merged.")
    return df_enriched


def train_final_spatial_model():
    """Main function to load, enrich with spatial features, and train the model."""
    logging.info("--- Starting SPATIAL Champion Model Training Pipeline ---")

    try:
        df = pd.read_csv(BASE_FEATURES_FILE)
        gdf = gpd.read_file(GEOJSON_FILE)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        gdf['id'] = gdf['id'].astype(str).str.zfill(5)
    except FileNotFoundError as e:
        logging.error(f"❌ Error: A required data file was not found. Details: {e}")
        return

    # 1. Generate and merge spatial features
    df_spatial = generate_spatial_features_on_the_fly(df, gdf)

    # 2. Define the Target Variable (Residual Fitting)
    logging.info("\n--- Calculating Forecast Residuals ---")
    df_spatial.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df_spatial['forecast_residual'] = df_spatial['kreisYield'] - df_spatial['stage1_forecast']

    # Clean up data: drop rows where forecast or features are missing
    df_spatial.dropna(subset=['stage1_forecast', 'forecast_residual'], inplace=True)
    df_spatial.dropna(subset=FINAL_FEATURE_COLS, inplace=True)

    logging.info(" -> Target variable (forecast_residual) created.")

    # 3. Train and Save Quantile Models
    X_train = df_spatial[FINAL_FEATURE_COLS]
    y_train = df_spatial['forecast_residual']
    logging.info(f"\nTraining on {len(X_train)} samples with {len(FINAL_FEATURE_COLS)} features.")

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, alpha in QUANTILES.items():
        logging.info(f"\n--- Training SPATIAL {name.upper()} Residual Model (Quantile: {alpha}) ---")

        model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, **BEST_PARAMS)
        model.fit(X_train, y_train)

        model_path = MODEL_OUTPUT_DIR / f'final_spatial_quantile_model_{name}.joblib'
        joblib.dump(model, model_path)
        logging.info(f"✅ SPATIAL {name.upper()} model saved to {model_path}")

    logging.info("\n--- All SPATIAL Models Trained and Saved Successfully ---")


if __name__ == "__main__":
    train_final_spatial_model()