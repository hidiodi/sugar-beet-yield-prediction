# File: src/models/create_final_hybrid_backtest.py
# Description: Performs a definitive, nested walk-forward validation for the full
#              two-stage hybrid model (Time-Series -> Spatial XGBoost Residual).
#
# REVISED VERSION v3: Corrects the data context issue during spatial feature generation
#                     to resolve the KeyError and ensure the backtest is sound.

import pandas as pd
import geopandas as gpd
import joblib
import warnings
from libpysal.weights import Queen
import logging
from pathlib import Path
from tqdm import tqdm
from pygam import LinearGAM, s
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# --- Configuration ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Input Data & Models ---
BASE_FEATURES_FILE = Path('data/05_model_input/stage1_preseason_features.csv')
GEOJSON_FILE = Path('data/01_raw/districts_official.geojson')
MODEL_SKELETON_DIR = Path('src/models/spatial_champion')

# --- Output Location ---
OUTPUT_DIR = Path('reports/figures/district_level_diagnostics/final_hybrid_champion')
OUTPUT_FILE = OUTPUT_DIR / 'full_backtest_predictions_hybrid.csv'

# --- Backtest Configuration ---
BACKTEST_START_YEAR = 2000
MIN_HISTORY_FOR_TS = 10

# --- Feature Definitions ---
BASE_FEATURE_COLS = [
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
    'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
    'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
    'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast', 'spring_temp_prob_warm_forecast',
    'spring_precip_prob_wet_forecast', 'spring_solar_rad_prob_wet_forecast', 'spring_evaporation_prob_wet_forecast',
    'spring_runoff_prob_wet_forecast', 'spring_soil_temp_l1_prob_warm_forecast', 'spring_snowfall_prob_wet_forecast',
    'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast', 'summer_solar_rad_prob_wet_forecast',
    'summer_evaporation_prob_wet_forecast', 'summer_runoff_prob_wet_forecast', 'summer_soil_temp_l1_prob_warm_forecast',
    'summer_snowfall_prob_wet_forecast', 'lat', 'lon', 'avg_elevation', 'avg_slope', 'avg_bdod_0_30cm',
    'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'avg_bdod_0_100cm', 'avg_clay_0_100cm',
    'avg_sand_0_100cm', 'avg_som_0_100cm', 'avg_phh2o_0_100cm', 'winter_cropland_ndvi_mean',
    'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
    'winter_cropland_snow_cover_days', 'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',
    'profit_margin_proxy_lag1', 'cost_of_inputs_lag1', 'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip',
    'antecedent_gdd_sum_anomaly_sq', 'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
    'spring_temp_prob_warm_forecast_sq', 'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq',
    'summer_precip_prob_wet_forecast_sq', 'state6_precip_interaction', 'is_drought_high_clay_in_state_11'
]
NEW_SPATIAL_FEATURES = [
    'neighbor_avg_kreisYield_lag1', 'neighbor_std_kreisYield_lag1', 'neighbor_avg_summer_precip_anomaly_forecast',
    'neighbor_avg_summer_temp_anomaly_forecast', 'neighbor_avg_avg_sand_0_30cm', 'neighbor_avg_avg_clay_0_30cm',
    'neighbor_avg_profit_margin_proxy_lag1'
]
FINAL_FEATURE_COLS = BASE_FEATURE_COLS + NEW_SPATIAL_FEATURES


def generate_spatial_features(df, neighbors_dict):
    df_indexed = df.set_index(['district_no', 'year']).sort_index()
    features_to_lag = ['kreisYield', 'summer_precip_anomaly_forecast', 'summer_temp_anomaly_forecast',
                       'avg_sand_0_30cm', 'avg_clay_0_30cm', 'profit_margin_proxy_lag1']
    all_lag_records = []
    # Loop through the index of the PASSED dataframe, which may contain multiple years
    for district, year in df_indexed.index:
        current_neighbors = neighbors_dict.get(district, []);
        if not current_neighbors: continue
        lag_record = {'district_no': district, 'year': year}
        for feature in features_to_lag:
            if feature == 'kreisYield':
                previous_year = year - 1
                # Check for neighbors in the previous year WITHIN THE SAME DATAFRAME
                neighbor_indices = [(n, previous_year) for n in current_neighbors if
                                    (n, previous_year) in df_indexed.index]
                if neighbor_indices:
                    neighbor_values = df_indexed.loc[neighbor_indices, feature];
                    lag_record['neighbor_avg_kreisYield_lag1'] = neighbor_values.mean();
                    lag_record['neighbor_std_kreisYield_lag1'] = neighbor_values.std()
            else:
                neighbor_indices = [(n, year) for n in current_neighbors if (n, year) in df_indexed.index]
                if neighbor_indices:
                    neighbor_values = df_indexed.loc[neighbor_indices, feature];
                    lag_record[f'neighbor_avg_{feature}'] = neighbor_values.mean()
        all_lag_records.append(lag_record)
    df_spatial = pd.DataFrame(all_lag_records).fillna(0)
    return pd.merge(df, df_spatial, on=['district_no', 'year'], how='left')


def create_hybrid_backtest():
    logging.info("--- Starting Definitive Nested Walk-Forward Backtest ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load base data and neighbor lookup ONCE
    df_full = pd.read_csv(BASE_FEATURES_FILE);
    gdf = gpd.read_file(GEOJSON_FILE)
    df_full['district_no'] = df_full['district_no'].astype(str).str.zfill(5);
    gdf['id'] = gdf['id'].astype(str).str.zfill(5)
    weights = Queen.from_dataframe(gdf, idVariable='id');
    neighbors_dict = weights.neighbors

    models = {name: joblib.load(MODEL_SKELETON_DIR / f'final_spatial_quantile_model_{name}.joblib') for name in
              ['lower', 'median', 'upper']}

    all_yearly_predictions = []

    end_year = df_full['year'].max()
    for current_year in tqdm(range(BACKTEST_START_YEAR, end_year + 1), desc="Backtesting Year-by-Year"):

        historic_df_base = df_full[df_full['year'] < current_year].copy()
        predict_df_base = df_full[df_full['year'] == current_year].copy()

        # --- Stage 1: Generate Time-Series Forecast for `current_year` ---
        stage1_forecasts = []
        for district in predict_df_base['district_no'].unique():
            district_history = historic_df_base[historic_df_base['district_no'] == district]
            if len(district_history) < MIN_HISTORY_FOR_TS: continue

            x_train, y_train = district_history['year'].values, district_history['kreisYield'].values
            try:
                trend_model = LinearGAM(s(0, n_splines=10)).fit(x_train, y_train)
                base_trend_forecast = trend_model.predict([current_year])[0]
                historical_residuals = y_train - trend_model.predict(x_train)
                residual_model = ARIMA(historical_residuals, order=(1, 0, 0)).fit()
                residual_forecast = residual_model.forecast(steps=1)[0]
                stage1_forecast = base_trend_forecast + residual_forecast
                if np.isfinite(stage1_forecast):
                    stage1_forecasts.append(
                        {'district_no': district, 'year': current_year, 'stage1_forecast': stage1_forecast})
            except Exception:
                continue

        if not stage1_forecasts: continue
        predict_df_base = pd.merge(predict_df_base, pd.DataFrame(stage1_forecasts), on=['district_no', 'year'],
                                   how='inner')
        if predict_df_base.empty: continue

        # --- Stage 2: Train XGBoost and Predict Residuals ---
        # A) Use pre-calculated honest forecasts for the training data's residuals
        historic_df_base.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast_hist'}, inplace=True)
        historic_df_base.dropna(subset=['stage1_forecast_hist'], inplace=True)
        historic_df_base['forecast_residual'] = historic_df_base['kreisYield'] - historic_df_base[
            'stage1_forecast_hist']

        # B) --- CORRECTED LOGIC ---
        # Combine data before generating spatial features to provide context
        loop_data = pd.concat([historic_df_base, predict_df_base])
        loop_data_spatial = generate_spatial_features(loop_data, neighbors_dict)

        # C) Re-split into final train and predict sets
        train_df_spatial = loop_data_spatial[loop_data_spatial['year'] < current_year].copy()
        predict_df_spatial = loop_data_spatial[loop_data_spatial['year'] == current_year].copy()

        # D) Prepare final training and prediction dataframes
        train_df_spatial.dropna(subset=FINAL_FEATURE_COLS, inplace=True)
        predict_df_spatial.dropna(subset=FINAL_FEATURE_COLS, inplace=True)

        X_train = train_df_spatial[FINAL_FEATURE_COLS]
        y_train = train_df_spatial['forecast_residual']
        X_predict = predict_df_spatial[FINAL_FEATURE_COLS]

        if X_train.empty or X_predict.empty: continue

        # E) Train and predict for each quantile
        for name, model in models.items():
            model.fit(X_train, y_train)
            predict_df_spatial[f'predicted_residual_{name}'] = model.predict(X_predict)
            predict_df_spatial[f'predicted_yield_{name}'] = predict_df_spatial['stage1_forecast'] + predict_df_spatial[
                f'predicted_residual_{name}']

        all_yearly_predictions.append(predict_df_spatial)

    if not all_yearly_predictions:
        logging.error("FATAL: The backtest loop produced no predictions.");
        return

    final_backtest_df = pd.concat(all_yearly_predictions, ignore_index=True)

    output_cols = ['district_no', 'year', 'kreisYield', 'stage1_forecast'] + [c for c in final_backtest_df if
                                                                              'predicted_yield' in c]
    final_backtest_df[output_cols].to_csv(OUTPUT_FILE, index=False)
    logging.info(f"✓ Definitive hybrid model backtest saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_hybrid_backtest()