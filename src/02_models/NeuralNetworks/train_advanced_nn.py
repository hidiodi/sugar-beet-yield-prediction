# File: train_champion_nn.py
# Description: FINAL CHAMPION NN MODEL. Implements the robust "predict the anomaly"
# strategy. It uses an extrapolated trend as a baseline and trains the NN to predict
# the deviation from that trend, using only non-trend features.

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# --- Configuration ---
CATEGORICAL_FEATURES = ['district_no']
# CRITICAL: We EXCLUDE trend-like features like 'national_avg_yield_lag1'
anomaly_features = [
    # Static & Geographic
    'avg_elevation', 'avg_soil_pawc', 'lon', 'lat',
    # Lagged Economic ANOMALIES
    'producer_price_index_lag1_anomaly', 'seed_price_index_lag1_anomaly',
    'energy_price_index_lag1_anomaly', 'fertilizer_price_index_lag1_anomaly',
    'plant_protection_price_index_lag1_anomaly',
    # Satellite
    'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly',
    'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
    'winter_cropland_snow_cover_days',
    # Weather Anomalies, Interactions, and Polynomials
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'temp_mean_mar_anomaly', 'precip_sum_mar_anomaly', 'srad_mean_mar_anomaly',
    'temp_mean_apr_anomaly', 'precip_sum_apr_anomaly', 'srad_mean_apr_anomaly',
    'temp_mean_may_anomaly', 'precip_sum_may_anomaly', 'srad_mean_may_anomaly',
    'temp_mean_jun_anomaly', 'precip_sum_jun_anomaly', 'srad_mean_jun_anomaly',
    'temp_mean_jul_anomaly', 'precip_sum_jul_anomaly', 'srad_mean_jul_anomaly',
    'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
    'july_heat_x_profit_margin', 'june_precip_x_input_costs',
    'temp_mean_jul_anomaly_sq', 'temp_mean_jun_anomaly_sq',
    'precip_sum_jul_anomaly_sq', 'srad_mean_jul_anomaly_sq'
]
CONTINUOUS_FEATURES = anomaly_features  # Set continuous features to our curated list
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_PATH = os.path.join('src/models', 'final_nn_model_champion.keras')


def build_advanced_nn_model(n_districts, n_continuous_features):
    """Defines the NN architecture."""
    categorical_input = layers.Input(shape=(len(CATEGORICAL_FEATURES),), name='categorical_input')
    continuous_input = layers.Input(shape=(n_continuous_features,), name='continuous_input')

    embedding_dim = int(np.sqrt(n_districts))
    embedding = layers.Embedding(input_dim=n_districts + 1, output_dim=embedding_dim, name='embedding')(
        categorical_input)
    embedding_flat = layers.Flatten(name='embedding_flat')(embedding)

    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(continuous_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    merged = layers.concatenate([embedding_flat, x], name='concatenate')
    final = layers.Dense(32, activation='relu')(merged)
    output = layers.Dense(1, name='output')(final)

    model = keras.Model(inputs=[categorical_input, continuous_input], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_and_validate_nn_champion():
    """
    Implements the robust "predict the anomaly" strategy for the NN model.
    """
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(by=['district_no', 'year'])

    validation_start_year = df['year'].max() - 5

    # === CRITICAL CHANGE 1: CALCULATE THE EXTRAPOLATED TREND ===
    print("--- Extrapolating yield trend using only past data (no look-ahead bias) ---")
    trend_predictions = []
    for district in df['district_no'].unique():
        district_df = df[df['district_no'] == district].copy()
        district_train = district_df[district_df['year'] < validation_start_year]
        if len(district_train) < 2: continue
        trend_model = LinearRegression()
        trend_model.fit(district_train[['year']], district_train['kreisYield'])
        district_df['yield_trend'] = trend_model.predict(district_df[['year']])
        trend_predictions.append(district_df[['district_no', 'year', 'yield_trend']])

    trend_df = pd.concat(trend_predictions)
    df = pd.merge(df, trend_df, on=['district_no', 'year'], how='left')
    df.dropna(subset=['yield_trend'], inplace=True)  # Drop districts with too little data for a trend

    # === CRITICAL CHANGE 2: THE TARGET IS NOW THE ANOMALY ===
    df['yield_anomaly'] = df['kreisYield'] - df['yield_trend']
    TARGET_COL = 'yield_anomaly'
    print(" -> Target variable is now the anomaly from the extrapolated trend.")

    df['district_no_encoded'] = df['district_no'].astype('category').cat.codes

    train_df = df[df['year'] <= validation_start_year].copy()
    validation_df = df[df['year'] > validation_start_year].copy()

    X_train_cat = train_df[['district_no_encoded']].values
    X_validation_cat = validation_df[['district_no_encoded']].values

    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(train_df[CONTINUOUS_FEATURES])
    X_validation_cont = scaler.transform(validation_df[CONTINUOUS_FEATURES])

    y_train = train_df[TARGET_COL].values
    y_validation_anomaly = validation_df[TARGET_COL].values  # This is the anomaly
    y_validation_actual = validation_df['kreisYield'].values  # This is the true, raw yield

    n_districts = df['district_no_encoded'].nunique()
    n_continuous = X_train_cont.shape[1]
    model = build_advanced_nn_model(n_districts, n_continuous)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.fit(
        [X_train_cat, X_train_cont], y_train,
        epochs=150, batch_size=64,
        validation_data=([X_validation_cat, X_validation_cont], y_validation_anomaly),
        callbacks=[early_stopping], verbose=1
    )

    # === CRITICAL CHANGE 3: RE-TREND THE PREDICTION FOR EVALUATION ===
    predicted_anomalies = model.predict([X_validation_cat, X_validation_cont]).flatten()
    y_pred_final = validation_df['yield_trend'].values + predicted_anomalies

    r2 = r2_score(y_validation_actual, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final))

    print("\n--- Overall Validation Performance (on original scale) ---")
    print(f"  R-squared (R2): {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} dt/ha")
    print("-------------------------------------------------")

    # (Final training and saving logic would follow a similar pattern)


if __name__ == "__main__":
    train_and_validate_nn_champion()