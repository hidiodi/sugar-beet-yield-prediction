# File: src/models/rnn_model.py
# Description: A deeper RNN model using stacked Bidirectional LSTMs with an Attention mechanism
#              to capture more complex temporal patterns and improve predictive accuracy.

import pandas as pd
import numpy as np
import os
import warnings
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Layer
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# --- V5 MODEL PATHS (HEAVIER REGULARIZATION) ---
MODEL_PATH = os.path.join('src/models', 'final_rnn_model_v5_regularized.h5')
SCALER_PATH = os.path.join('src/models', 'rnn_scaler_v5_regularized.joblib')

# ============================ V2 FEATURE SET ============================
FEATURE_COLS = [
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
    'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'avg_bdod_0_100cm',
    'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm', 'avg_phh2o_0_100cm', 'winter_cropland_ndvi_mean',
    'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
    'winter_cropland_snow_cover_days', 'national_avg_yield_lag1', 'producer_price_index_lag1',
    'seed_price_index_lag1', 'energy_price_index_lag1', 'fertilizer_price_index_lag1',
    'plant_protection_price_index_lag1', 'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
    'producer_price_index_lag1_anomaly', 'seed_price_index_lag1_anomaly', 'energy_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly', 'plant_protection_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme', 'is_summer_forecast_dry',
    'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip', 'antecedent_gdd_sum_anomaly_sq',
    'summer_heat_x_profit_margin', 'summer_precip_x_input_costs', 'spring_temp_prob_warm_forecast_sq',
    'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq', 'summer_precip_prob_wet_forecast_sq'
]
# ======================================================================

class Attention(Layer):
    """
    Custom Keras Attention Layer.
    Calculates a weighted average of the sequence, where weights are learned.
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def create_sequences(X, y, time_steps=1):
    """Creates sequences from numpy arrays for RNN model training."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_deep_attention_model(input_shape):
    """Defines and compiles a deeper, more regularized RNN model with stacked LSTMs and Attention."""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),  # Increased dropout
        Bidirectional(LSTM(64, return_sequences=True)), # Second BiLSTM layer
        BatchNormalization(),
        Dropout(0.4),  # Increased dropout
        Attention(), # Attention layer
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),  # Increased dropout
        Dense(32, activation='relu'), # Additional dense layer
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_evaluate_rnn(time_steps=5):
    """
    Loads data, preprocesses it for the RNN, builds, trains,
    and evaluates the deep attention-based RNN model.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    df.sort_values(by=['district_no', 'year'], inplace=True)
    target_col = 'kreisYield'
    detrended_target_col = 'kreisYield_detrended'

    print("\n--- Applying Causal (Trailing Mean) Detrending ---")
    df['yield_trend'] = df.groupby('district_no')[target_col].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x)
    df.dropna(subset=['yield_trend'], inplace=True)
    df[detrended_target_col] = df[target_col] - df['yield_trend']

    validation_start_year = 2007
    test_start_year = 2015

    train_df = df[df['year'] < validation_start_year].copy()
    validation_df = df[(df['year'] >= validation_start_year) & (df['year'] < test_start_year)].copy()
    test_df = df[df['year'] >= test_start_year].copy()

    scaler = StandardScaler()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    validation_df[FEATURE_COLS] = scaler.transform(validation_df[FEATURE_COLS])
    test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])

    def prepare_data_for_rnn(dataframe):
        X_list, y_list, indices = [], [], []
        for district_id in dataframe['district_no'].unique():
            district_df = dataframe[dataframe['district_no'] == district_id].sort_values('year')
            if len(district_df) > time_steps:
                X, y = create_sequences(
                    district_df[FEATURE_COLS].values,
                    district_df[detrended_target_col].values,
                    time_steps
                )
                X_list.append(X)
                y_list.append(y)
                indices.extend(district_df.index[time_steps:])
        if not X_list: return np.array([]), np.array([]), pd.Index([])
        return np.concatenate(X_list), np.concatenate(y_list), pd.Index(indices)

    X_train, y_train, _ = prepare_data_for_rnn(train_df)
    X_validation, y_validation, val_indices = prepare_data_for_rnn(validation_df)
    X_test, _, test_indices = prepare_data_for_rnn(test_df)

    if X_train.shape[0] == 0:
        print("❌ Error: Not enough training data to create sequences.")
        return

    model = build_deep_attention_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) # Reduced patience
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001)

    print("\n--- Training Deeper, Regularized RNN Model with Attention ---")
    model.fit(X_train, y_train, epochs=200, batch_size=64,
              validation_data=(X_validation, y_validation),
              callbacks=[early_stopping, reduce_lr], verbose=1)

    validation_df_aligned = validation_df.loc[val_indices]
    test_df_aligned = test_df.loc[test_indices]

    y_pred_detrended_val = model.predict(X_validation)
    y_pred_final_val = y_pred_detrended_val.flatten() + validation_df_aligned['yield_trend'].values
    r2_val = r2_score(validation_df_aligned[target_col], y_pred_final_val)
    rmse_val = np.sqrt(mean_squared_error(validation_df_aligned[target_col], y_pred_final_val))
    print("-------------------------------------------------")
    print(f"Validation Performance (Years {validation_start_year} to {test_start_year - 1})")
    print(f"  R-squared (R2): {r2_val:.4f}")
    print(f"  RMSE: {rmse_val:.2f} dt/ha")
    print("-------------------------------------------------")

    y_pred_detrended_test = model.predict(X_test)
    y_pred_final_test = y_pred_detrended_test.flatten() + test_df_aligned['yield_trend'].values
    r2_test = r2_score(test_df_aligned[target_col], y_pred_final_test)
    rmse_test = np.sqrt(mean_squared_error(test_df_aligned[target_col], y_pred_final_test))
    print(f"\n--- FINAL DEEP ATTENTION RNN TEST Performance (Years {test_start_year}+ Holdout) ---")
    print(f"  R-squared (R2): {r2_test:.4f}")
    print(f"  RMSE: {rmse_test:.2f} dt/ha")
    print("-------------------------------------------------")

    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Deep Attention RNN model successfully saved to {MODEL_PATH}")
        print(f"Scaler for Deep Attention RNN model saved to {SCALER_PATH}")
    except Exception as e:
        print(f"❌ Warning: Could not save the new model. Error: {e}")

if __name__ == "__main__":
    train_evaluate_rnn()

