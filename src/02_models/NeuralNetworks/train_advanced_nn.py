# File: train_champion_nn.py
# Description: FINAL CHAMPION NN MODEL. Implements the robust "predict the anomaly"
# strategy. It uses a causal rolling mean to detrend and trains the NN to predict
# the deviation from that trend, using only non-trend features.

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
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
# --- FIX #1: Use the correct, consistent feature list ---
CONTINUOUS_FEATURES = [
    'avg_elevation', 'avg_soil_pawc', 'lon', 'lat',
    'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
    'producer_price_index_lag1_anomaly',
    'seed_price_index_lag1_anomaly',
    'energy_price_index_lag1_anomaly', 'fertilizer_price_index_lag1_anomaly',
    'plant_protection_price_index_lag1_anomaly',
    'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast',
    'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast',
    'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast',
    'summer_heat_x_profit_margin', 'summer_precip_x_input_costs',
    'spring_temp_anomaly_forecast_sq', 'summer_temp_anomaly_forecast_sq',
    'spring_precip_anomaly_forecast_sq', 'summer_precip_anomaly_forecast_sq'
]
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_PATH = os.path.join('src/models', 'final_nn_model_champion.keras')
SCALER_PATH = os.path.join('src/models', 'final_nn_scaler.joblib')
DISTRICT_ENCODER_PATH = os.path.join('src/models', 'final_nn_district_encoder.joblib')


def build_advanced_nn_model(n_districts, n_continuous_features):
    """Defines the NN architecture."""
    # Input layers
    categorical_input = layers.Input(shape=(1,), name='categorical_input')
    continuous_input = layers.Input(shape=(n_continuous_features,), name='continuous_input')

    # Embedding for districts
    embedding_dim = min(50, int(np.sqrt(n_districts)))  # Capped embedding dimension
    embedding = layers.Embedding(input_dim=n_districts, output_dim=embedding_dim, name='embedding')(categorical_input)
    embedding_flat = layers.Flatten(name='embedding_flat')(embedding)

    # Dense layers for continuous features
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(continuous_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Merged pathway
    merged = layers.concatenate([embedding_flat, x], name='concatenate')
    final = layers.Dense(32, activation='relu')(merged)
    output = layers.Dense(1, name='output')(final)

    model = keras.Model(inputs=[categorical_input, continuous_input], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_and_validate_nn_champion():
    """
    Implements the robust "predict the anomaly" strategy for the NN model,
    using causal detrending to prevent data leakage.
    """
    df = pd.read_csv(DATA_PATH)

    # --- FIX #2: Use the robust, Causal Rolling Mean Detrending ---
    print("\n--- Applying Causal (Trailing Mean) Detrending to Prevent Data Leakage ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x)
    df.dropna(subset=['yield_trend'], inplace=True)

    # The target is the anomaly from this causal trend
    df['yield_anomaly'] = df['kreisYield'] - df['yield_trend']
    TARGET_COL = 'yield_anomaly'
    print(" -> Detrending complete. Target is now the anomaly.")

    # Check for missing feature columns before proceeding
    missing_cols = [col for col in CONTINUOUS_FEATURES if col not in df.columns]
    if missing_cols:
        print(f"❌ CRITICAL ERROR: The following feature columns are missing: {missing_cols}")
        return

    # Create a consistent categorical encoding
    df['district_no_encoded'] = df['district_no'].astype('category').cat.codes
    district_encoder = dict(enumerate(df['district_no'].astype('category').cat.categories))

    # Split data into train and validation sets
    validation_start_year = 2015  # Using the same split as the final XGBoost tests
    train_df = df[df['year'] < validation_start_year].copy()
    validation_df = df[df['year'] >= validation_start_year].copy()
    print(f"\nTraining on years < {validation_start_year}, validating on years >= {validation_start_year}")

    # Prepare categorical data
    X_train_cat = train_df[['district_no_encoded']].values
    X_validation_cat = validation_df[['district_no_encoded']].values

    # Prepare and scale continuous data (fit ONLY on training data)
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(train_df[CONTINUOUS_FEATURES])
    X_validation_cont = scaler.transform(validation_df[CONTINUOUS_FEATURES])

    # Prepare target variables
    y_train = train_df[TARGET_COL].values
    y_validation_actual = validation_df['kreisYield'].values  # True, raw yield for final evaluation

    n_districts = df['district_no_encoded'].nunique()
    n_continuous = X_train_cont.shape[1]

    model = build_advanced_nn_model(n_districts, n_continuous)
    model.summary()  # Print model architecture

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    print("\n--- Starting Neural Network Training ---")
    model.fit(
        [X_train_cat, X_train_cont], y_train,
        epochs=150, batch_size=64,
        validation_split=0.1,  # Use a portion of training data for early stopping monitoring
        callbacks=[early_stopping], verbose=1
    )

    # Evaluate on the hold-out validation set
    print("\n--- Evaluating on Hold-out Validation Set ---")
    predicted_anomalies = model.predict([X_validation_cat, X_validation_cont]).flatten()
    y_pred_final = validation_df['yield_trend'].values + predicted_anomalies

    r2 = r2_score(y_validation_actual, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final))

    print("\n--- Overall Validation Performance (on original scale) ---")
    print(f"  R-squared (R2): {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} dt/ha")
    print("-------------------------------------------------")

    # Save the final model, scaler, and encoder
    print("\n--- Saving final model and preprocessing objects ---")
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(district_encoder, DISTRICT_ENCODER_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Scaler saved to {SCALER_PATH}")
    print(f"✅ Encoder saved to {DISTRICT_ENCODER_PATH}")


if __name__ == "__main__":
    train_and_validate_nn_champion()