# File: src/models/train_advanced_nn.py
# Description: MODIFIED to use a time-based split for validation and to report
#              validation loss on a per-district basis.

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
CONTINUOUS_FEATURES = [
    'avg_elevation', 'avg_soil_pawc', 'winter_temp_anomaly', 'winter_precip_anomaly',
    'national_avg_yield_lag1', 'producer_price_index_lag1', 'spring_temp_anomaly_hybrid',
    'spring_precip_anomaly_hybrid', 'summer_temp_anomaly_hybrid', 'summer_precip_anomaly_hybrid',
]
TARGET_COL = 'kreisYield'
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_PATH = os.path.join('src/models', 'final_nn_model_champion.keras')


def build_advanced_nn_model(n_districts, n_continuous_features):
    """
    Defines a more advanced NN architecture with separate paths for categorical and continuous data.
    """
    categorical_input = layers.Input(shape=(len(CATEGORICAL_FEATURES),), name='categorical_input')
    continuous_input = layers.Input(shape=(n_continuous_features,), name='continuous_input')

    embedding_dim = int(np.sqrt(n_districts))
    embedding = layers.Embedding(input_dim=n_districts + 1, output_dim=embedding_dim, name='embedding')(
        categorical_input)
    embedding_flat = layers.Flatten(name='embedding_flat')(embedding)

    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(continuous_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    merged = layers.concatenate([embedding_flat, x], name='concatenate')

    final = layers.Dense(64, activation='relu')(merged)
    output = layers.Dense(1, name='output')(final)

    model = keras.Model(inputs=[categorical_input, continuous_input], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_and_validate_nn_with_holdout():
    """
    Loads pre-season data, uses a time-based split to evaluate the NN model,
    reports performance per district, and then trains a final model.
    """
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}. Please run the feature engineering script.")
        return

    df['district_no_encoded'] = df['district_no'].astype('category').cat.codes

    last_year = df['year'].max()
    validation_start_year = last_year - 5

    print(f"--- Using Last 5 Years for Validation ---")
    print(f"Training data will be from years before {validation_start_year + 1}")
    print(f"Validation data will be from years {validation_start_year + 1} to {last_year}")

    train_df = df[df['year'] <= validation_start_year].copy()
    validation_df = df[df['year'] > validation_start_year].copy()

    X_train_cat = train_df[['district_no_encoded']].values
    X_validation_cat = validation_df[['district_no_encoded']].values

    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(train_df[CONTINUOUS_FEATURES])
    X_validation_cont = scaler.transform(validation_df[CONTINUOUS_FEATURES])

    y_train = train_df[TARGET_COL].values
    y_validation = validation_df[TARGET_COL].values

    print(f"Training set size: {len(X_train_cat)} samples")
    print(f"Validation set size: {len(X_validation_cat)} samples")

    n_districts = df['district_no_encoded'].nunique()
    n_continuous = X_train_cont.shape[1]
    model = build_advanced_nn_model(n_districts, n_continuous)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        [X_train_cat, X_train_cont], y_train,
        epochs=150, batch_size=128,
        validation_data=([X_validation_cat, X_validation_cont], y_validation),
        callbacks=[early_stopping], verbose=1
    )
    print(
        f"  Training stopped at epoch {len(history.history['loss'])} (best val_loss at epoch {np.argmin(history.history['val_loss']) + 1})")

    y_pred = model.predict([X_validation_cat, X_validation_cont]).flatten()
    r2 = r2_score(y_validation, y_pred)
    rmse = np.sqrt(mean_squared_error(y_validation, y_pred))

    print("\n--- Overall Validation Performance ---")
    print(f"  R-squared (R2): {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- Per-District Performance Analysis ---
    validation_df['predictions'] = y_pred
    validation_df['sq_error'] = (validation_df[TARGET_COL] - validation_df['predictions']) ** 2

    district_performance = validation_df.groupby('district_no').agg(
        mean_sq_error=('sq_error', 'mean'),
        num_samples=('sq_error', 'size')
    ).reset_index()
    district_performance['rmse'] = np.sqrt(district_performance['mean_sq_error'])
    district_performance = district_performance.sort_values('rmse', ascending=True)

    print("\n--- Validation RMSE per District (Holdout Years) ---")
    for _, row in district_performance.iterrows():
        print(f"  District {int(row['district_no'])}: RMSE = {row['rmse']:.2f} dt/ha ({row['num_samples']} samples)")
    print("-------------------------------------------------")

    print("\n--- Training Final Model on Data Before the Holdout Period for Deployment ---")
    final_model = build_advanced_nn_model(n_districts, n_continuous)
    final_early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    final_model.fit(
        [X_train_cat, X_train_cont], y_train,
        epochs=np.argmin(history.history['val_loss']) + 1,
        batch_size=128,
        callbacks=[final_early_stopping],
        verbose=1
    )

    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        final_model.save(MODEL_PATH)
        joblib.dump(scaler, MODEL_PATH.replace('.keras', '_scaler.joblib'))
        print(f"\n✅ Final Advanced NN model successfully trained and saved to {MODEL_PATH}")
        print(f"✅ StandardScaler fitted on training data saved to {MODEL_PATH.replace('.keras', '_scaler.joblib')}")
    except Exception as e:
        print(f"\n❌ Warning: Could not save the final model or scaler. Error: {e}")


if __name__ == "__main__":
    train_and_validate_nn_with_holdout()