# File: src/models/train_bagged_nn.py
# Description: Uses a Bagging ensemble of advanced NNs to improve stability and performance.

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample  # For creating bootstrap samples

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# --- Configuration (remains the same) ---
CATEGORICAL_FEATURES = ['district_no']
CONTINUOUS_FEATURES = [
    'avg_elevation', 'avg_soil_pawc', 'winter_temp_anomaly', 'winter_precip_anomaly',
    'national_avg_yield_lag1', 'producer_price_index_lag1', 'spring_temp_anomaly_hybrid',
    'spring_precip_anomaly_hybrid', 'summer_temp_anomaly_hybrid', 'summer_precip_anomaly_hybrid',
    'fertilizer_price_index', 'energy_price_index'
]
TARGET_COL = 'kreisYield'
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')


def build_advanced_nn_model(n_districts, n_continuous_features):
    """
    Defines the architecture of our base learner NN. (Unchanged from previous script)
    """
    categorical_input = layers.Input(shape=(1,), name='categorical_input')
    continuous_input = layers.Input(shape=(n_continuous_features,), name='continuous_input')

    embedding_dim = int(np.sqrt(n_districts))
    embedding = layers.Embedding(input_dim=n_districts + 1, output_dim=embedding_dim)(categorical_input)
    embedding_flat = layers.Flatten()(embedding)

    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(continuous_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    merged = layers.concatenate([embedding_flat, x])
    final = layers.Dense(64, activation='relu')(merged)
    output = layers.Dense(1)(final)

    model = keras.Model(inputs=[categorical_input, continuous_input], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def train_bagging_ensemble():
    """
    Trains a bagging ensemble of Neural Network models for improved performance.
    """
    df = pd.read_csv(DATA_PATH)
    df['district_no'] = df['district_no'].astype('category').cat.codes

    # --- We will do a single, stable split to evaluate the ensemble's performance ---
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['year'])

    # --- Prepare Test Data (this will be used to evaluate all models) ---
    X_test_cat = test_df[CATEGORICAL_FEATURES]
    y_test = test_df[TARGET_COL]

    # Scale the continuous features
    scaler = StandardScaler()
    X_train_cont_full = scaler.fit_transform(train_df[CONTINUOUS_FEATURES])  # Fit on full training data
    X_test_cont = scaler.transform(test_df[CONTINUOUS_FEATURES])

    # Store the scaled training data in a DataFrame to make bootstrapping easier
    train_df_scaled_cont = pd.DataFrame(X_train_cont_full, index=train_df.index, columns=CONTINUOUS_FEATURES)

    # --- Bagging Ensemble Training Loop ---
    N_MODELS = 10  # Number of models in our ensemble
    ensemble_predictions = []

    print(f"--- Training Bagging Ensemble with {N_MODELS} Neural Network Models ---")

    for i in range(N_MODELS):
        print(f"\n--- Training Model {i + 1}/{N_MODELS} ---")

        # 1. Create a bootstrap sample of the training data
        # We sample the indices to ensure categorical and continuous data stay aligned
        bootstrap_indices = resample(train_df.index, replace=True, n_samples=len(train_df), random_state=i)

        X_train_cat_sample = train_df.loc[bootstrap_indices][CATEGORICAL_FEATURES]
        X_train_cont_sample = train_df_scaled_cont.loc[bootstrap_indices]
        y_train_sample = train_df.loc[bootstrap_indices][TARGET_COL]

        # 2. Build and Train the model on the bootstrap sample
        n_districts = df['district_no'].nunique()
        n_continuous = X_train_cont_sample.shape[1]
        model = build_advanced_nn_model(n_districts, n_continuous)

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        history = model.fit(
            [X_train_cat_sample, X_train_cont_sample], y_train_sample,
            epochs=150, batch_size=128,
            validation_split=0.2,
            callbacks=[early_stopping], verbose=0
        )
        print(f"  Training stopped at epoch {len(history.history['loss'])}")

        # 3. Get predictions on the single, held-out test set
        model_preds = model.predict([X_test_cat, X_test_cont]).flatten()
        ensemble_predictions.append(model_preds)

    # --- Evaluate the Ensemble Performance ---
    # Average the predictions from all models
    final_predictions = np.mean(ensemble_predictions, axis=0)

    ensemble_r2 = r2_score(y_test, final_predictions)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))

    print("\n-------------------------------------------------")
    print("--- Final Bagging Ensemble Performance Summary ---")
    print(f"Ensemble R-squared (R2): {ensemble_r2:.4f}")
    print(f"Ensemble RMSE: {ensemble_rmse:.2f} dt/ha")
    print("-------------------------------------------------")


if __name__ == "__main__":
    train_bagging_ensemble()