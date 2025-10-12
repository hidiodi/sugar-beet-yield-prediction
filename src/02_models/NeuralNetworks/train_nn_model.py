# File: src/models/train_advanced_nn.py
# Description: A final, expert-level attempt to close the performance gap using a more sophisticated NN architecture.

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# --- Configuration ---
# We separate features into two groups: those that behave like categories, and continuous numbers
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
    Defines a more advanced NN architecture with separate paths for categorical and continuous data.
    """
    # --- Input Layers for our two types of data ---
    categorical_input = layers.Input(shape=(1,), name='categorical_input')
    continuous_input = layers.Input(shape=(n_continuous_features,), name='continuous_input')

    # --- Categorical Path (Embedding Layer) ---
    # The embedding dimension is a hyperparameter; sqrt(n_categories) is a good starting point.
    embedding_dim = int(np.sqrt(n_districts))
    embedding = layers.Embedding(input_dim=n_districts + 1, output_dim=embedding_dim, name='embedding')(
        categorical_input)
    embedding_flat = layers.Flatten(name='embedding_flat')(embedding)

    # --- Continuous Path (Standard Dense Layers) ---
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(continuous_input)
    x = layers.BatchNormalization()(x)  # Batch norm is very effective in NNs
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # --- Concatenate (Merge) the two paths ---
    merged = layers.concatenate([embedding_flat, x], name='concatenate')

    # --- Final Dense Layers on the combined representation ---
    final = layers.Dense(64, activation='relu')(merged)
    output = layers.Dense(1, name='output')(final)

    # --- Build and Compile the final model ---
    model = keras.Model(inputs=[categorical_input, continuous_input], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def train_and_evaluate_advanced_nn():
    """
    Performs a 10-fold cross-validation for the advanced Neural Network.
    """
    df = pd.read_csv(DATA_PATH)
    # Important: Convert district_no to a categorical type for the embedding layer
    df['district_no'] = df['district_no'].astype('category').cat.codes

    print("--- Starting 10-Fold CV for Advanced Neural Network ---")
    r2_scores, rmse_scores = [], []
    N_FOLDS = 10

    for i in range(N_FOLDS):
        print(f"\n--- FOLD {i + 1}/{N_FOLDS} ---")

        # --- Data Split ---
        # The splitting logic remains the same
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=i, stratify=df['year'])

        # --- Prepare Inputs for the Model ---
        X_train_cat = train_df[CATEGORICAL_FEATURES]
        X_test_cat = test_df[CATEGORICAL_FEATURES]

        scaler = StandardScaler()
        X_train_cont = scaler.fit_transform(train_df[CONTINUOUS_FEATURES])
        X_test_cont = scaler.transform(test_df[CONTINUOUS_FEATURES])

        y_train, y_test = train_df[TARGET_COL], test_df[TARGET_COL]

        # --- Build and Train ---
        n_districts = df['district_no'].nunique()
        n_continuous = X_train_cont.shape[1]
        model = build_advanced_nn_model(n_districts, n_continuous)

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        history = model.fit(
            [X_train_cat, X_train_cont], y_train,
            epochs=150, batch_size=128,
            validation_split=0.2,
            callbacks=[early_stopping], verbose=0
        )
        print(f"  Training stopped at epoch {len(history.history['loss'])}")

        # --- Evaluate ---
        y_pred = model.predict([X_test_cat, X_test_cont]).flatten()
        r2, rmse = r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2);
        rmse_scores.append(rmse)
        print(f"  R-squared (R2): {r2:.4f}, RMSE: {rmse:.2f} dt/ha")

    print("\n-------------------------------------------------")
    print("--- Final Advanced NN Performance Summary ---")
    print(f"Average R-squared (R2): {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.2f} +/- {np.std(rmse_scores):.2f} dt/ha")
    print("-------------------------------------------------")


if __name__ == "__main__":
    # A slightly different splitting strategy is needed for this model, so the main loop is self-contained.
    train_and_evaluate_advanced_nn()