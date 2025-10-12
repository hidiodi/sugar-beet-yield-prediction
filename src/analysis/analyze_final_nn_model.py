# File: src/models/analyze_final_nn_model.py
# Description: Performs analysis (PDP) on the final Neural Network model trained with walk-forward validation.
#              SHAP analysis has been removed due to persistent gradient computation errors with DeepExplainer.

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os
import warnings
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler  # For type hinting and clarity

import tensorflow as tf
from tensorflow import keras

# Suppress warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
warnings.filterwarnings("ignore")

# --- Configuration - Must match the training script (train_advanced_nn.py) ---
NN_CATEGORICAL_FEATURES = ['district_no_encoded']  # Note: 'district_no' is pre-encoded to 'district_no_encoded'
NN_CONTINUOUS_FEATURES = [
    'avg_elevation', 'avg_soil_pawc', 'winter_temp_anomaly', 'winter_precip_anomaly',
    'national_avg_yield_lag1', 'producer_price_index_lag1', 'spring_temp_anomaly_hybrid',
    'spring_precip_anomaly_hybrid', 'summer_temp_anomaly_hybrid', 'summer_precip_anomaly_hybrid',
    'fertilizer_price_index', 'energy_price_index'
]
TARGET_COL = 'kreisYield'

# --- Paths ---
MODEL_PATH = os.path.join('src/models', 'final_nn_model_champion_walkforward.keras')
SCALER_PATH = os.path.join('src/models', 'final_nn_model_champion_walkforward_scaler.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
REPORT_DIR = os.path.join('reports', 'stage1_nn_explanation_walkforward')


def load_model_scaler_and_data():
    """
    Loads the trained Keras NN model, StandardScaler, and the full dataset.
    """
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(DATA_PATH)
        # Ensure 'district_no' is encoded consistently as during training
        df['district_no_encoded'] = df['district_no'].astype('category').cat.codes
        print(f"✅ Successfully loaded {len(df)} samples from {DATA_PATH}.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Cannot proceed.")
        return None, None, None

    # --- 2. Load Model ---
    try:
        nn_model = keras.models.load_model(MODEL_PATH)
        print(f"✅ Successfully loaded NN model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error: NN model file not found or failed to load from {MODEL_PATH}. Error: {e}")
        return None, None, None

    # --- 3. Load Scaler ---
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Successfully loaded StandardScaler from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"Error: StandardScaler file not found at {SCALER_PATH}. Ensure it was saved during training.")
        return None, None, None

    # --- Data Integrity Check for features ---
    required_features = NN_CATEGORICAL_FEATURES + NN_CONTINUOUS_FEATURES
    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        print(f"Error: The data file is missing required feature columns for NN: {missing_cols}")
        return None, None, None

    return nn_model, scaler, df


# Wrapper for NN model prediction to be used by PDP
class NNPredictWrapper:
    def __init__(self, model, scaler, categorical_features, continuous_features):
        self.model = model
        self.scaler = scaler
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        # Store original feature names for consistent output if needed
        self.feature_names_combined = self.categorical_features + self.continuous_features

    def fit(self, X=None, y=None):
        """
        Dummy fit method to satisfy sklearn.inspection.PartialDependenceDisplay requirements.
        The actual model is already trained.
        """
        return self

    def predict(self, X_df):
        """
        Takes a DataFrame of raw features, preprocesses it, and returns predictions.
        Used by PartialDependenceDisplay.
        """
        # Ensure X_df has the necessary columns
        # Note: X_df might only contain the features requested by PDP, not all features.
        # We need to reconstruct the full input based on the NN_CATEGORICAL_FEATURES and NN_CONTINUOUS_FEATURES.

        # Identify which features from the wrapper's full list are in X_df
        cat_in_X_df = [col for col in self.categorical_features if col in X_df.columns]
        cont_in_X_df = [col for col in self.continuous_features if col in X_df.columns]

        # Extract features from X_df. Handle missing columns by creating dummy ones if needed
        # (though PDP should pass all relevant features for a given plot)
        X_cat = X_df[cat_in_X_df].values if cat_in_X_df else np.empty((len(X_df), 0), dtype=np.int32)
        X_cont_unscaled = X_df[cont_in_X_df] if cont_in_X_df else pd.DataFrame(index=X_df.index)

        # To use the scaler, all continuous features must be present.
        # Create a full dummy DataFrame for scaling, filling missing columns with zeros or appropriate defaults
        # This is a bit of a hack but necessary for `scaler.transform` to work if `X_df` is a subset.
        full_cont_df = pd.DataFrame(0, index=X_df.index, columns=self.continuous_features)
        for col in cont_in_X_df:
            full_cont_df[col] = X_cont_unscaled[col]

        X_cont_scaled = self.scaler.transform(full_cont_df)

        # Model expects a list of inputs.
        # Handle cases where one input might be empty (e.g., if only plotting continuous feature)
        model_inputs = []
        if self.categorical_features:  # If NN model was built with a categorical input
            # Match the input shape expected by the model for categorical features
            # The model expects (batch_size, 1) for district_no_encoded
            if X_cat.shape[1] == 1:  # If district_no_encoded is the only categorical feature
                model_inputs.append(X_cat)
            else:  # If there were multiple categorical features, need to adapt
                # For now, assuming only one 'district_no_encoded' categorical feature
                raise ValueError(
                    "NNPredictWrapper currently assumes only one categorical feature (district_no_encoded).")

        if self.continuous_features:  # If NN model was built with a continuous input
            model_inputs.append(X_cont_scaled)

        if not model_inputs:
            raise ValueError("No valid inputs for the NN model could be constructed from X_df.")

        return self.model.predict(model_inputs).flatten()

    # This method is needed specifically for `sklearn.inspection.PartialDependenceDisplay`
    # It expects an estimator with a `feature_names_in_` attribute if `features` argument is by name.
    @property
    def feature_names_in_(self):
        return self.feature_names_combined


def plot_feature_dependence(nn_model: keras.Model, scaler: StandardScaler, df_full: pd.DataFrame,
                            features_to_plot: list):
    """Generates Partial Dependence Plots (PDPs) for the functional relationship of the NN."""
    print("\nGenerating Partial Dependence Plots (Feature Relationships) for NN...")

    # Create the wrapper for the NN model
    nn_wrapper = NNPredictWrapper(nn_model, scaler, NN_CATEGORICAL_FEATURES, NN_CONTINUOUS_FEATURES)

    try:
        n_features = len(features_to_plot)
        if n_features == 0:
            print("No features specified for PDP. Skipping.")
            return

        n_cols = min(n_features, 3)
        n_rows = int(np.ceil(n_features / n_cols))
        fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(6 * n_cols, 5 * n_rows))
        ax = ax.flatten() if n_features > 1 else [ax]

        # The DataFrame for PDP should include all relevant features,
        # but the wrapper handles extracting and preprocessing correctly.
        df_for_pdp = df_full[NN_CATEGORICAL_FEATURES + NN_CONTINUOUS_FEATURES]

        PartialDependenceDisplay.from_estimator(
            nn_wrapper, df_for_pdp, features_to_plot, kind='average', ax=ax,
        )
        fig.suptitle("Partial Dependence: How Features Influence Yield Prediction (NN)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(REPORT_DIR, 'pdp_feature_relationships_nn.png')
        plt.savefig(save_path)
        print(f"✅ PDP Plots saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating PDPs for NN: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    nn_model, scaler, df_full = load_model_scaler_and_data()

    if nn_model is not None and scaler is not None and df_full is not None:
        print(f"\n--- Starting Explanation Analysis for Stage 1 Pre-Season NN Model (Walk-Forward) ---")

        # 1. Feature Relationships: How do the most important features drive predictions?
        # Note: Ensure these features exist in NN_CONTINUOUS_FEATURES or is 'district_no_encoded'
        top_features_for_pdp = [
            'national_avg_yield_lag1',
            'summer_temp_anomaly_hybrid',
            'producer_price_index_lag1',
            'avg_soil_pawc',
            'winter_temp_anomaly',
            'district_no_encoded'  # Use the encoded version
        ]
        plot_feature_dependence(nn_model, scaler, df_full, top_features_for_pdp)

        print("\n✅ Analysis Complete. Outputs saved to directory: " + REPORT_DIR)
    else:
        print("\nAnalysis failed due to missing model, scaler, or data.")