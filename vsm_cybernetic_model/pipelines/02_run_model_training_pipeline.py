# 02_run_model_training_pipeline.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from vsm_cybernetic_model.configs import main_config as cfg

def train_regulator_model():
    """
    Trains and saves the final Stage 2 XGBoost regulator model.

    This function loads the final feature matrix, splits it into training
    and testing sets, trains an XGBoost model to predict the yield gap,
    and saves the trained model.
    """
    print("--- Training Stage 2 XGBoost Regulator Model ---")

    # 1. Load Data
    try:
        df = pd.read_csv(cfg.FINAL_FEATURES_PATH)
        print(f"Loaded final feature matrix from '{cfg.FINAL_FEATURES_PATH}'")
    except FileNotFoundError:
        print(f"Error: Final features file not found at '{cfg.FINAL_FEATURES_PATH}'.")
        print("Please run the feature engineering pipeline in 'transform' mode first.")
        return

    # 2. Prepare Data for Modeling
    df = df.dropna() # Drop rows with missing components

    # In a real scenario, the target `yield_gap` would be pre-calculated.
    # Here, we create a dummy target for demonstration.
    if 'yield_gap' not in df.columns:
        print("Creating dummy 'yield_gap' target variable.")
        df['yield_gap'] = df.filter(regex='PC1').sum(axis=1) * 0.1 + 5


    feature_cols = [col for col in df.columns if 'PC' in col]
    target_col = 'yield_gap'

    X = df[feature_cols]
    y = df[target_col]

    if X.empty:
        print("Error: No data available for training.")
        return

    print(f"Training model with {len(feature_cols)} features.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train XGBoost Model
    print("Training XGBoost regressor...")
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate Model (Simple)
    score = model.score(X_test, y_test)
    print(f"Model R^2 score on test set: {score:.4f}")

    # 5. Save Model
    print(f"Saving trained regulator model to '{cfg.REGULATOR_MODEL_PATH}'")
    joblib.dump(model, cfg.REGULATOR_MODEL_PATH)

    print("--- Stage 2 Regulator training complete ---")

if __name__ == "__main__":
    train_regulator_model()
