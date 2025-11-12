# vsm_cybernetic_model/pipelines/run_regulator_training_pipeline.py
import pandas as pd
from xgboost import XGBRegressor
import joblib
import sys
import logging

from vsm_cybernetic_model.configs import main_config as cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_regulator_model():
    """Trains the final Stage 2 XGBoost regulator model on the VSM principal components."""
    logging.info("--- Starting Stage 2 Regulator Model Training ---")

    try:
        vsm_features_df = pd.read_csv(cfg.FINAL_FEATURES_PATH)
        logging.info(f"Loaded VSM features from '{cfg.FINAL_FEATURES_PATH}'")

        master_df = pd.read_csv(cfg.DATA_DIR / "04_master" / "master_dataset.csv")
        # Load the raw 'yield' column
        target_df = master_df[['district_no', 'year', 'yield']].copy()
        # Rename it for consistency with the rest of the pipeline
        target_df.rename(columns={'yield': 'kreisYield'}, inplace=True)
        logging.info("Loaded target variable 'yield' from master dataset and renamed to 'kreisYield'.")

    except FileNotFoundError as e:
        logging.error(f"FATAL: A required file was not found. {e}. Aborting.")
        sys.exit(1)

    final_df = pd.merge(vsm_features_df, target_df, on=['district_no', 'year'], how='inner')

    feature_cols = [col for col in final_df.columns if col.startswith('VSM')]
    target_col = 'kreisYield'
    final_df.dropna(subset=feature_cols + [target_col], inplace=True)

    if final_df.empty:
        logging.error("FATAL: No data available for training after dropping NaNs. Aborting.")
        sys.exit(1)

    X = final_df[feature_cols]
    y = final_df[target_col]
    logging.info(f"Data prepared. Training on {len(X)} samples with {len(feature_cols)} VSM features.")

    base_score_value = y.mean()
    logging.info(f"Setting XGBoost base_score to {base_score_value:.4f}")

    # Using hyperparameters similar to your original model for a fair comparison
    xgb_regulator = XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
        base_score=base_score_value
    )

    logging.info("Training the final XGBoost regulator model...")
    xgb_regulator.fit(X, y)
    logging.info("Training complete.")

    output_path = cfg.REGULATOR_MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_regulator, output_path)
    logging.info(f"✓ Final regulator model saved to '{output_path}'")
    logging.info("--- Stage 2 Training Finished Successfully ---")

if __name__ == "__main__":
    train_regulator_model()