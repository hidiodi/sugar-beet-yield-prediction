import pandas as pd
import logging
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys
import numpy as np
import json

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])
INPUT_FILENAME = 'super_ensemble_training_data.csv'
MODEL_FILENAME = 'super_ensemble_meta_learner_TSCV.json'  # Updated filename
LABEL_MAP_FILENAME = 'meta_learner_label_map_TSCV.json'  # Updated filename

# --- Time Series Split Configuration ---
# All data <= SPLIT_YEAR will be used for training.
# All data > SPLIT_YEAR will be used for testing (Out-of-Time Validation).
SPLIT_YEAR = 2019


def train_and_evaluate_meta_learner_ts():
    """
    Loads prepared data and trains an XGBoost classifier using a strict Time-Series Split.
    """
    input_path = OUTPUT_DIR / INPUT_FILENAME
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}. Please run super_ensemble_data_prep.py first.")
        return

    logging.info("--- Loading Prepared Data for Meta-Learner Training (TSCV) ---")
    df = pd.read_csv(input_path)

    # 1. Define Features (X) and Target (Y)
    TARGET_COL = 'Best_Model'

    EXCLUDE_COLS = ['year', 'district_no', 'kreisYield', TARGET_COL]
    PRED_COLS = [col for col in df.columns if col.endswith('_pred')]

    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS + PRED_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]

    logging.info(f"Target variable: {TARGET_COL}")
    logging.info(f"Number of training features: {len(feature_cols)}")

    # 2. Strict Time-Series Split (TSCV)
    # Train: All data up to and including SPLIT_YEAR
    # Test: All data after SPLIT_YEAR

    X_train = df[df['year'] <= SPLIT_YEAR][feature_cols]
    y_train = df[df['year'] <= SPLIT_YEAR][TARGET_COL]

    X_test = df[df['year'] > SPLIT_YEAR][feature_cols]
    y_test = df[df['year'] > SPLIT_YEAR][TARGET_COL]

    logging.info("\n" + "=" * 50)
    logging.info(f"Time Series Split Point: {SPLIT_YEAR}")
    logging.info(f"Training Period: <={SPLIT_YEAR} ({len(X_train)} samples)")
    logging.info(f"Testing Period: >{SPLIT_YEAR} ({len(X_test)} samples)")
    logging.info("=" * 50)

    if X_test.empty:
        logging.error("Testing set is empty. Adjust SPLIT_YEAR.")
        return

    # 3. Train the XGBoost Classifier
    logging.info("\n--- Training XGBoost Meta-Learner (TSCV) ---")

    model = XGBClassifier(
        objective='multi:softmax',
        n_estimators=100,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )

    # Need to map string labels to integers for XGBoost
    label_map = {label: i for i, label in enumerate(y_train.unique())}
    y_train_encoded = y_train.map(label_map)
    y_test_encoded = y_test.map(label_map)

    model.fit(X_train, y_train_encoded)

    # 4. Evaluate the Model on Out-of-Time Test Set
    y_pred_encoded = model.predict(X_test)

    # Decode predictions back to original model names for clear reporting
    reverse_label_map = {i: label for label, i in label_map.items()}
    y_pred = pd.Series(y_pred_encoded).map(reverse_label_map)

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"\nMeta-Learner Classification Accuracy (Out-of-Time Test Set): **{accuracy:.4f}**")

    logging.info("\nClassification Report (How well it picks the best model):")
    logging.info(classification_report(y_test, y_pred))

    # 5. Save the Trained Model and Map
    model_path = OUTPUT_DIR / MODEL_FILENAME
    model.save_model(model_path)

    # Save the label map for future use in prediction
    with open(OUTPUT_DIR / LABEL_MAP_FILENAME, 'w') as f:
        json.dump(label_map, f)

    logging.info(f"--- TSCV Meta-Learner training complete. Model saved to: {model_path} ---")


if __name__ == '__main__':
    train_and_evaluate_meta_learner_ts()