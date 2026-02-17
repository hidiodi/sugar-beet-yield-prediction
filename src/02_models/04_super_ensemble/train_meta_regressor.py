import pandas as pd
import logging
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
import sys
import numpy as np
import json

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib
analysis_config = importlib.import_module("src.03_analysis.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = analysis_config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])

INPUT_FILENAME = 'super_ensemble_training_data.csv'
MODEL_FILENAME = 'super_ensemble_meta_learner_TSCV.json'
LABEL_MAP_FILENAME = 'meta_learner_label_map_TSCV.json'

WALK_FORWARD_START_YEAR = 2000
LAST_HISTORICAL_YEAR = 2024


def train_classifier():
    input_path = OUTPUT_DIR / INPUT_FILENAME
    if not input_path.exists(): return

    logging.info("--- Training Meta-Learner (v4: History-Aware + Cleaned) ---")
    df = pd.read_csv(input_path)

    # --- CLEANING STEP (DISABLED) ---
    # Per user request (and paper description), we now INCLUDE all data,
    # even if component models have large errors.
    # if 'Is_Garbage_Data' in df.columns:
    #     n_garbage = df['Is_Garbage_Data'].sum()
    #     if n_garbage > 0:
    #         logging.info(f"🧹 Removing {n_garbage} 'Garbage' rows (Oracle Error > 200) from Training...")
    #         df = df[df['Is_Garbage_Data'] == 0].copy()

    # Feature Selection
    exclude_cols = [
        'year', 'district_no', 'kreisYield', 'Best_Model', 'Oracle_Error',
        'Predicted_Model', 'Switch_Prediction', 'Target_Encoded',
        'Is_Garbage_Data', 'Raw_Bias', 'Regret_Weight', 'Median_Error'
    ]
    pred_cols = [c for c in df.columns if c.endswith('_pred') and c != 'Statistical_Trend_pred']
    feature_cols = [c for c in df.columns if c not in exclude_cols + pred_cols]

    logging.info(f"Features: {feature_cols}")

    # Walk-Forward Setup
    results = []
    final_model = None
    final_label_map = {}

    # Iterate through years
    for year in range(WALK_FORWARD_START_YEAR, LAST_HISTORICAL_YEAR + 1):
        train = df[df['year'] < year].copy()
        test = df[df['year'] == year].copy()

        if train.empty or test.empty: continue

        # --- FIX: Encode labels LOCALLY for this specific time slice ---
        # This ensures labels are always 0..N-1 with no gaps for the specific training set
        le = LabelEncoder()
        y_train = le.fit_transform(train['Best_Model'])

        X_train = train[feature_cols]
        X_test = test[feature_cols]

        # Dynamic num_class based on what has been seen so far
        num_classes_now = len(le.classes_)

        clf = XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes_now,  # Update dynamically
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.75,
            gamma=0.1,
            n_jobs=-1,
            random_state=42
        )

        clf.fit(X_train, y_train)

        # Predict and map back to String immediately
        pred_encoded = clf.predict(X_test)

        # Handle edge case: Model predicts a class, mapped back to string
        # Note: If test set has a 'Best_Model' never seen in train, we can't predict it anyway.
        pred_labels = le.inverse_transform(pred_encoded)
        test['Predicted_Model'] = pred_labels

        def get_pred_value(row):
            return row.get(f"{row['Predicted_Model']}_pred", np.nan)

        test['Switch_Prediction'] = test.apply(get_pred_value, axis=1)

        # --- FIX: Honest Soft Voting (Super Ensemble) ---
        probas = clf.predict_proba(X_test)
        weighted_preds = np.zeros(len(test))

        # probas columns correspond to classes 0, 1, ..., num_classes_now-1
        # which map to le.classes_
        for idx in range(num_classes_now):
            model_name = le.classes_[idx]
            pred_col = f"{model_name}_pred"
            if pred_col in test.columns:
                weighted_preds += (test[pred_col].values * probas[:, idx])

        test['Super_Ensemble_pred'] = weighted_preds
        results.append(test)

        # Save the model and map from the FINAL iteration (which has the most complete history)
        if year == LAST_HISTORICAL_YEAR:
            final_model = clf
            # Create the map based on the final encoder
            label_map = dict(zip(le.classes_, le.transform(le.classes_)))
            final_label_map = {k: int(v) for k, v in label_map.items()}

    df_res = pd.concat(results)
    mae_switch = mean_absolute_error(df_res['kreisYield'], df_res['Switch_Prediction'])
    mae_trend = mean_absolute_error(df_res['kreisYield'], df_res['Statistical_Trend_pred'])
    mae_ens = mean_absolute_error(df_res['kreisYield'], df_res['Super_Ensemble_pred'])

    logging.info("\n" + "=" * 60)
    logging.info(f"META-LEARNER v4 RESULTS ({WALK_FORWARD_START_YEAR}-{LAST_HISTORICAL_YEAR})")
    logging.info("=" * 60)
    logging.info(f"Trend MAE:        {mae_trend:.4f}")
    logging.info(f"Hard Switch MAE:  {mae_switch:.4f}")
    logging.info(f"Super Ensemble MAE: {mae_ens:.4f} (Honest Walk-Forward)")

    # Save honest predictions
    honesty_path = OUTPUT_DIR / 'super_ensemble_walkforward_predictions.csv'
    df_res.to_csv(honesty_path, index=False)
    logging.info(f"✓ Honest predictions saved to {honesty_path}")

    if final_model:
        final_model.save_model(OUTPUT_DIR / MODEL_FILENAME)
        with open(OUTPUT_DIR / LABEL_MAP_FILENAME, 'w') as f:
            json.dump(final_label_map, f, indent=4)
        logging.info(f"✓ Model and Label Map saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    train_classifier()