import pandas as pd
import logging
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
import sys
import numpy as np
import json

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])

INPUT_FILENAME = 'super_ensemble_training_data.csv'
MODEL_FILENAME = 'super_ensemble_meta_learner_TSCV.json'
LABEL_MAP_FILENAME = 'meta_learner_label_map_TSCV.json'

WALK_FORWARD_START_YEAR = 2015
LAST_HISTORICAL_YEAR = 2024


def train_classifier():
    input_path = OUTPUT_DIR / INPUT_FILENAME
    if not input_path.exists(): return

    logging.info("--- Training Meta-Learner (v2: Consensus-Aware) ---")
    df = pd.read_csv(input_path)

    # Feature Selection
    exclude_cols = ['year', 'district_no', 'kreisYield', 'Best_Model', 'Oracle_Error', 'Predicted_Model',
                    'Switch_Prediction', 'Target_Encoded']
    pred_cols = [c for c in df.columns if c.endswith('_pred') and c != 'Statistical_Trend_pred']

    feature_cols = [c for c in df.columns if c not in exclude_cols + pred_cols]

    # Log key features
    logging.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # Target
    le = LabelEncoder()
    df['Target_Encoded'] = le.fit_transform(df['Best_Model'])
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    label_map_json = {k: int(v) for k, v in label_map.items()}

    # Walk-Forward
    results = []
    final_model = None

    for year in range(WALK_FORWARD_START_YEAR, LAST_HISTORICAL_YEAR + 1):
        train = df[df['year'] < year]
        test = df[df['year'] == year].copy()

        if train.empty or test.empty: continue

        X_train = train[feature_cols]
        y_train = train['Target_Encoded']
        X_test = test[feature_cols]

        # Tuned Hyperparameters for better generalization
        clf = XGBClassifier(
            objective='multi:softmax',
            num_class=len(label_map),
            n_estimators=200,  # More trees (was 150)
            max_depth=4,  # Keep depth 4 for interactions
            learning_rate=0.03,  # Slower learning (was 0.04)
            subsample=0.75,  # More randomness to prevent overfitting
            colsample_bytree=0.75,
            gamma=0.1,  # Slight regularization
            n_jobs=-1,
            random_state=42
        )

        clf.fit(X_train, y_train)

        pred_encoded = clf.predict(X_test)
        pred_labels = le.inverse_transform(pred_encoded)
        test['Predicted_Model'] = pred_labels

        def get_pred_value(row):
            model_col = f"{row['Predicted_Model']}_pred"
            return row.get(model_col, np.nan)

        test['Switch_Prediction'] = test.apply(get_pred_value, axis=1)
        results.append(test)

        if year == LAST_HISTORICAL_YEAR:
            final_model = clf

    # Evaluation
    df_res = pd.concat(results)
    acc = accuracy_score(df_res['Target_Encoded'], le.transform(df_res['Predicted_Model']))
    mae_switch = mean_absolute_error(df_res['kreisYield'], df_res['Switch_Prediction'])
    mae_trend = mean_absolute_error(df_res['kreisYield'], df_res['Statistical_Trend_pred'])

    logging.info("\n" + "=" * 60)
    logging.info(f"META-LEARNER v2 RESULTS ({WALK_FORWARD_START_YEAR}-{LAST_HISTORICAL_YEAR})")
    logging.info("=" * 60)
    logging.info(f"Accuracy: {acc:.2%}")
    logging.info(f"Trend MAE:        {mae_trend:.4f}")
    logging.info(f"Hard Switch MAE:  {mae_switch:.4f}")
    logging.info(f"Improvement:      {mae_trend - mae_switch:+.4f}")

    if final_model:
        final_model.save_model(OUTPUT_DIR / MODEL_FILENAME)
        with open(OUTPUT_DIR / LABEL_MAP_FILENAME, 'w') as f:
            json.dump(label_map_json, f, indent=4)


if __name__ == '__main__':
    train_classifier()