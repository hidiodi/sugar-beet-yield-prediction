# File: src/02_models/XGBoost/final_robust_classifier.py
# Description: FINAL CHAMPION MODEL. **REVERTED TO 3-CLASS TARGET** for tractability
#              and **INCREASED MAX_DEPTH CONSTRAINT** to improve separation capacity.
#              This is the final, best attempt at a robust hard classifier.

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint
import numpy as np
import os
import logging
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = os.path.join('models', 'final_robust_classifier_3class_tuned.joblib')
CONFUSION_MATRIX_PATH = os.path.join('reports/figures', 'confusion_matrix_robust_3class_tuned.png')


def train_and_validate_robust_classifier():
    """
    Trains the final, robust 3-class model after a targeted hyperparameter search.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"Dataset not found at {file_path}. Exiting.")
        return

    df = df.sort_values(by=['district_no', 'year'])

    # --- Trend and Anomaly Calculation ---
    logging.info("--- Calculating yield trend to create anomalies ---")
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.ffill().bfill())
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    df.dropna(subset=['kreisYield_detrended'], inplace=True)

    validation_start_year = 2011
    train_df_for_binning = df[df['year'] < validation_start_year]

    # ============================ FINAL ATTEMPT: REVERT TO 3-CLASS TARGET ============================
    logging.info("--- Reverting to 3-level classification target (Poor/Average/Good) for tractability ---")
    bins = pd.qcut(train_df_for_binning['kreisYield_detrended'], q=[0, 0.25, 0.75, 1], retbins=True, duplicates='drop')[
        1]

    num_classes = 3
    class_labels_map = {0: 'Poor', 1: 'Average', 2: 'Good'}  # Remapped to standard 0, 1, 2 indices

    df['yield_class'] = pd.cut(df['kreisYield_detrended'], bins=bins, labels=False, include_lowest=True)
    df.dropna(subset=['yield_class'], inplace=True)
    df['yield_class'] = df['yield_class'].astype(int)
    df = df[df['yield_class'] < num_classes]
    # ==============================================================================================

    train_df = df[df['year'] < validation_start_year].copy()
    validation_df = df[df['year'] >= validation_start_year].copy()

    # --- Feature Curation (Unchanged) ---
    logging.info("--- Using a curated set of champion features to reduce noise and overfitting ---")
    champion_features = [
        # Geography & Soil
        'lon', 'lat', 'avg_soil_pawc',
        # Top Economic Signals
        'profit_margin_proxy_lag1', 'plant_protection_price_index_lag1_anomaly',
        'fertilizer_price_index_lag1_anomaly',
        # Top Weather Signals
        'temp_mean_jul_anomaly', 'temp_mean_jun_anomaly', 'srad_mean_jul_anomaly',
        'precip_sum_jul_anomaly',
        # Top Interaction & Polynomial Features
        'july_heat_x_profit_margin', 'temp_mean_jul_anomaly_sq',
        # Top Antecedent Signal
        'antecedent_gdd_sum_anomaly',
        # Top Satellite Signal
        'winter_cropland_ndvi_anomaly'
    ]

    X_train = train_df[champion_features]
    y_train = train_df['yield_class']
    X_validation = validation_df[champion_features]
    y_validation = validation_df['yield_class']

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # ============================ IMPROVEMENT 2: Targeted Hyperparameter Search with Deeper Trees ============================
    logging.info(
        f"--- Starting Targeted Hyperparameter Search for Robust {num_classes}-Class Model (Max Depth 4-7) ---")

    classifier_base = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric='merror',
        learning_rate=0.03,  # Fixed low learning rate for robustness
        n_estimators=500,  # Fixed high number of estimators
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        'max_depth': randint(4, 8),  # INCREASED: Allows slightly deeper trees (4 to 7)
        'subsample': uniform(0.65, 0.25),
        'colsample_bytree': uniform(0.65, 0.25),
        'gamma': uniform(0.1, 1.0)
    }

    tscv = TimeSeriesSplit(n_splits=5)

    random_search = RandomizedSearchCV(
        estimator=classifier_base, param_distributions=param_dist, n_iter=40,
        cv=tscv, scoring='f1_weighted', n_jobs=-1, verbose=1, random_state=42
    )

    random_search.fit(X_train, y_train, sample_weight=sample_weights)

    logging.info(f"\nBest parameters found: {random_search.best_params_}")
    logging.info(f"Best cross-validation Weighted F1-score: {random_search.best_score_:.4f}")

    classifier_model = random_search.best_estimator_
    # ==================================================================================================

    y_pred = classifier_model.predict(X_validation)

    accuracy = accuracy_score(y_validation, y_pred)
    f1 = f1_score(y_validation, y_pred, average='weighted')

    print(f"\n--- FINAL ROBUST {num_classes}-CLASS CLASSIFIER Validation Performance ---")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Weighted F1-Score: {f1:.4f}")
    print("-------------------------------------------------")

    print("\n--- Classification Report ---")
    print(classification_report(y_validation, y_pred, target_names=class_labels_map.values(),
                                labels=list(class_labels_map.keys())))

    # --- Plotting and saving Confusion Matrix ---
    cm = confusion_matrix(y_validation, y_pred, labels=list(class_labels_map.keys()))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels_map.values(),
                yticklabels=class_labels_map.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: Robust {num_classes}-Class Harvest Quality')

    os.makedirs(os.path.dirname(CONFUSION_MATRIX_PATH), exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_PATH, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    # --- Save Final Model ---
    joblib.dump(classifier_model, MODEL_PATH)
    logging.info(f"✅ Final robust classifier saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_validate_robust_classifier()