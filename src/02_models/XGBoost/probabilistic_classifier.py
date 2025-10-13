# File: src/02_models/XGBoost/final_tuned_probabilistic_classifier.py
# Description: FINAL TUNED PROBABILISTIC MODEL. **UPDATED TO INCLUDE POST-HOC CALIBRATION**
#              AND **A VISUALIZATION PLOT** comparing the predicted class-mean yield
#              against the actual yield for all validation instances.

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, classification_report, confusion_matrix
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import r2_score, mean_squared_error  # <-- Added for plotting metrics
from scipy.stats import uniform, randint
import numpy as np
import os
import logging
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Corrected logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Best parameters found in the log:
BEST_PARAMS = {
    'colsample_bytree': 0.6203074124157587,
    'gamma': 0.44330857447532995,
    'learning_rate': 0.011104670874948189,
    'max_depth': 4,
    'n_estimators': 350,
    'subsample': 0.786239207252984
}

PREDICTION_PLOT_PATH = os.path.join('reports/figures', 'classification_predicted_vs_actual.png')


def plot_predicted_vs_actual(df_results, num_classes, title="Predicted Yield vs. Actual Yield (Classification Model)"):
    """
    Creates a scatter plot comparing the assigned mean predicted yield (from classification)
    against the actual yield for the validation set.
    """
    logging.info("Generating Predicted vs. Actual Yield plot...")

    # Calculate R2 and RMSE for the displayed "predicted" yield (class mean)
    r2 = r2_score(df_results['kreisYield'], df_results['predicted_yield_proxy'])
    rmse = np.sqrt(mean_squared_error(df_results['kreisYield'], df_results['predicted_yield_proxy']))

    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x='kreisYield',
        y='predicted_yield_proxy',
        hue='yield_class',
        data=df_results,
        palette='viridis',
        alpha=0.6,
        s=40
    )

    # Add the diagonal line for perfect prediction (y=x)
    max_val = max(df_results['kreisYield'].max(), df_results['predicted_yield_proxy'].max())
    min_val = min(df_results['kreisYield'].min(), df_results['predicted_yield_proxy'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

    # Add text box for metrics
    textstr = f'$R^2$ (Proxy): {r2:.4f}\nRMSE (Proxy): {rmse:.2f} dt/ha'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

    plt.title(title, fontsize=12)
    plt.xlabel('Actual Yield (dt/ha)', fontsize=10)
    plt.ylabel('Predicted Class-Mean Yield (dt/ha)', fontsize=10)
    plt.legend(title=f'Actual Yield Class ({num_classes})', loc='lower right')
    plt.grid(True)

    os.makedirs(os.path.dirname(PREDICTION_PLOT_PATH), exist_ok=True)
    plt.savefig(PREDICTION_PLOT_PATH, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ Predicted vs. Actual yield plot saved to {PREDICTION_PLOT_PATH}")


def train_and_validate_tuned_probabilistic_classifier():
    """
    Trains, tunes, and applies post-hoc calibration to a 4-class model, and generates a
    predicted vs. actual plot.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"Dataset not found at {file_path}. Exiting.")
        return

    df = pd.read_csv(file_path)
    df = df.sort_values(by=['district_no', 'year'])

    # --- Feature Engineering & Detrending ---
    logging.info("--- Calculating yield trend to create anomalies ---")
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean())
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.ffill().bfill())
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    df.dropna(subset=['kreisYield_detrended'], inplace=True)

    validation_start_year = 2011
    # Used only for quantile boundary determination
    train_df_for_binning_raw = df[df['year'] < validation_start_year]

    # --- 4-Level Classification Target & Class Mean Calculation ---
    logging.info("--- Creating 4-level classification target (Very Poor/Poor/Average/Good) ---")
    quantiles = [0, 0.20, 0.50, 0.80, 1]

    try:
        bins = pd.qcut(train_df_for_binning_raw['kreisYield_detrended'], q=quantiles, retbins=True, duplicates='drop')[
            1]
    except ValueError as e:
        quantiles = [0, 0.25, 0.75, 1]
        bins = pd.qcut(train_df_for_binning_raw['kreisYield_detrended'], q=quantiles, retbins=True, duplicates='drop')[
            1]
        logging.warning("Falling back to 3 classes due to binning issues.")

    num_classes = len(bins) - 1
    class_labels_map = {0: 'Very Poor', 1: 'Poor', 2: 'Average', 3: 'Good'}
    class_labels_map = {i: class_labels_map.get(i, f'Class {i}') for i in range(num_classes)}

    # 1. CREATE 'yield_class' on the full DataFrame 'df'
    df['yield_class'] = pd.cut(df['kreisYield_detrended'], bins=bins, labels=False, include_lowest=True)
    df.dropna(subset=['yield_class'], inplace=True)
    df['yield_class'] = df['yield_class'].astype(int)
    df = df[df['yield_class'] < num_classes]

    # 2. DEFINE TRAIN/VALIDATION SETS (which now include 'yield_class')
    train_df = df[df['year'] < validation_start_year].copy()
    validation_df = df[df['year'] >= validation_start_year].copy()

    # 3. CALCULATE CLASS MEANS using the correct DataFrame 'train_df'
    class_mean_detrended_yields = train_df.groupby('yield_class')['kreisYield_detrended'].mean().to_dict()
    logging.info(f"Class Mean Detrended Yields: {class_mean_detrended_yields}")

    champion_features = [
        'lon', 'lat', 'avg_soil_pawc', 'profit_margin_proxy_lag1',
        'plant_protection_price_index_lag1_anomaly', 'fertilizer_price_index_lag1_anomaly',
        'temp_mean_jul_anomaly', 'temp_mean_jun_anomaly', 'srad_mean_jul_anomaly',
        'precip_sum_jul_anomaly', 'july_heat_x_profit_margin', 'temp_mean_jul_anomaly_sq',
        'antecedent_gdd_sum_anomaly', 'winter_cropland_ndvi_anomaly'
    ]

    X_train = train_df[champion_features]
    y_train = train_df['yield_class']
    X_validation = validation_df[champion_features]
    y_validation = validation_df['yield_class']

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # --- Training the BASE XGBoost Model with BEST PARAMS ---
    logging.info(f"\n--- Training Base XGBoost Classifier with Best Params ({num_classes} Classes) ---")

    xgb_base = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        **BEST_PARAMS
    )
    xgb_base.fit(X_train, y_train, sample_weight=sample_weights)

    # --- Raw Model Evaluation (omitted printout, calculated for calibration comparison) ---
    y_pred_proba_raw = xgb_base.predict_proba(X_validation)
    logloss_raw = log_loss(y_validation, y_pred_proba_raw)
    brier_raw = np.mean([brier_score_loss(y_validation == i, y_pred_proba_raw[:, i]) for i in range(num_classes)])

    # --- Calibrated Model Setup ---
    logging.info("\n--- Applying Post-Hoc Isotonic Regression Calibration ---")
    calibrated_model = CalibratedClassifierCV(
        estimator=xgb_base,
        method='isotonic',
        cv='prefit'
    )
    # Fit the calibrator on the OOD validation set
    calibrated_model.fit(X_validation, y_validation)

    # --- Calibrated Model Evaluation ---
    y_pred_proba_calibrated = calibrated_model.predict_proba(X_validation)
    y_pred_class = calibrated_model.predict(X_validation)
    logloss_cal = log_loss(y_validation, y_pred_proba_calibrated)
    brier_cal = np.mean(
        [brier_score_loss(y_validation == i, y_pred_proba_calibrated[:, i]) for i in range(num_classes)])

    print(f"\n--- {num_classes}-Class Probabilistic Classifier Performance Comparison ---")
    print("-------------------------------------------------------------------")
    print(f"| Metric       | Raw XGBoost (Tuned) | Calibrated (Isotonic) |")
    print("|--------------|---------------------|-----------------------|")
    print(f"| Log Loss     | {logloss_raw:.4f}          | {logloss_cal:.4f}             |")
    print(f"| Brier Score  | {brier_raw:.4f}          | {brier_cal:.4f}             |")
    print("-------------------------------------------------------------------")

    # --- Dataframe for Plotting & Visualization ---
    results_df = validation_df[['district_no', 'year', 'kreisYield', 'yield_class', 'yield_trend']].copy().reset_index(
        drop=True)
    results_df['predicted_class'] = y_pred_class

    # Map the predicted class to the Mean Detrended Yield, then re-trend it
    results_df['mean_detrended_pred'] = results_df['predicted_class'].map(class_mean_detrended_yields)
    results_df['predicted_yield_proxy'] = results_df['mean_detrended_pred'] + results_df['yield_trend']
    results_df['yield_class'] = results_df['yield_class'].map(class_labels_map)  # Map actual class for plotting Hue

    # --- PLOT GENERATION ---
    plot_predicted_vs_actual(results_df, num_classes,
                             title=f"Predicted Yield (Class Mean) vs. Actual Yield ({num_classes} Classes)")

    # --- Sample Display ---
    print("\n--- Sample of Calibrated Probabilistic Forecasts ---")
    validation_sample = validation_df[['district_no', 'year', 'kreisYield', 'yield_class']].copy().reset_index(
        drop=True)
    proba_df = pd.DataFrame(y_pred_proba_calibrated, columns=[f'P({v})' for v in class_labels_map.values()])
    validation_sample = pd.concat([validation_sample, proba_df], axis=1)
    validation_sample['True Class'] = validation_sample['yield_class'].map(class_labels_map)
    validation_sample = validation_sample.drop(columns=['yield_class'])
    print(validation_sample.head(10).to_string(index=False))
    print("-------------------------------------------------")

    # --- Calibration Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(num_classes):
        CalibrationDisplay.from_predictions(
            y_validation == i, y_pred_proba_calibrated[:, i], n_bins=10, ax=ax, name=f"{class_labels_map[i]} Class"
        )

    ax.set_title(f'Reliability Curve (Calibration Plot) - {num_classes} Classes - CALIBRATED')
    plt.grid(True)

    RELIABILITY_CURVE_PATH = os.path.join('reports/figures',
                                          f'reliability_curve_tuned_{num_classes}class_calibrated.png')
    os.makedirs(os.path.dirname(RELIABILITY_CURVE_PATH), exist_ok=True)
    plt.savefig(RELIABILITY_CURVE_PATH, bbox_inches='tight')
    logging.info(f"✅ Reliability curve (CALIBRATED) saved to {RELIABILITY_CURVE_PATH}")

    # --- Save the Calibrated Model ---
    MODEL_PATH = os.path.join('models', f'final_tuned_probabilistic_classifier_{num_classes}class_calibrated.joblib')
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(calibrated_model, MODEL_PATH)
    logging.info(f"✅ Final **calibrated** probabilistic classifier saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_validate_tuned_probabilistic_classifier()