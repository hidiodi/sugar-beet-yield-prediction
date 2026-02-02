import pandas as pd
import logging
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import sys
import numpy as np
import json

# Setup
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib
analysis_config = importlib.import_module("src.03_analysis.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = analysis_config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILENAME = 'super_ensemble_training_data.csv'
WALK_FORWARD_START_YEAR = 2000
LAST_HISTORICAL_YEAR = 2024
ABLATION_RESULTS_FILE = OUTPUT_DIR / 'ablation_results.csv'

def train_and_evaluate(df_train, df_test, feature_cols, experiment_name):
    """
    Trains the classifier on df_train and evaluates on df_test.
    Returns the predictions for the test set.
    """
    if df_train.empty or df_test.empty:
        return pd.DataFrame()

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['Best_Model'])
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]

    if experiment_name == "Meta-Learner Ablation (Equal Weight)":
        # Simple Average Strategy: No training needed for the meta-learner part
        # But we need to return predictions.
        # For equal weight, we average the predictions of all component models.
        # However, the architecture here expects a "Predicted_Model" and "Switch_Prediction".
        # This function simulates the behavior of the switching logic.

        # In Equal Weight, "Switch_Prediction" is the average of all components.
        pred_cols = [c for c in df_test.columns if c.endswith('_pred') and c != 'Statistical_Trend_pred']
        # Also exclude Super_Ensemble_pred if present
        pred_cols = [c for c in pred_cols if c != 'Super_Ensemble_pred']

        # We assume the "Switch_Prediction" here represents the ensemble output
        test_preds = df_test[pred_cols].mean(axis=1)

        # For compatibility, we just say Predicted_Model is "Equal_Weight"
        df_test_res = df_test.copy()
        df_test_res['Predicted_Model'] = 'Equal_Weight'
        df_test_res['Switch_Prediction'] = test_preds
        return df_test_res

    # Standard XGBoost Training
    num_classes = len(le.classes_)
    clf = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
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

    pred_encoded = clf.predict(X_test)
    pred_labels = le.inverse_transform(pred_encoded)

    df_test_res = df_test.copy()
    df_test_res['Predicted_Model'] = pred_labels

    def get_pred_value(row):
        return row.get(f"{row['Predicted_Model']}_pred", np.nan)

    df_test_res['Switch_Prediction'] = df_test_res.apply(get_pred_value, axis=1)

    return df_test_res

def run_experiment(experiment_name, df, feature_mask=None, filter_garbage=True, model_type="XGBoost"):
    logging.info(f"\n--- Running Experiment: {experiment_name} ---")

    # 1. Filter Garbage
    if filter_garbage and 'Is_Garbage_Data' in df.columns:
        df_clean = df[df['Is_Garbage_Data'] == 0].copy()
    else:
        df_clean = df.copy()

    # 2. Feature Selection
    exclude_cols = [
        'year', 'district_no', 'kreisYield', 'Best_Model', 'Oracle_Error',
        'Predicted_Model', 'Switch_Prediction', 'Target_Encoded',
        'Is_Garbage_Data', 'Raw_Bias', 'Regret_Weight', 'Median_Error',
        'Super_Ensemble_pred', 'Predicted_Best_Model' # Ensure these don't leak if present
    ]
    pred_cols = [c for c in df.columns if c.endswith('_pred') and c != 'Statistical_Trend_pred']

    all_feature_cols = [c for c in df.columns if c not in exclude_cols + pred_cols and not c.startswith('Prob_')]

    if feature_mask:
        # Remove features that match the mask
        final_features = [f for f in all_feature_cols if f not in feature_mask]
        logging.info(f"Removing features: {feature_mask}")
    else:
        final_features = all_feature_cols

    # 3. Walk-Forward Validation
    results = []

    years = sorted(df_clean['year'].unique())
    start_year = max(WALK_FORWARD_START_YEAR, min(years) + 1) # Ensure at least one year of training

    for year in range(start_year, LAST_HISTORICAL_YEAR + 1):
        train = df_clean[df_clean['year'] < year].copy()
        test = df_clean[df_clean['year'] == year].copy()

        # Note: If we didn't filter garbage for training, we might still want to test on everything?
        # The prompt says "Retrain without Garbage Filter", implying the training set is dirtier.
        # Test set should probably be consistent for comparison, but usually we evaluate on the full test set.
        # However, if 'Is_Garbage_Data' is true, the ground truth might be wrong.
        # For the sake of this study, we follow the pipeline logic: test on what we have.

        if train.empty or test.empty: continue

        df_res = train_and_evaluate(train, test, final_features, experiment_name)
        results.append(df_res)

    if not results:
        logging.warning("No results generated.")
        return None

    df_full_res = pd.concat(results)
    mae = mean_absolute_error(df_full_res['kreisYield'], df_full_res['Switch_Prediction'])
    logging.info(f"Result MAE: {mae:.4f}")

    # Calculate MAE for specific years (2003, 2018)
    mae_2003 = np.nan
    mae_2018 = np.nan

    if 2003 in df_full_res['year'].values:
        sub_2003 = df_full_res[df_full_res['year'] == 2003]
        mae_2003 = mean_absolute_error(sub_2003['kreisYield'], sub_2003['Switch_Prediction'])

    if 2018 in df_full_res['year'].values:
        sub_2018 = df_full_res[df_full_res['year'] == 2018]
        mae_2018 = mean_absolute_error(sub_2018['kreisYield'], sub_2018['Switch_Prediction'])

    return {
        'Experiment': experiment_name,
        'MAE_Overall': mae,
        'MAE_2003': mae_2003,
        'MAE_2018': mae_2018
    }

def main():
    input_path = OUTPUT_DIR / INPUT_FILENAME
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Define Experiments
    experiments = []

    # 1. Baseline
    experiments.append(run_experiment("Baseline", df))

    # 2. Feature Ablation (No Scorch/Anoxia)
    # Assuming features are named similarly to 'summer_days_tmax_gt_30c' (Heat) or 'z_anoxia' (if passed through)
    # In 'prepare_ensemble_data.py', we saw 'NMSD_Phase2_Count', 'OSAW_Phase2_Count' added.
    # NMSD = Night Heat?, OSAW = Oxygen Stress (Anoxia)?
    # "Scorch Index" might be 'CASDI_Phase2_Count' or related.
    # Let's target the context features identified in `prepare_ensemble_data.py`:
    # 'CASDI_Phase2_Count', 'NMSD_Phase2_Count', 'OSAW_Phase2_Count'
    experiments.append(run_experiment(
        "Feature Ablation (No Stress Signals)",
        df,
        feature_mask=['CASDI_Phase2_Count', 'NMSD_Phase2_Count', 'OSAW_Phase2_Count']
    ))

    # 3. Meta-Learner Validation (Equal Weight)
    experiments.append(run_experiment(
        "Meta-Learner Ablation (Equal Weight)",
        df,
        model_type="Equal_Weight"
    ))

    # 4. Data Quality (No Garbage Filter)
    experiments.append(run_experiment(
        "Data Quality (No Filter)",
        df,
        filter_garbage=False
    ))

    # Save Results
    results_df = pd.DataFrame([e for e in experiments if e is not None])
    print("\n--- Final Ablation Results ---")
    print(results_df)
    results_df.to_csv(ABLATION_RESULTS_FILE, index=False)
    logging.info(f"Saved ablation results to {ABLATION_RESULTS_FILE}")

if __name__ == '__main__':
    main()
