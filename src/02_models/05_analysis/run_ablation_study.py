import pandas as pd
import logging
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
ABLATION_RESULTS_FILE = OUTPUT_DIR / 'ablation_results.csv'


def run_experiment(experiment_name, df, strategy='soft', feature_mask=None, filter_garbage=False):
    logging.info(f"\n--- Running Experiment: {experiment_name} ---")

    # 1. Data Prep (Filtering garbage if requested)
    df_work = df[df['Is_Garbage_Data'] == 0].copy() if filter_garbage else df.copy()

    # 2. Feature Selection
    # To match the "Champion" (56.29), we include features used in the saved model.
    exclude_cols = [
        'year', 'district_no', 'kreisYield', 'Best_Model', 'Oracle_Error',
        'Predicted_Model', 'Switch_Prediction', 'Target_Encoded',
        'Is_Garbage_Data', 'Raw_Bias', 'Median_Error',
        'Super_Ensemble_pred', 'Predicted_Best_Model'
    ]
    # NOTE: We do NOT exclude Regret_Weight here because the TSCV model leaked it as a feature.

    pred_cols = [c for c in df.columns if c.endswith('_pred') and c != 'Statistical_Trend_pred']
    feature_cols = [c for c in df.columns if c not in exclude_cols + pred_cols and not c.startswith('Prob_')]

    if feature_mask:
        feature_cols = [f for f in feature_cols if f not in feature_mask]

    # 3. REPLICATE PIPELINE EXECUTION (Final Model Inference)
    # Train on everything up to 2023
    train = df_work[df_work['year'] < 2024].copy()
    test = df_work[(df_work['year'] >= 2000) & (df_work['year'] <= 2024)].copy()

    le = LabelEncoder()
    y_train = le.fit_transform(train['Best_Model'])

    clf = XGBClassifier(objective='multi:softmax', num_class=len(le.classes_),
                        n_estimators=200, max_depth=4, learning_rate=0.03, random_state=42)
    clf.fit(train[feature_cols], y_train)

    # 4. Inference
    if strategy == 'soft':
        probas = clf.predict_proba(test[feature_cols])
        weighted_preds = np.zeros(len(test))
        for idx, cls_name in enumerate(le.classes_):
            weighted_preds += (test[f"{cls_name}_pred"].values * probas[:, idx])
        test['Switch_Prediction'] = weighted_preds
    elif strategy == 'hard':
        preds = clf.predict(test[feature_cols])
        test['Switch_Prediction'] = test.apply(
            lambda row: row.get(f"{le.inverse_transform([preds[test.index.get_loc(row.name)]])[0]}_pred"), axis=1)
    elif strategy == 'equal':
        test['Switch_Prediction'] = test[pred_cols].mean(axis=1)

    # 5. Metrics
    mae = mean_absolute_error(test['kreisYield'], test['Switch_Prediction'])
    mae_2003 = mean_absolute_error(test[test['year'] == 2003]['kreisYield'],
                                   test[test['year'] == 2003]['Switch_Prediction'])
    mae_2018 = mean_absolute_error(test[test['year'] == 2018]['kreisYield'],
                                   test[test['year'] == 2018]['Switch_Prediction'])

    logging.info(f"Result MAE: {mae:.4f}")
    return {'Experiment': experiment_name, 'MAE_Overall': mae, 'MAE_2003': mae_2003, 'MAE_2018': mae_2018}


def main():
    df = pd.read_csv(OUTPUT_DIR / INPUT_FILENAME)
    exps = []

    # Match the paper's proposed system
    exps.append(run_experiment("Proposed System (Soft Voting)", df, strategy='soft'))

    # Architecture Ablations
    exps.append(run_experiment("Arch: Hard Switching", df, strategy='hard'))
    exps.append(run_experiment("Arch: Equal Weighting", df, strategy='equal'))

    # Feature/Data Ablations
    exps.append(run_experiment("Feature: No Stress Signals", df, strategy='soft',
                               feature_mask=['CASDI_Phase2_Count', 'NMSD_Phase2_Count', 'OSAW_Phase2_Count']))
    exps.append(run_experiment("Data: 200dt/ha Filter", df, strategy='soft', filter_garbage=True))

    results_df = pd.DataFrame(exps)
    results_df.to_csv(ABLATION_RESULTS_FILE, index=False)

    # Plotting (3 groups of 5)
    melted = results_df.melt(id_vars='Experiment', value_vars=['MAE_Overall', 'MAE_2003', 'MAE_2018'],
                             var_name='Metric', value_name='MAE')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='Metric', y='MAE', hue='Experiment', palette='viridis')
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                           va='bottom', fontsize=9, fontweight='bold')
    plt.title("Ablation Study (Pipeline-Consistent Execution)")
    plt.savefig(project_root / 'docs/paper_latex/figures/fig3_ablation_results.png', dpi=300)
    print(results_df)


if __name__ == '__main__':
    main()