import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
import json
import joblib

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib

analysis_config = importlib.import_module("src.03_analysis.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
CONFIG = analysis_config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])

INPUT_FILENAME = 'super_ensemble_training_data.csv'
METADATA_FILENAME = 'super_ensemble_weights.json'
MODEL_FILENAME = 'super_ensemble_meta_model.joblib'

WALK_FORWARD_START_YEAR = 2000
LAST_HISTORICAL_YEAR = 2024


def train_meta_regressor():
    logging.info("--- Training Meta-Classifier (Soft Voting) ---")

    input_path = OUTPUT_DIR / INPUT_FILENAME
    if not input_path.exists(): return

    df = pd.read_csv(input_path)

    # Features: Context (Teleconnections) + Model Consensus (Diff from Trend)
    context_cols = ['enso_mei_winter_avg', 'nao_winter_avg', 'sca_winter_avg',
                    'effective_winter_water', 'trafficability_stress']

    # Add "Disagreement" features (Model - Trend)
    # This helps the classifier know when the models are diverging
    base_models = ['Hybrid_XGB_pred', 'Robust_Linear_pred', 'V31_Solar_Gated_pred']
    base_models = [m for m in base_models if m in df.columns]

    feature_cols = []
    for model in base_models:
        diff_col = f"{model}_diff"
        df[diff_col] = df[model] - df['Statistical_Trend_pred']
        feature_cols.append(diff_col)

    feature_cols.extend([c for c in context_cols if c in df.columns])

    # Target: Best Model Label
    df = df.dropna(subset=['Best_Model'])

    logging.info(f"Features: {feature_cols}")
    logging.info(f"Classes: {df['Best_Model'].unique()}")

    results = []

    # Walk-Forward Validation
    for year in range(WALK_FORWARD_START_YEAR, LAST_HISTORICAL_YEAR + 1):
        train = df[df['year'] < year].copy()
        test = df[df['year'] == year].copy()

        if train.empty or test.empty: continue

        # Classifier: Predict PROBABILITY of each model being the winner
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=5,
                                          class_weight='balanced', random_state=42))
        ])

        clf.fit(train[feature_cols], train['Best_Model'])

        # Get Probabilities
        probs = clf.predict_proba(test[feature_cols])
        classes = clf.classes_

        # Soft Voting Blend
        # Final Pred = Sum(Prob(Model) * Model_Pred)
        final_pred = np.zeros(len(test))
        for i, model_name in enumerate(classes):
            if model_name in test.columns:
                final_pred += probs[:, i] * test[model_name].values
            else:
                # Fallback to Trend if model missing
                final_pred += probs[:, i] * test['Statistical_Trend_pred'].values

        test['Super_Ensemble_pred'] = final_pred

        # Label the prediction with the highest probability model (for analysis)
        best_idx = np.argmax(probs, axis=1)
        test['Predicted_Best_Model'] = [classes[i] for i in best_idx]

        results.append(test)

    df_res = pd.concat(results)

    # --- Evaluation ---
    mae_trend = mean_absolute_error(df_res['kreisYield'], df_res['Statistical_Trend_pred'])
    mae_ens = mean_absolute_error(df_res['kreisYield'], df_res['Super_Ensemble_pred'])
    skill = (1 - (mae_ens / mae_trend)) * 100
    acc = accuracy_score(df_res['Best_Model'], df_res['Predicted_Best_Model'])

    logging.info("\n" + "=" * 60)
    logging.info(f"CLASSIFIER ENSEMBLE RESULTS ({WALK_FORWARD_START_YEAR}-{LAST_HISTORICAL_YEAR})")
    logging.info("=" * 60)
    logging.info(f"Trend MAE:          {mae_trend:.4f}")
    logging.info(f"Super Ensemble MAE: {mae_ens:.4f}")
    logging.info(f"Skill Improvement:  {skill:.4f}%")
    logging.info(f"Classification Acc: {acc:.2%}")

    # Recent Volatility
    df_recent = df_res[df_res['year'] >= 2010]
    if not df_recent.empty:
        mae_trend_rec = mean_absolute_error(df_recent['kreisYield'], df_recent['Statistical_Trend_pred'])
        mae_ens_rec = mean_absolute_error(df_recent['kreisYield'], df_recent['Super_Ensemble_pred'])
        skill_rec = (1 - (mae_ens_rec / mae_trend_rec)) * 100
        logging.info(f"RECENT VOLATILITY Skill: {skill_rec:.4f}%")

    # --- Train Final Production Model ---
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=6,
                                      class_weight='balanced', random_state=42))
    ])
    final_model.fit(df[feature_cols], df['Best_Model'])

    # Save Outputs
    joblib.dump(final_model, OUTPUT_DIR / MODEL_FILENAME)
    df_res.to_csv(OUTPUT_DIR / 'super_ensemble_final_forecast_TSCV.csv', index=False)

    metadata = {
        "strategy": "Soft Voting Classifier (Random Forest)",
        "features": feature_cols,
        "classes": list(final_model.classes_),
        "skill_recent": skill_rec
    }
    with open(OUTPUT_DIR / METADATA_FILENAME, 'w') as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"\n✓ Saved Classifier Meta-Model to {OUTPUT_DIR / MODEL_FILENAME}")


if __name__ == '__main__':
    train_meta_regressor()