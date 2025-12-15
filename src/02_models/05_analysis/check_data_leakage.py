import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder  # <--- Added

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib
analysis_config = importlib.import_module("src.03_analysis.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
OUTPUT_DIR = Path(analysis_config.MODEL_COMPARISON_CONFIG['OUTPUT_DIR'])
DATA_FILE = OUTPUT_DIR / 'super_ensemble_training_data.csv'


def run_forensic_audit():
    logging.info("--- 🕵️ COMPONENT INTEGRITY & LEAKAGE AUDIT V2 (Fixed) ---")

    if not DATA_FILE.exists(): return
    df = pd.read_csv(DATA_FILE)

    # ---------------------------------------------------------
    # TEST 1: The "Perfect Match" Trap (Exact Float Matching)
    # ---------------------------------------------------------
    logging.info("\n[TEST 1] Exact Match Scan")
    components = [c for c in df.columns if c.endswith('_pred')]

    for comp in components:
        # Check for error < 0.0001 (Float tolerance)
        perfect_matches = df[abs(df['kreisYield'] - df[comp]) < 0.0001]
        if len(perfect_matches) > 0:
            logging.error(f"  ❌ FATAL: {comp} has {len(perfect_matches)} EXACT matches with actual yield.")
        else:
            logging.info(f"  ✅ {comp}: No exact matches found.")

    # ---------------------------------------------------------
    # TEST 2: Temporal Stationarity (The "Too Stable" Test)
    # ---------------------------------------------------------
    logging.info("\n[TEST 2] Component Error Stationarity")
    df['Block'] = pd.cut(df['year'], bins=[1999, 2010, 2018, 2024], labels=['Early', 'Mid', 'Late'])

    for comp in components:
        means = []
        for block in ['Early', 'Mid', 'Late']:
            subset = df[df['Block'] == block]
            if not subset.empty:
                means.append(mean_absolute_error(subset['kreisYield'], subset[comp]))

        spread = max(means) - min(means)
        if spread < 5.0:
            logging.warning(f"  ⚠️ {comp}: SUSPICIOUSLY STABLE (Spread {spread:.1f}).")
        else:
            logging.info(f"  ✅ {comp}: Healthy volatility (Spread {spread:.1f}).")

    # ---------------------------------------------------------
    # TEST 3: Meta-Learner Driver Analysis (FIXED)
    # ---------------------------------------------------------
    logging.info("\n[TEST 3] Meta-Learner Driver Analysis")

    # --- FIX START: Create Target_Encoded on the fly ---
    if 'Best_Model' not in df.columns:
        logging.error("Column 'Best_Model' missing. Cannot run feature analysis.")
        return

    le = LabelEncoder()
    # Filter out rows where Best_Model might be NaN (if any)
    df_clean = df.dropna(subset=['Best_Model']).copy()
    df_clean['Target_Encoded'] = le.fit_transform(df_clean['Best_Model'])
    target_col = 'Target_Encoded'
    # --- FIX END ---

    exclude_cols = ['year', 'district_no', 'kreisYield', 'Best_Model', 'Oracle_Error', 'Predicted_Model',
                    'Switch_Prediction', 'Target_Encoded', 'Block', 'District_Name'] + components
    features = [c for c in df_clean.columns if c not in exclude_cols]

    # Train/Test split
    train = df_clean[df_clean['year'] < 2020].copy()
    test = df_clean[df_clean['year'] >= 2020].copy()

    clf = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
    clf.fit(train[features], train[target_col])

    baseline_acc = clf.score(test[features], test[target_col])
    logging.info(f"  Baseline Accuracy: {baseline_acc:.1%}")

    logging.info("  Top 5 Most Important Features:")
    imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False).head(5)
    logging.info(imp.to_string())

    if 'Oracle_Error' in imp.index or any('Error_' in i for i in imp.index):
        logging.error("  ❌ LEAK DETECTED: Error columns used as features!")

    # ---------------------------------------------------------
    # TEST 4: Regime Bias Check
    # ---------------------------------------------------------
    logging.info("\n[TEST 4] Regime Bias Audit")
    preds = clf.predict(test[features])
    unique, counts = np.unique(preds, return_counts=True)

    # Map back to names
    class_names = le.inverse_transform(unique)
    dist = dict(zip(class_names, counts))

    logging.info(f"  Predicted Class Distribution: {dist}")
    if len(dist) == 1:
        logging.warning("  ⚠️ Model collapsed to single class.")
    else:
        logging.info("  ✅ Model is actively switching.")


if __name__ == "__main__":
    run_forensic_audit()