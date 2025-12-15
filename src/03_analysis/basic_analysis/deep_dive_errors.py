import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')

# PATHS
FORECAST_FILE = Path(config.MODEL_COMPARISON_CONFIG['OUTPUT_DIR']) / 'super_ensemble_final_forecast_TSCV.csv'
TRAINING_FILE = Path(config.MODEL_COMPARISON_CONFIG['OUTPUT_DIR']) / 'super_ensemble_training_data.csv'
GEO_FILE = config.DATA_DIR / '01_raw/districts_official.geojson'


def load_district_names():
    if not GEO_FILE.exists(): return {}
    gdf = gpd.read_file(GEO_FILE)
    id_col = None
    name_col = None
    for c in ['GEN', 'name', 'NAME', 'District']:
        if c in gdf.columns: name_col = c
    for c in ['AGS', 'RS', 'id', 'district_no']:
        if c in gdf.columns and str(gdf[c].iloc[0]).isdigit(): id_col = c
    if name_col and id_col:
        return dict(zip(gdf[id_col].astype(str).str.zfill(5), gdf[name_col]))
    return {}


def define_regime(row, threshold=50):
    """Defines the 'True Regime' based on Trend Deviation."""
    # If Actual Yield is way below Trend -> CRASH
    # If Actual Yield is way above Trend -> BUMPER
    deviation = row['kreisYield'] - row['Statistical_Trend_pred']
    if deviation < -threshold: return 'Crash'
    if deviation > threshold: return 'Bumper'
    return 'Normal'


def analyze_forensic_failures():
    logging.info("--- 🕵️ FORENSIC ERROR ANALYSIS V2 (Decomposition) ---")

    # 1. Load & Merge
    df_pred = pd.read_csv(FORECAST_FILE)
    df_train = pd.read_csv(TRAINING_FILE)

    # Merge component predictions to calculate "Oracle" potential
    cols_to_merge = [c for c in df_train.columns if c.endswith('_pred') and c not in df_pred.columns]
    df = pd.merge(df_pred, df_train[['year', 'district_no'] + cols_to_merge],
                  on=['year', 'district_no'], how='left')

    names = load_district_names()
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    df['District_Name'] = df['district_no'].map(names).fillna('Unknown')

    # 2. Calculate "Oracle" (Best Possible) Prediction
    pred_cols = [c for c in df.columns if c.endswith('_pred') and c != 'Super_Ensemble_pred']

    def get_best_possible(row):
        best_err = float('inf')
        best_val = row['Statistical_Trend_pred']
        for col in pred_cols:
            if pd.isna(row[col]): continue
            err = abs(row['kreisYield'] - row[col])
            if err < best_err:
                best_err = err
                best_val = row[col]
        return best_val

    df['Oracle_Pred'] = df.apply(get_best_possible, axis=1)

    # 3. Error Decomposition
    df['Total_Error'] = (df['Super_Ensemble_pred'] - df['kreisYield']).abs()
    df['Systemic_Error'] = (df['Oracle_Pred'] - df['kreisYield']).abs()  # Even best model failed
    df['Selection_Error'] = df['Total_Error'] - df['Systemic_Error']  # Failed to pick best model

    logging.info("\n📊 ERROR DECOMPOSITION (Where did we lose points?)")
    logging.info(f"Total MAE:            {df['Total_Error'].mean():.2f}")
    logging.info(f"Oracle MAE (Ceiling): {df['Systemic_Error'].mean():.2f} (Limit of current components)")
    logging.info(f"Selection MAE (Meta): {df['Selection_Error'].mean():.2f} (Loss due to wrong choices)")

    pct_systemic = (df['Systemic_Error'].sum() / df['Total_Error'].sum()) * 100
    logging.info(f"Conclusion: {pct_systemic:.1f}% of error is due to components simply being wrong.")
    logging.info(f"            {100 - pct_systemic:.1f}% of error is due to the Meta-Learner picking the wrong one.")

    # 4. Regime Classification Matrix
    logging.info("\n🚥 REGIME DETECTION ACCURACY")
    # Define "True" Regime
    df['True_Regime'] = df.apply(define_regime, axis=1)

    # Define "Predicted" Regime (Did we switch?)
    # If Super Ensemble is far from Trend, we "Predicted" a regime change
    df['Pred_Deviation'] = df['Super_Ensemble_pred'] - df['Statistical_Trend_pred']

    def classify_pred(val):
        if val < -30: return 'Crash'  # We predicted a drop
        if val > 30: return 'Bumper'  # We predicted a jump
        return 'Normal'  # We stuck to trend

    df['Pred_Regime'] = df['Pred_Deviation'].apply(classify_pred)

    # Confusion Matrix
    labels = ['Crash', 'Normal', 'Bumper']
    cm = confusion_matrix(df['True_Regime'], df['Pred_Regime'], labels=labels)

    logging.info(f"{'':<10} | {'Pred Crash':<10} | {'Pred Normal':<10} | {'Pred Bumper':<10}")
    logging.info("-" * 50)
    for i, label in enumerate(labels):
        logging.info(f"{'True ' + label:<10} | {cm[i][0]:<10} | {cm[i][1]:<10} | {cm[i][2]:<10}")

    # 5. Top Missed Opportunities (High Selection Error)
    logging.info("\n🤦 TOP MISSED OPPORTUNITIES (Meta-Learner Blunders)")
    logging.info("Cases where a good model existed, but we ignored it.")

    misses = df.sort_values('Selection_Error', ascending=False).head(5)
    logging.info(f"{'Year':<5} | {'District':<15} | {'Actual':<6} | {'Our Pred':<8} | {'Oracle':<8} | {'Better Model'}")

    for _, row in misses.iterrows():
        # Find who was best
        best_model = "Unknown"
        min_e = 9999
        for col in pred_cols:
            e = abs(row['kreisYield'] - row[col])
            if e < min_e:
                min_e = e
                best_model = col.replace('_pred', '')

        logging.info(
            f"{row['year']:<5} | {row['District_Name'][:15]:<15} | {row['kreisYield']:<6.0f} | {row['Super_Ensemble_pred']:<8.0f} | {row['Oracle_Pred']:<8.0f} | {best_model}")


if __name__ == "__main__":
    analyze_forensic_failures()