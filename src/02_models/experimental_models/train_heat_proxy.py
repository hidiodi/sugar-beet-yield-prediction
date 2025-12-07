import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
LOG_LEVEL = logging.INFO
DATA_PATH = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
OUTPUT_DIR = config.DATA_DIR / '06_model_output'
PRED_FILE = config.DATA_DIR / '02_intermediate/predicted_heat_stress_march.csv'

# Model Hyperparameters (Tuned for noisy, small-signal regression)
HEAT_MODEL_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.02,  # Slow learning to find weak signals
    'max_depth': 4,  # Shallow trees to prevent overfitting noise
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'reg_alpha': 5.0,  # High L1 regularization to kill useless features
    'n_jobs': -1,
    'random_state': 42
}


def load_data():
    logging.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 1. Define Target: The ACTUAL observed hot days (Observed in Sept)
    # We want to learn to predict this using only March data.
    if 'summer_days_tmax_gt_30c' not in df.columns:
        raise ValueError("Target 'summer_days_tmax_gt_30c' not found in dataset!")

    df['target_heat_days'] = df['summer_days_tmax_gt_30c']

    # 2. Merge Climate Indices (NAO, ENSO)
    # TODO: If you have an external CSV with indices, merge it here!
    # Example:
    # indices = pd.read_csv('data/climate_indices.csv')
    # df = pd.merge(df, indices, on='year', how='left')

    # Placeholder for demonstration (Replace with real loading logic)
    if 'nao_index_djf' not in df.columns:
        df['nao_index_djf'] = 0.0  # Placeholder
    if 'enso_index_djf' not in df.columns:
        df['enso_index_djf'] = 0.0  # Placeholder

    return df


def train_heat_model(df):
    logging.info("Training Heat Prediction Model (March Horizon)...")

    # FEATURES (Available in March)
    features = [
        # ECMWF Forecasts (The 'Raw' Signal)
        'summer_temp_prob_warm_forecast',
        'summer_precip_anomaly_forecast',
        'summer_solar_rad_anomaly_forecast',

        # Climate Teleconnections (The 'Macro' Signal)
        'nao_index_djf',
        'enso_index_djf',

        # Antecedent Conditions (The 'Memory')
        'effective_winter_water',  # Dry winter often precedes hot summer
        'winter_buffer_x_summer_heat',  # Interaction proxy

        # Static Geography (Heat varies by location)
        'avg_sand_0_30cm', 'avg_clay_0_30cm', 'lat', 'lon'
    ]

    # Filter valid columns
    feats = [f for f in features if f in df.columns]
    logging.info(f"Using Features: {feats}")

    # Strict Walk-Forward Validation
    preds = []

    # Monotonic Constraints:
    # High Forecast Prob -> More Hot Days (+1)
    # High Winter Rain -> Fewer Hot Days (-1) (Evaporative cooling)
    constraints_map = {
        'summer_temp_prob_warm_forecast': 1,
        'effective_winter_water': -1
    }
    constraints = tuple([constraints_map.get(f, 0) for f in feats])
    params = HEAT_MODEL_PARAMS.copy()
    params['monotone_constraints'] = constraints

    start_year = 2005
    for year in sorted(df['year'].unique()):
        if year < start_year: continue

        train = df[df['year'] < year]
        test = df[df['year'] == year]

        if len(train) < 50: continue

        model = xgb.XGBRegressor(**params)
        model.fit(train[feats], train['target_heat_days'])

        p = model.predict(test[feats])

        res = test[['year', 'district_no']].copy()
        res['predicted_heat_days'] = p
        res['actual_heat_days'] = test['target_heat_days']
        preds.append(res)

    return pd.concat(preds)


def evaluate_and_save(preds):
    # Metrics
    r2 = r2_score(preds['actual_heat_days'], preds['predicted_heat_days'])
    mae = mean_absolute_error(preds['actual_heat_days'], preds['predicted_heat_days'])
    pearson, _ = pearsonr(preds['predicted_heat_days'], preds['actual_heat_days'])

    print("\n" + "=" * 40)
    print(" HEAT PROXY MODEL RESULTS")
    print("=" * 40)
    print(f"R²:           {r2:.4f}")
    print(f"Correlation:  {pearson:.4f}")
    print(f"MAE:          {mae:.2f} days")

    # Save for use in Main Model
    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    preds[['year', 'district_no', 'predicted_heat_days']].to_csv(PRED_FILE, index=False)
    print(f"\nPredictions saved to {PRED_FILE}")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(data=preds, x='predicted_heat_days', y='actual_heat_days',
                scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'})
    plt.title(f"Heat Predictor: Forecast vs Observed (R={pearson:.2f})")
    plt.xlabel("Predicted Hot Days (>30°C)")
    plt.ylabel("Observed Hot Days")
    plt.savefig(OUTPUT_DIR / 'heat_model_performance.png')
    print("Plot saved.")


def main():
    logging.basicConfig(level=LOG_LEVEL, format='%(message)s')
    df = load_data()
    preds = train_heat_model(df)
    evaluate_and_save(preds)


if __name__ == "__main__":
    main()