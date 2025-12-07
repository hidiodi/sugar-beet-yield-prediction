import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
LOG_LEVEL = logging.INFO
# Forecast Features (Input)
DATA_PATH = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
# Observed Weather (Target)
WEATHER_DIR = config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']['DAILY_WEATHER_DIR']
# Climate Indices (Context)
INDICES_PATH = config.DATA_DIR / '02_intermediate/climateIndices/long_range_climate_features.csv'
# Output
OUTPUT_DIR = config.DATA_DIR / '06_model_output'
PRED_FILE = config.DATA_DIR / '02_intermediate/predicted_heat_stress_march.csv'

# Hyperparameters (Tuned for Trend + Noise)
HEAT_MODEL_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.015,  # Slow learning
    'max_depth': 4,  # Depth 4 captures interactions (e.g., NAO * Trend)
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 5.0,  # L1 Regularization
    'n_jobs': -1,
    'random_state': 42
}


def load_ground_truth_from_raw_weather():
    """Calculates TRUE observed heat days from raw weather files (Safe)."""
    logging.info(f"Calculating Ground Truth Heat Days from {WEATHER_DIR}...")
    files = list(WEATHER_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No weather files found in {WEATHER_DIR}")

    results = []
    for f in tqdm(files, desc="Scanning Weather History"):
        try:
            df = pd.read_csv(f, usecols=['district_no', 'date', 'tmax'])
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month

            # Summer: June, July, August
            summer = df[df['month'].isin([6, 7, 8])]

            # Count days > 30°C
            annual_counts = summer.groupby(['district_no', 'year']).apply(
                lambda x: (x['tmax'] > 30).sum()
            ).reset_index(name='actual_heat_days')

            results.append(annual_counts)
        except Exception:
            continue

    return pd.concat(results, ignore_index=True)


def load_data_clean_room():
    # 1. Load Features (Forecasts)
    logging.info(f"Loading Forecast Features from {DATA_PATH}...")
    df_features = pd.read_csv(DATA_PATH)

    # 2. Load Truth (Observations)
    df_truth = load_ground_truth_from_raw_weather()

    # Merge
    df_features['district_no'] = df_features['district_no'].astype(int)
    df_truth['district_no'] = df_truth['district_no'].astype(int)

    df = pd.merge(df_features, df_truth, on=['year', 'district_no'], how='inner')
    logging.info(f"Merged Data: {len(df)} rows.")

    # 3. Load Climate Indices (Leading Indicators)
    if INDICES_PATH.exists():
        logging.info(f"Loading Climate Indices from {INDICES_PATH}...")
        indices = pd.read_csv(INDICES_PATH)

        # Clean column names (strip whitespace)
        indices.columns = indices.columns.str.strip()

        # Ensure year match
        indices['year'] = indices['year'].astype(int)

        # Check if columns exist
        required_indices = ['nao_winter_avg', 'sca_winter_avg', 'enso_mei_winter_avg']
        for c in required_indices:
            if c not in indices.columns:
                logging.warning(f"WARNING: Index '{c}' not found in CSV! Available: {indices.columns.tolist()}")

        df = pd.merge(df, indices, on='year', how='left')

        # Fill NaNs
        for col in required_indices:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            else:
                df[col] = 0.0
    else:
        logging.warning("Indices file missing. Using 0s.")

    return df


def train_heat_model(df):
    logging.info("Training Heat Prediction Model (Trend-Aware)...")

    features = [
        # 1. THE TREND (Crucial for Climate Change)
        'year',

        # 2. Climate Teleconnections (Drivers)
        'nao_winter_avg',
        'sca_winter_avg',
        'enso_mei_winter_avg',

        # 3. ECMWF Forecasts (Weak but useful signal)
        'summer_temp_prob_warm_forecast',
        'summer_precip_anomaly_forecast',
        'summer_solar_rad_anomaly_forecast',

        # 4. Soil Memory
        'effective_winter_water',
        'winter_buffer_x_summer_heat',  # Interaction

        # 5. Geography
        'avg_sand_0_30cm', 'avg_clay_0_30cm', 'lat', 'lon'
    ]

    feats = [f for f in features if f in df.columns]
    logging.info(f"Using Features: {feats}")

    preds = []

    # Constraints: Year is positive (Climate Change = More Heat)
    constraints_map = {
        'year': 1,
        'summer_temp_prob_warm_forecast': 1,
        'effective_winter_water': -1
    }
    constraints = tuple([constraints_map.get(f, 0) for f in feats])
    params = HEAT_MODEL_PARAMS.copy()
    params['monotone_constraints'] = constraints

    # Validation Loop
    start_year = 2005
    for year in sorted(df['year'].unique()):
        if year < start_year: continue

        train = df[df['year'] < year]
        test = df[df['year'] == year]

        if len(train) < 50: continue

        model = xgb.XGBRegressor(**params)
        model.fit(train[feats], train['actual_heat_days'])

        p = model.predict(test[feats])
        p = np.maximum(p, 0)  # No negative days

        res = test[['year', 'district_no']].copy()
        res['predicted_heat_days'] = p
        res['actual_heat_days'] = test['actual_heat_days']
        preds.append(res)

    return pd.concat(preds)


def evaluate_and_save(preds):
    r2 = r2_score(preds['actual_heat_days'], preds['predicted_heat_days'])
    mae = mean_absolute_error(preds['actual_heat_days'], preds['predicted_heat_days'])
    pearson, _ = pearsonr(preds['predicted_heat_days'], preds['actual_heat_days'])

    print("\n" + "=" * 40)
    print(" HEAT PROXY MODEL RESULTS (TREND-AWARE)")
    print("=" * 40)
    print(f"R²:           {r2:.4f}")
    print(f"Correlation:  {pearson:.4f}")
    print(f"MAE:          {mae:.2f} days")

    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    preds[['year', 'district_no', 'predicted_heat_days']].to_csv(PRED_FILE, index=False)
    print(f"\nPredictions saved to {PRED_FILE}")


def main():
    logging.basicConfig(level=LOG_LEVEL, format='%(message)s')
    try:
        df = load_data_clean_room()
        preds = train_heat_model(df)
        if not preds.empty:
            evaluate_and_save(preds)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()