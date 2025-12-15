import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib
models_config = importlib.import_module("src.02_models.config")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = global_config.DATA_DIR / '06_model_output' / 'recovery_models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE1_PATH = models_config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
STAGE2_PATH = global_config.DATA_DIR / '05_model_input/stage2_refined_features.csv'


def calculate_trend_walk_forward(df):
    logger.info("--- Calculating Trend (Walk-Forward) ---")
    df = df.sort_values(['district_no', 'year'])
    results = []
    min_train = 10

    for district, group in df.groupby('district_no'):
        if len(group) < min_train: continue
        years = group['year'].values
        yields = group['kreisYield'].values

        for i in range(min_train, len(group)):
            train_y = yields[:i]
            train_x = years[:i]
            target_year = years[i]
            try:
                z = np.polyfit(train_x, train_y, 1)
                p = np.poly1d(z)
                forecast = p(target_year)
            except:
                forecast = np.mean(train_y)
            results.append({'district_no': district, 'year': target_year, 'trend_forecast': forecast})

    return pd.DataFrame(results)


def load_merged_data():
    logger.info("--- Loading and Merging Data ---")

    # 1. Stage 1 (Base Features + Yield)
    if not STAGE1_PATH.exists():
        logger.error(f"Stage 1 file missing: {STAGE1_PATH}")
        return None
    df1 = pd.read_csv(STAGE1_PATH)
    df1['district_no'] = df1['district_no'].astype(str).str.zfill(5)

    # 2. Stage 2 (New Features)
    if not STAGE2_PATH.exists():
        logger.error(f"Stage 2 file missing: {STAGE2_PATH}")
        return None
    df2 = pd.read_csv(STAGE2_PATH)
    df2['district_no'] = df2['district_no'].astype(str).str.zfill(5)

    # Merge
    df = pd.merge(df1, df2, on=['year', 'district_no'], how='left')

    # 3. Calculate Trend
    if 'kreisYield' not in df.columns:
        logger.error("Target column 'kreisYield' missing.")
        return None

    df_trend = calculate_trend_walk_forward(df)
    df_trend['district_no'] = df_trend['district_no'].astype(str).str.zfill(5)

    df = pd.merge(df, df_trend, on=['year', 'district_no'], how='inner')

    # 4. Feature Engineering
    df['residual_target'] = df['kreisYield'] - df['trend_forecast']

    # Use stage1 forecast as wofost proxy if available, else trend
    if 'wofost_yield_water_limited' not in df.columns:
        df['wofost_yield_water_limited'] = df['stage1_forecast'] if 'stage1_forecast' in df.columns else df[
            'trend_forecast']

    df['wofost_residual'] = df['wofost_yield_water_limited'] - df['trend_forecast']

    # Fill Stage 2 NaNs
    for c in ['VegetationVigorIndex', 'RootZoneDepletion', 'mild_winter_days', 'fungal_risk_days']:
        if c in df.columns: df[c] = df[c].fillna(0)

    # Drop rows where we can't train or evaluate
    df = df.dropna(subset=['residual_target', 'trend_forecast', 'wofost_residual'])
    logger.info(f"Final Merged Dataset: {len(df)} rows")
    return df


def train_stage2_model(df):
    logger.info("--- Training Stage 2 Robust Model (Walk-Forward) ---")

    features = [
        'wofost_residual',  # Stage 1 Core
        'VegetationVigorIndex',  # Stage 2 Satellite
        'RootZoneDepletion',  # Stage 2 Water
        'mild_winter_days',  # Stage 2 Pest
        'fungal_risk_days'  # Stage 2 Disease
    ]

    features = [f for f in features if f in df.columns]
    logger.info(f"Using Features: {features}")

    preds = []
    years = sorted(df['year'].unique())
    start_year = 2000

    for year in years:
        if year < start_year: continue

        train = df[df['year'] < year]
        test = df[df['year'] == year]
        if train.empty: continue

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('huber', HuberRegressor(epsilon=1.35, max_iter=200))
        ])

        model.fit(train[features], train['residual_target'])

        pred_res = model.predict(test[features])

        res = test[['year', 'district_no', 'kreisYield', 'trend_forecast']].copy()
        res['stage2_pred'] = res['trend_forecast'] + pred_res
        preds.append(res)

    return pd.concat(preds, ignore_index=True)


def main():
    df = load_merged_data()
    if df is None: return

    results = train_stage2_model(df)
    if results.empty:
        logger.error("No predictions generated.")
        return

    # Calculate Metrics
    mae_trend = mean_absolute_error(results['kreisYield'], results['trend_forecast'])
    mae_stage2 = mean_absolute_error(results['kreisYield'], results['stage2_pred'])

    logger.info("\n" + "=" * 60)
    logger.info(" FINAL STRATEGY VERIFICATION ")
    logger.info("=" * 60)
    logger.info(f"N Samples:        {len(results)}")
    logger.info(f"Trend Model MAE:  {mae_trend:.4f}")
    logger.info(f"Robust Model MAE: {mae_stage2:.4f}")
    logger.info("-" * 30)
    logger.info(f"IMPROVEMENT:      {mae_trend - mae_stage2:+.4f} dt/ha")

    if mae_stage2 < mae_trend:
        logger.info("✅ SUCCESS: Robust Model improves over Trend.")
    else:
        logger.info("❌ FAIL: Robust Model did not beat Trend.")

    results.to_csv(OUTPUT_DIR / 'stage2_forecasts.csv', index=False)


if __name__ == "__main__":
    main()