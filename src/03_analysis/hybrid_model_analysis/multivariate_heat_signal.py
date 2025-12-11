import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder

# --- PATH SETUP ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from src import config

# --- CONFIGURATION ---
MIN_DAYS_CAP = 0.0
MAX_DAYS_CAP = 90.0

# Valid Features
VALID_FEATURES = [
    'crs_anomaly', 'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly',
    'antecedent_gdd_sum_anomaly', 'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast',
    'spring_solar_rad_anomaly_forecast', 'spring_evaporation_anomaly_forecast',
    'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
    'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast',
    'summer_precip_anomaly_forecast', 'summer_solar_rad_anomaly_forecast',
    'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
    'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast',
    'state_name', 'latitude', 'longitude', 'avg_elevation', 'avg_slope',
    'avg_bdod_0_30cm', 'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm',
    'avg_phh2o_0_30cm', 'nao_winter_avg', 'sca_winter_avg', 'enso_mei_winter_avg',
    'winter_precip_sum', 'feb_frost_days', 'winter_precip_anomaly',
    'effective_winter_water', 'winter_cropland_ndvi_anomaly', 'summer_water_balance_anomaly'
]

logging.basicConfig(level=logging.INFO, format='%(message)s')


# --- 1. DATA LOADER ---
def load_real_production_data():
    logging.info("LOADING REAL DATA (Power-Scaled Weighted Trend)...")
    paths = config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']

    # A. Forecast Features
    df_fcst = pd.read_csv(paths['ECMWF_FORECAST_FEATURES_CSV'])
    df_fcst['district_no'] = df_fcst['district_no'].astype(str).str.zfill(5)

    available_features = [c for c in VALID_FEATURES if c in df_fcst.columns]
    agg_dict = {feat: 'mean' for feat in available_features if pd.api.types.is_numeric_dtype(df_fcst[feat])}

    if 'state_name' in available_features:
        agg_dict['state_name'] = 'first'
    if 'summer_temp_anomaly_forecast' in df_fcst.columns:
        agg_dict['summer_temp_anomaly_forecast'] = 'mean'

    df_grouped = df_fcst.groupby(['district_no', 'year']).agg(agg_dict).reset_index()

    # B. Ground Truth
    logging.info("Parsing Daily Weather for Ground Truth...")
    weather_dir = paths['DAILY_WEATHER_DIR']
    obs_list = []

    all_files = list(weather_dir.glob("*.csv"))
    for f in all_files:
        try:
            tmp = pd.read_csv(f, usecols=['district_no', 'date', 'tmax'])
            tmp['date'] = pd.to_datetime(tmp['date'])
            tmp = tmp[tmp['date'].dt.month.isin([6, 7, 8])]

            # Count days > 30
            heat_counts = tmp.groupby(['district_no', tmp['date'].dt.year])['tmax'].apply(lambda x: (x > 30).sum())
            for (dist, yr), count in heat_counts.items():
                obs_list.append({'district_no': str(dist).zfill(5), 'year': yr, '__obs_heat': count})
        except:
            continue

    df_obs = pd.DataFrame(obs_list)
    df_real = pd.merge(df_grouped, df_obs, on=['district_no', 'year'], how='inner')

    # Encode Categorical
    if 'state_name' in df_real.columns:
        enc = OrdinalEncoder()
        df_real['state_encoded'] = enc.fit_transform(df_real[['state_name']])
        df_real = df_real.drop(columns=['state_name'])

    # Impute
    feature_cols = [c for c in df_real.columns if c not in ['district_no', 'year', '__obs_heat']]
    df_real[feature_cols] = df_real[feature_cols].fillna(df_real[feature_cols].mean())

    logging.info(f"Loaded {len(df_real)} real samples.")
    return df_real

# File: src/multivariate_heat_signal.py

# --- 2. WEIGHTED TREND ENGINE ---
def calculate_weighted_trend(df_history, current_year):
    """
    Fits a RECENCY-WEIGHTED Linear Trend.
    """
    results = []

    for dist, group in df_history.groupby('district_no'):
        # Ensure sufficient history for trend calculation
        if len(group) < 5:
            trend_val = group['__obs_heat'].mean()
            resid_std = group['__obs_heat'].std()
        else:
            X = group[['year']].values
            y = group['__obs_heat'].values

            # Recency Weights: 1.15 ^ (Year - Min)
            years = group['year'].values
            weights = np.power(1.15, (years - years.min()))

            model = LinearRegression()
            try:
                model.fit(X, y, sample_weight=weights)
                # Predict the trend value for the current year
                trend_val = model.predict([[current_year]])[0]

                # Residuals (Unweighted)
                hist_trend = model.predict(X)
                residuals = y - hist_trend
                resid_std = np.std(residuals)
            except:
                trend_val = y.mean()
                resid_std = y.std()

        # Safety
        if pd.isna(resid_std) or resid_std < 0.1:
            resid_std = 1.0

        # Physical floor
        trend_val = max(0.0, trend_val)

        results.append({
            'district_no': dist,
            'trend_anchor': trend_val,
            'resid_std': resid_std
        })

    return pd.DataFrame(results)


# --- 3. MODELING ENGINE ---
def train_and_predict_power_z(df_history, df_current):
    current_year = df_current['year'].iloc[0]

    # 1. Get Trend Anchors
    # We still use the recency-weighted linear trend as the baseline
    anchors = calculate_weighted_trend(df_history, current_year)

    # Merge Anchors
    df_curr = df_current.merge(anchors, on='district_no', how='left')

    # Fill missing anchors
    df_curr['trend_anchor'] = df_curr['trend_anchor'].fillna(anchors['trend_anchor'].mean())
    df_curr['resid_std'] = df_curr['resid_std'].fillna(anchors['resid_std'].mean())

    # 2. Target: Anomaly vs. Trend Anchor (Simplified Objective)
    df_train = df_history.merge(anchors, on='district_no', how='left')
    df_train['trend_anchor'] = df_train['trend_anchor'].fillna(anchors['trend_anchor'].mean())
    df_train['target_anomaly'] = df_train['__obs_heat'] - df_train['trend_anchor']

    # 3. Train
    # Using Anomaly as target and including state_encoded
    exclude = ['district_no', 'year', '__obs_heat', 'target_anomaly', 'trend_anchor',
               'state_name', 'mean', 'std', 'resid_std']
    X_cols = [c for c in df_train.columns if c not in exclude]

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_depth=6,
        l2_regularization=2.0,
        random_state=42
    )
    model.fit(df_train[X_cols], df_train['target_anomaly'])

    # 4. Predict (Predicting the Anomaly)
    train_anomaly_preds = model.predict(df_train[X_cols])
    curr_anomaly_preds = model.predict(df_curr[X_cols])

    # 5. Bias Correction (Bias-correct the ANOMALY prediction)
    bias_adj = -np.mean(train_anomaly_preds - df_train['target_anomaly'])
    curr_anomaly_preds_adj = curr_anomaly_preds + bias_adj

    # 6. LINEAR VOLATILITY INFLATION (New/Restored Step)
    # Scale the predicted anomaly magnitude to match the observed anomaly magnitude
    std_train_anomaly = df_train['target_anomaly'].std()
    std_pred_anomaly = np.std(train_anomaly_preds)

    inflation = std_train_anomaly / std_pred_anomaly if std_pred_anomaly > 0.01 else 1.0
    # Apply a slight clip to prevent runaway inflation, but allow a larger boost
    inflation = np.clip(inflation, 0.8, 3.0)

    final_anomaly = curr_anomaly_preds_adj * inflation

    # 7. Reconstruct
    df_res = df_curr.copy()

    # MODIFIED: Scale the final anomaly by the district's local residual standard deviation (resid_std)
    # This adjusts the generic anomaly prediction magnitude to the expected local volatility.
    scaled_anomaly = final_anomaly * df_res['resid_std']

    # Reconstruct by adding the SCALED anomaly back to the trend anchor
    df_res['pred_days'] = df_res['trend_anchor'] + scaled_anomaly

    # Clip
    df_res['pred_days'] = np.clip(df_res['pred_days'], MIN_DAYS_CAP, MAX_DAYS_CAP)

    # Save Anomaly for audit, using the original column name
    df_res['fcst_z_local'] = final_anomaly  # Note: fcst_z_local is now the UN-scaled anomaly

    return df_res[['district_no', 'year', '__obs_heat', 'pred_days', 'fcst_z_local']]

# --- 4. WALK-FORWARD LOOP ---
def run_walk_forward(df):
    min_year = df['year'].min() # This should be 1981
    max_year = df['year'].max()
    predictions = []

    # MODIFIED: Start the loop earlier, e.g., after 5 years of data (1981-1985)
    START_PREDICT_YEAR = min_year + 5

    logging.info(f"Starting Power-Scaled Z-Score Walk-Forward (Full History: {START_PREDICT_YEAR} - {max_year})...")

    for year in range(START_PREDICT_YEAR, max_year + 1):
        df_history = df[df['year'] < year].copy()
        df_current = df[df['year'] == year].copy()

        # MODIFIED: Relax history check significantly for backtesting
        # We need at least 5 years of history (guaranteed by START_PREDICT_YEAR) and a few hundred samples.

        if len(df_history) < 100:
            logging.warning(f"Skipping year {year}: Insufficient history ({len(df_history)} samples)")
            continue

        pred_df = train_and_predict_power_z(df_history, df_current)
        predictions.append(pred_df)

    return pd.concat(predictions)


# --- 5. AUDIT REPORT ---
def run_audit_report(df):
    obs = df['__obs_heat']
    fcst = df['pred_days']

    bias = (fcst - obs).mean()
    mae = (fcst - obs).abs().mean()
    corr = obs.corr(fcst)
    vol_ratio = fcst.std() / obs.std()

    logging.info("=" * 60)
    logging.info(f" FINAL POWER-SCALED SIGNAL AUDIT REPORT")
    logging.info("=" * 60)
    logging.info(f"{'Metric':<20} | {'Value':<10} | {'Target'}")
    logging.info("-" * 60)
    logging.info(f"{'Correlation':<20} | {corr:<10.3f} | {'> 0.65'}")
    logging.info(f"{'Bias':<20} | {bias:<10.3f} | {'+/- 0.5'}")
    logging.info(f"{'MAE':<20} | {mae:<10.3f} | {'Minimize'}")
    logging.info(f"{'Volatility Ratio':<20} | {vol_ratio:<10.3f} | {'0.95 - 1.05'}")
    logging.info("-" * 60)

    # NEW: Log critical years check
    critical_years = [2003, 2014, 2018]
    logging.info("-" * 60)
    logging.info(">>> CRITICAL YEAR CHECK (Obs vs Pred)")
    for year in critical_years:
        year_data = df[df['year'] == year]
        if not year_data.empty:
            mean_obs = year_data['__obs_heat'].mean()
            mean_pred = year_data['pred_days'].mean()
            logging.info(f"Year {year:<15} | Obs: {mean_obs:<6.1f} | Pred: {mean_pred:<6.1f} | Bias: {mean_pred - mean_obs:<6.2f}")
    logging.info("-" * 60)

    # Signal Check
    try:
        df['z_bin'] = pd.cut(df['fcst_z_local'], bins=[-np.inf, -0.8, -0.25, 0.25, 0.8, np.inf],
                             labels=["Cool", "Avg-", "Avg", "Avg+", "Hot"])
    except:
        df['z_bin'] = pd.cut(df['fcst_z_local'], 5)

    signal_stats = df.groupby('z_bin', observed=False)[['__obs_heat', 'pred_days']].mean()
    signal_stats['bias'] = signal_stats['pred_days'] - signal_stats['__obs_heat']

    logging.info(">>> SIGNAL SENSITIVITY (Relative to Trend)")
    logging.info(f"{'Signal':<15} | {'Obs':<6} | {'Pred':<6} | {'Bias':<6}")
    logging.info("-" * 50)
    for sig in signal_stats.index:
        row = signal_stats.loc[sig]
        logging.info(f"{sig:<15} | {row['__obs_heat']:<6.1f} | {row['pred_days']:<6.1f} | {row['bias']:<6.2f}")
    logging.info("")

    # Extreme Check
    thresh = df['__obs_heat'].quantile(0.90)
    extremes = df[df['__obs_heat'] > thresh]
    if len(extremes) > 0:
        pred_thresh = df['pred_days'].quantile(0.90)
        ext_capture = (extremes['pred_days'] > pred_thresh).mean()
        ext_bias = (extremes['pred_days'] - extremes['__obs_heat']).mean()

        logging.info(">>> EXTREME EVENTS")
        logging.info(f"Capture Rate     : {ext_capture:.1%} (Aiming for > 50%)")
        logging.info(f"Extreme Bias     : {ext_bias:.2f} (Aiming for 0.0)")


if __name__ == "__main__":
    df = load_real_production_data()
    results = run_walk_forward(df)
    run_audit_report(results)

    out_path = config.BASE_DIR / 'data/processed/heat_signal_multivariate.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logging.info(f"Saved signal to: {out_path}")