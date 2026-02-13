import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- PATH SETUP ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
import importlib

models_config = importlib.import_module("src.02_models.config")

# --- CONFIGURATION ---
MAX_DAYS_CAP = 92.0
HEAT_THRESHOLD = 25.0

logging.basicConfig(level=logging.INFO, format='%(message)s')


# --- 1. DATA LOADER ---
def load_data():
    logging.info("LOADING DATA (Target: Days > 25°C)...")
    paths = models_config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']

    obs_list = []
    weather_dir = paths['DAILY_WEATHER_DIR']

    if weather_dir.exists():
        for f in weather_dir.glob("*.csv"):
            try:
                df_w = pd.read_csv(f, usecols=['district_no', 'date', 'tmax'])
                df_w['date'] = pd.to_datetime(df_w['date'])
                df_w = df_w[df_w['date'].dt.month.isin([6, 7, 8])]

                grp = df_w.groupby(['district_no', df_w['date'].dt.year])['tmax']
                counts = grp.apply(lambda x: (x > HEAT_THRESHOLD).sum())

                for (d, y), v in counts.items():
                    obs_list.append({'district_no': str(d).zfill(5), 'year': y, 'heat_days_obs': v})
            except:
                continue

    df_obs = pd.DataFrame(obs_list)
    if df_obs.empty:
        logging.error("No ground truth found.")
        return pd.DataFrame()

    df_fcst = pd.read_csv(paths['ECMWF_FORECAST_FEATURES_CSV'])
    df_fcst['district_no'] = df_fcst['district_no'].astype(str).str.zfill(5)

    num_cols = df_fcst.select_dtypes(include=np.number).columns.tolist()
    agg = {c: 'mean' for c in num_cols if c not in ['district_no', 'year']}
    df_grouped = df_fcst.groupby(['district_no', 'year']).agg(agg).reset_index()

    try:
        df_master = pd.read_csv(paths['MASTER_DATASET'])
        df_master['district_no'] = df_master['district_no'].astype(str).str.zfill(5)

        static_cols = [
            'district_no', 'latitude', 'longitude',
            'avg_phh2o_0_100cm', 'avg_bdod_0_30cm',
            'avg_sand_0_30cm', 'avg_slope', 'nao_winter_avg'
        ]
        static_cols = [c for c in static_cols if c in df_master.columns]

        if 'year' in df_master.columns:
            df_grouped = pd.merge(df_grouped, df_master[static_cols + ['year']], on=['district_no', 'year'], how='left')
        else:
            df_grouped = pd.merge(df_grouped, df_master[static_cols], on='district_no', how='left')
    except Exception as e:
        logging.warning(f"Static features missing: {e}")

    df_final = pd.merge(df_grouped, df_obs, on=['district_no', 'year'], how='inner')
    df_final = df_final.fillna(0)
    return df_final


# --- 2. DYNAMIC FEATURE ENGINEERING ---
def create_interaction_features(df):
    df = df.copy()
    if 'summer_temp_anomaly_forecast' in df.columns and 'avg_sand_0_30cm' in df.columns:
        sand_proxy = df['avg_sand_0_30cm'] / (df['avg_sand_0_30cm'].max() + 1.0)
        df['heat_sand_interaction'] = df['summer_temp_anomaly_forecast'] * sand_proxy

    if 'summer_temp_anomaly_forecast' in df.columns:
        t = df['summer_temp_anomaly_forecast']
        df['summer_temp_sq'] = np.sign(t) * (t ** 2)
        df['summer_temp_cube'] = t ** 3

    if 'spring_solar_rad_anomaly_forecast' in df.columns and 'spring_soil_temp_l1_anomaly_forecast' in df.columns:
        df['spring_drying_pressure'] = (
                df['spring_solar_rad_anomaly_forecast'] * df['spring_soil_temp_l1_anomaly_forecast']
        )

    if 'summer_temp_sq' in df.columns and 'spring_runoff_anomaly_forecast' in df.columns:
        df['compound_drought_risk'] = df['summer_temp_sq'] * (-1 * df['spring_runoff_anomaly_forecast'])
    return df


# --- 3. WALK-FORWARD MODELING (NO LEAK) ---
def train_and_predict(df_train, df_test):
    min_year = df_train['year'].min()
    climatology_df = df_train[df_train['year'] <= (min_year + 10)]
    if climatology_df.empty: climatology_df = df_train

    district_means = climatology_df.groupby('district_no')['heat_days_obs'].mean()
    global_mean = climatology_df['heat_days_obs'].mean()

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train['hist_avg_heat'] = df_train['district_no'].map(district_means).fillna(global_mean)
    df_test['hist_avg_heat'] = df_test['district_no'].map(district_means).fillna(global_mean)

    exclude = ['district_no', 'year', 'heat_days_obs', 'pred_raw', 'pred_days', 'seas5_member']
    features = [c for c in df_train.columns if c not in exclude]

    X_train, y_train = df_train[features], df_train['heat_days_obs']
    X_test = df_test[features]

    weights = (y_train / (y_train.mean() + 1e-5)) ** 3
    weights = np.clip(weights, 1.0, 50.0)

    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=800, learning_rate=0.015,
        max_depth=8, min_child_weight=5, colsample_bytree=0.8, subsample=0.8,
        n_jobs=-1, random_state=42
    )

    model.fit(X_train, y_train, sample_weight=weights)
    return model.predict(X_test), model.predict(X_train), model.feature_importances_, features


def run_walk_forward(df):
    df = create_interaction_features(df)
    years = sorted(df['year'].unique())
    start_year = years[max(5, int(len(years) * 0.2))]

    logging.info(f"Walk-forward start year: {start_year}")
    results, importances = [], []

    for yr in years:
        if yr < start_year: continue
        train, test = df[df['year'] < yr], df[df['year'] == yr]
        if test.empty: continue

        test_raw, train_raw, feat_imp, feat_names = train_and_predict(train, test)

        # Correcting Volatility based ONLY on Train stats
        obs_std_train, pred_std_train = train['heat_days_obs'].std(), train_raw.std()
        mu_pred_train = train_raw.mean()

        scale_factor = np.clip((obs_std_train / (pred_std_train + 0.01)) * 1.35, 1.0, 4.0)
        test_final = np.clip(mu_pred_train + (test_raw - mu_pred_train) * scale_factor, 0, MAX_DAYS_CAP)

        res = test[['district_no', 'year', 'heat_days_obs']].copy()
        res['pred_raw'], res['pred_days'] = test_raw, test_final
        results.append(res);
        importances.append(feat_imp)

    full_res = pd.concat(results)
    avg_imp = np.mean(importances, axis=0)
    imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': avg_imp}).sort_values('Importance', ascending=False)
    logging.info("\nTOP PREDICTORS:\n" + imp_df.head(10).to_string())
    return full_res


# --- 4. VISUALIZATION ---
def generate_calibration_plot(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    plt.scatter(df['heat_days_obs'], df['pred_days'], alpha=0.15, color='gray', s=10, label='All Districts')

    highlights = {2003: 'red', 2014: 'blue', 2018: 'orange', 2022: 'purple'}
    for yr, color in highlights.items():
        sub = df[df['year'] == yr]
        if not sub.empty:
            obs_m, pred_m = sub['heat_days_obs'].mean(), sub['pred_days'].mean()
            plt.scatter(sub['heat_days_obs'], sub['pred_days'], color=color, s=25,
                        label=f"{yr} (Obs:{obs_m:.1f} vs Pred:{pred_m:.1f})")

    lims = [0, MAX_DAYS_CAP]
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Perfect Fit')

    corr = df['heat_days_obs'].corr(df['pred_days'])
    plt.title(f"Forecast Calibration Check (Correlation: {corr:.2f})")
    plt.xlabel("Observed Heat Days (Reality)")
    plt.ylabel("Forecasted Heat Days (Model)")
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Calibration plot saved to: {output_path}")


def audit_results(df):
    obs, pred = df['heat_days_obs'], df['pred_days']
    corr, mae = obs.corr(pred), mean_absolute_error(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))

    ext_mae = mean_absolute_error(df[obs > obs.quantile(0.90)]['heat_days_obs'],
                                  df[obs > obs.quantile(0.90)]['pred_days'])

    logging.info("=" * 60 + "\n FINAL HEAT SIGNAL MODEL RESULTS\n" + "=" * 60)
    logging.info(f"Correlation: {corr:.3f} | MAE: {mae:.1f} | Ext_MAE: {ext_mae:.1f} | RMSE: {rmse:.1f}")
    return df


if __name__ == "__main__":
    df_raw = load_data()
    if not df_raw.empty:
        df_final = run_walk_forward(df_raw)
        audit_results(df_final)

        # Original output folder for CSV to maintain pipeline
        csv_out = models_config.BASE_DIR / 'data/processed/heat_signal_multivariate_moderate.csv'
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(csv_out, index=False)
        logging.info(f"Saved CSV: {csv_out}")

        # New plot in reports/analysis
        plot_out = models_config.BASE_DIR / 'reports/analysis/heat_forecast_calibration.png'
        generate_calibration_plot(df_final, plot_out)