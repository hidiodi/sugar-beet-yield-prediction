# File: src/01_data/FeatureEngineering/build_stage1_features.py
# REFACTORED (v12.0): Water Balance Features (The Drought/Surge Signal)

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.FEATURE_ENGINEERING_CONFIG


def calculate_water_balance(df):
    """
    Calculates P-E (Precipitation - Evaporation) anomalies.
    Positive = Surplus (Good for 2014), Negative = Deficit (Bad for 2018).
    """
    logging.info("--- Calculating Water Balance (P-E) ---")

    # Spring Balance (Establishment Phase)
    if 'spring_precip_anomaly_forecast' in df.columns and 'spring_evaporation_anomaly_forecast' in df.columns:
        # Note: Evaporation is usually positive in magnitude.
        # Anomaly > 0 means MORE evap (bad). Precip Anomaly > 0 means MORE rain (good).
        df['spring_water_balance'] = df['spring_precip_anomaly_forecast'] - df['spring_evaporation_anomaly_forecast']
    else:
        df['spring_water_balance'] = 0

    # Summer Balance (Bulking Phase - Critical for Drought)
    if 'summer_precip_anomaly_forecast' in df.columns and 'summer_evaporation_anomaly_forecast' in df.columns:
        df['summer_water_balance'] = df['summer_precip_anomaly_forecast'] - df['summer_evaporation_anomaly_forecast']
    else:
        df['summer_water_balance'] = 0

    return df


def process_antecedent_weather_only(weather_dir: Path, start_year: int, end_year: int, forecast_month: int = 3):
    logging.info("--- Processing Antecedent Weather ---")
    all_weather_files = list(weather_dir.glob("*.csv"))
    if not all_weather_files: return pd.DataFrame()

    df_list = []
    for f in all_weather_files:
        try:
            temp = pd.read_csv(f)
            if 'date' in temp.columns:
                temp['date'] = pd.to_datetime(temp['date'])
                temp['year'] = temp['date'].dt.year
                temp = temp[(temp['year'] >= start_year - 1) & (temp['year'] <= end_year)]
                if not temp.empty: df_list.append(temp)
        except Exception:
            continue

    if not df_list: return pd.DataFrame()

    df_daily = pd.concat(df_list, ignore_index=True)
    df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)
    if 'prec' in df_daily.columns and 'precip' not in df_daily.columns:
        df_daily.rename(columns={'prec': 'precip'}, inplace=True)

    df_daily['tavg'] = (df_daily['tmin'] + df_daily['tmax']) / 2.0
    df_daily['harvest_year'] = df_daily['date'].apply(lambda d: d.year + 1 if d.month >= 10 else d.year)

    antecedent_mask = (
            ((df_daily['date'].dt.month < forecast_month) & (df_daily['year'] == df_daily['harvest_year'])) |
            ((df_daily['date'].dt.month >= 10) & (df_daily['year'] == df_daily['harvest_year'] - 1))
    )
    df_ant = df_daily[antecedent_mask].copy()

    df_ant['is_sowing_day'] = ((df_ant['tavg'] > 5) & (df_ant['precip'] < 1.0) & (df_ant['date'].dt.month == 2)).astype(
        int)
    df_ant['aphid_gdd'] = (df_ant['tmax'] - 3).clip(lower=0)
    df_ant['deep_frost_hit'] = (df_ant['tmin'] < -5).astype(int)

    df_agg = df_ant.groupby(['district_no', 'harvest_year']).agg(
        antecedent_precip_sum=('precip', 'sum'),
        sowing_potential_days=('is_sowing_day', 'sum'),
        winter_aphid_pressure=('aphid_gdd', 'sum'),
        winter_pest_kill_days=('deep_frost_hit', 'sum')
    ).reset_index().rename(columns={'harvest_year': 'year'})

    tox_mask = ((df_daily['year'] == df_daily['harvest_year'] - 1) & (df_daily['date'].dt.month.isin([8, 9, 10, 11])))
    df_tox = df_daily[tox_mask].groupby(['district_no', 'harvest_year'])['precip'].sum().reset_index()
    df_tox.rename(columns={'precip': 'prev_autumn_precip', 'harvest_year': 'year'}, inplace=True)

    df_final = pd.merge(df_agg, df_tox, on=['district_no', 'year'], how='left')
    df_final['prev_autumn_precip'] = df_final['prev_autumn_precip'].fillna(200)
    df_final['toxic_carryover_index'] = 1000.0 / (df_final['prev_autumn_precip'] + 10.0)

    return df_final


def load_and_process_economics(producer_file, input_file):
    try:
        logging.info("Processing Economics...")
        df_prod = pd.read_csv(producer_file)
        df_prod = df_prod[df_prod['ID'] == 'LWPR-132'].copy()
        df_prod = df_prod.melt(id_vars=['ID', 'Description'], var_name='year', value_name='producer_price_index')
        df_prod['year'] = pd.to_numeric(df_prod['year'], errors='coerce')
        df_prod['producer_price_index'] = pd.to_numeric(df_prod['producer_price_index'], errors='coerce')
        df_prod.dropna(subset=['year', 'producer_price_index'], inplace=True)
        df_prod['year'] = df_prod['year'].astype(int)

        df_input = pd.read_csv(input_file)
        IDS = {'LWBM-13': 'fertilizer_price_index', 'LWBM-11': 'seed_price_index'}
        df_input = df_input[df_input['ID'].isin(IDS.keys())].copy()
        df_input = df_input.melt(id_vars=['ID', 'Description'], var_name='period', value_name='val')
        df_input['year'] = pd.to_numeric(df_input['period'].str.split('/').str[1], errors='coerce')
        df_input['val'] = pd.to_numeric(df_input['val'], errors='coerce')
        df_input.dropna(subset=['year', 'val'], inplace=True)
        df_input['year'] = df_input['year'].astype(int)

        df_input = df_input.groupby(['year', 'ID'])['val'].mean().unstack().reset_index()
        df_input.rename(columns=IDS, inplace=True)

        df_econ = pd.merge(df_prod[['year', 'producer_price_index']], df_input, on='year', how='outer')
        return df_econ
    except Exception:
        return pd.DataFrame(columns=['year'])


def main():
    logging.info("--- Starting Feature Engineering (v12.0 Aggressive Physics) ---")
    paths = CONFIG['FILE_PATHS']
    paths['OUTPUT_DIR'].mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(paths['MASTER_DATASET'])
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)
    df['kreisYield'] = pd.to_numeric(df['yield'], errors='coerce')
    df.dropna(subset=['kreisYield'], inplace=True)

    # --- 2. Forecasts (Explicitly include Evap/SoilTemp) ---
    if 'spring_temp_anomaly_forecast' not in df.columns:
        df_fcst = pd.read_csv(paths['ECMWF_FORECAST_FEATURES_CSV'])
        df_fcst['district_no'] = df_fcst['district_no'].astype(str).str.zfill(5)
        agg_dict = {
            'spring_temp_anomaly_forecast': 'mean',
            'spring_precip_anomaly_forecast': 'mean',
            'spring_solar_rad_anomaly_forecast': 'mean',
            'summer_temp_anomaly_forecast': 'mean',
            'summer_precip_anomaly_forecast': 'mean',
            'summer_solar_rad_anomaly_forecast': 'mean'
        }
        # Add Extended Physics if available
        for opt in ['spring_evaporation_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
                    'summer_evaporation_anomaly_forecast', 'summer_soil_temp_l1_anomaly_forecast']:
            if opt in df_fcst.columns:
                agg_dict[opt] = 'mean'

        df_fcst_agg = df_fcst.groupby(['year', 'district_no']).agg(agg_dict).reset_index()
        df = pd.merge(df, df_fcst_agg, on=['year', 'district_no'], how='left')

    df_trend = pd.read_csv(paths['WALKFORWARD_FORECAST_CSV'])
    df_trend['district_no'] = df_trend['district_no'].astype(str).str.zfill(5)
    df_trend.rename(columns={'final_corrected_forecast': 'stat_trend_forecast'}, inplace=True)
    df = pd.merge(df, df_trend[['year', 'district_no', 'stat_trend_forecast']], on=['year', 'district_no'], how='left')

    nat_yield = df.groupby('year')['kreisYield'].mean().reset_index()
    nat_yield['national_avg_yield_lag1'] = nat_yield['kreisYield'].shift(1)
    df = pd.merge(df, nat_yield[['year', 'national_avg_yield_lag1']], on='year', how='left')

    logging.info("Merging WOFOST data...")
    df_wofost_yield = pd.read_csv(paths['WOFOST_ENSEMBLE_CSV'])
    df_wofost_yield['district_no'] = df_wofost_yield['district_no'].astype(str).str.zfill(5)

    DMC = config.WOFOST_CONFIG['CONSTANTS']['DMC_SUGARBEET']
    df_wofost_yield['yield_fresh'] = (df_wofost_yield['yield_water_limited'] / DMC) / 100.0

    df_cone = df_wofost_yield.groupby(['year', 'district_no']).agg(
        wofost_esp_mean=('yield_fresh', 'mean'),
        wofost_esp_p10=('yield_fresh', lambda x: x.quantile(0.10)),
        wofost_esp_p90=('yield_fresh', lambda x: x.quantile(0.90)),
        wofost_water_stress_mean=('cumulative_water_stress', 'mean')
    ).reset_index()

    df = pd.merge(df, df_cone, on=['year', 'district_no'], how='left')
    df['wofost_esp_mean'] = df['wofost_esp_mean'].fillna(df['stat_trend_forecast'])

    logging.info("Injecting Physics State...")
    df_init = pd.read_csv(paths['WOFOST_INITIAL_CONDITIONS'], dtype={'district_no': str})
    df_init['district_no'] = df_init['district_no'].str.zfill(5)
    df_init['sowing_date'] = pd.to_datetime(df_init['sowing_date'])
    df_init['wofost_sowing_doy'] = df_init['sowing_date'].dt.dayofyear
    df_init.rename(columns={'WAV': 'wofost_initial_wav'}, inplace=True)

    df = pd.merge(df, df_init[['year', 'district_no', 'wofost_sowing_doy', 'wofost_initial_wav']],
                  on=['year', 'district_no'], how='left')

    df_static = pd.read_csv(paths['WOFOST_STATIC_SITE_DATA'], dtype={'district_no': str})
    df_static['district_no'] = df_static['district_no'].str.zfill(5)
    phys_cols = ['district_no', 'year', 'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_phh2o_0_30cm', 'avg_elevation',
                 'latitude']
    phys_cols = [c for c in phys_cols if c in df_static.columns]

    df = pd.merge(df, df_static[phys_cols], on=['year', 'district_no'], how='left', suffixes=('', '_phys'))
    for col in ['avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_phh2o_0_30cm', 'avg_elevation', 'latitude']:
        if f'{col}_phys' in df.columns:
            df[col] = df[f'{col}_phys'].fillna(df.get(col, np.nan))
            df.drop(columns=[f'{col}_phys'], inplace=True)

    df_weather = process_antecedent_weather_only(paths['DAILY_WEATHER_DIR'], CONFIG['WEATHER_FEATURE_YEAR_START'],
                                                 CONFIG['WEATHER_FEATURE_YEAR_END'])
    if not df_weather.empty:
        cols_to_drop = [c for c in df_weather.columns if c in df.columns and c not in ['year', 'district_no']]
        if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
        df = pd.merge(df, df_weather, on=['year', 'district_no'], how='left')

    df_sat = pd.read_csv(paths['SATELLITE_FEATURES_CSV'], dtype={'district_no': str})
    df_sat['district_no'] = df_sat['district_no'].str.zfill(5)
    sat_cols = ['winter_cropland_ndvi_anomaly', 'winter_cropland_snow_cover_days']
    cols_to_drop = [c for c in df_sat.columns if c in df.columns and c not in ['year', 'district_no']]
    if cols_to_drop: df.drop(columns=cols_to_drop, inplace=True)
    df = pd.merge(df, df_sat[['year', 'district_no'] + [c for c in sat_cols if c in df_sat.columns]],
                  on=['year', 'district_no'], how='left')

    df_econ = load_and_process_economics(paths['PRODUCER_PRICE_CSV'], paths['INPUT_PRICE_CSV'])
    if not df_econ.empty:
        df = pd.merge(df, df_econ, on='year', how='left')
        for c in ['producer_price_index', 'fertilizer_price_index']:
            if c in df.columns:
                df[f'{c}_lag1'] = df.groupby('district_no')[c].shift(1).ffill().bfill()

    # --- CALCULATE WATER BALANCE (NEW) ---
    df = calculate_water_balance(df)

    # --- INTERACTIONS ---
    df['trend_vs_phys_gap'] = df['wofost_esp_mean'] - df['stat_trend_forecast']

    if 'latitude' in df.columns:
        df['is_vector_endemic_zone'] = (df['latitude'] < 52.0).astype(int)
        df['vector_pressure_local'] = df.get('winter_aphid_pressure', 0) * df['is_vector_endemic_zone']

    if 'avg_sand_0_100cm' in df.columns and 'antecedent_precip_sum' in df.columns:
        df['nitrogen_leaching_index'] = df['antecedent_precip_sum'] * df['avg_sand_0_100cm']

    if 'wofost_sowing_doy' in df.columns:
        mean_sowing = df.groupby('district_no')['wofost_sowing_doy'].transform('mean')
        df['sowing_anomaly'] = df['wofost_sowing_doy'] - mean_sowing

    if 'wofost_initial_wav' in df.columns:
        # Simple interaction: Initial Water * Clay (Buffer Capacity)
        df['deep_soil_buffer'] = df['wofost_initial_wav'] * df.get('avg_clay_0_100cm', 1.0)

    fill_cols = ['antecedent_precip_sum', 'sowing_potential_days', 'winter_aphid_pressure',
                 'winter_pest_kill_days', 'toxic_carryover_index', 'nitrogen_leaching_index']
    for c in fill_cols:
        if c in df.columns: df[c] = df[c].fillna(0)

    output_file = paths['OUTPUT_FILE']
    df.to_csv(output_file, index=False)

    logging.info(f"✓ Feature Engineering Complete (Water Balance). Saved to {output_file}")


if __name__ == '__main__':
    main()