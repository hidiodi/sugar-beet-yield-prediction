# File: src/02_models/Wofost7.1/build_forecast_weather.py
# Description: Smart-ESP Weather Engine.
# OPTIMIZED (v5.1): Respects 'DISTRICT_LIMIT' for fast testing.

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG
FORECAST_PARTS_DIR = config.PROCESSED_DATA_DIR / 'forecast_weather_parts'

ANALOG_WEIGHTS = {
    'spring_temp': 1.0, 'spring_precip': 0.3,
    'summer_temp': 1.2, 'summer_precip': 0.3
}


class AnalogYearSelector:
    def __init__(self, full_hist_df):
        self.history = full_hist_df.copy()
        self.library = self._build_anomaly_library()

    def _build_anomaly_library(self):
        logging.info("Building Historical Anomaly Library...")
        df = self.history.copy()
        df['month'] = df['date'].dt.month
        climatology = df.groupby(['district_no', 'month'])[['tmean', 'prec']].mean().reset_index()
        climatology.rename(columns={'tmean': 'clim_tmean', 'prec': 'clim_prec'}, inplace=True)
        df = pd.merge(df, climatology, on=['district_no', 'month'], how='left')
        df['temp_anom'] = df['tmean'] - df['clim_tmean']
        df['prec_anom'] = df['prec'] - df['clim_prec']

        df['season'] = 'other'
        df.loc[df['month'].isin([3, 4, 5]), 'season'] = 'spring'
        df.loc[df['month'].isin([6, 7, 8]), 'season'] = 'summer'

        seasonal = df[df['season'] != 'other'].groupby(['district_no', 'year', 'season']).agg(
            temp_anom=('temp_anom', 'mean'), prec_anom=('prec_anom', 'mean')
        ).unstack()
        seasonal.columns = [f'{s}_{v}' for v, s in seasonal.columns]
        return seasonal.reset_index().fillna(0)

    def find_best_match(self, district_no, target_year, forecast_row):
        candidates = self.library[
            (self.library['district_no'] == district_no) &
            (self.library['year'] < target_year)
            ].copy()
        if candidates.empty: return target_year - 1

        d_spt = (candidates['spring_temp_anom'] - forecast_row.get('spring_temp_anomaly_forecast', 0)) ** 2
        d_spp = (candidates['spring_prec_anom'] - forecast_row.get('spring_precip_anomaly_forecast', 0)) ** 2
        d_sut = (candidates['summer_temp_anom'] - forecast_row.get('summer_temp_anomaly_forecast', 0)) ** 2
        d_sup = (candidates['summer_prec_anom'] - forecast_row.get('summer_precip_anomaly_forecast', 0)) ** 2

        candidates['score'] = (
                d_spt * ANALOG_WEIGHTS['spring_temp'] + d_spp * ANALOG_WEIGHTS['spring_precip'] +
                d_sut * ANALOG_WEIGHTS['summer_temp'] + d_sup * ANALOG_WEIGHTS['summer_precip']
        )
        return int(candidates.loc[candidates['score'].idxmin(), 'year'])

    def get_weather_trace(self, district_no, year):
        return self.history[
            (self.history['district_no'] == district_no) &
            (self.history['year'] == year)
            ].copy()


def generate_forecasts_for_group(group_name, group_df, analog_engine, output_dir):
    district_no, target_year = group_name
    forecast_rows = []

    for _, member_row in group_df.iterrows():
        member_id = member_row['seas5_member']
        analog_year = analog_engine.find_best_match(district_no, target_year, member_row)
        daily_data = analog_engine.get_weather_trace(district_no, analog_year)
        if daily_data.empty: continue

        daily_data = daily_data.copy()
        mask = daily_data['date'].dt.month >= 3
        daily_data = daily_data[mask]

        is_target_leap = (target_year % 4 == 0) and (target_year % 100 != 0 or target_year % 400 == 0)
        new_dates = []
        valid_indices = []

        for idx, row in daily_data.iterrows():
            d = row['date']
            if d.month == 2 and d.day == 29 and not is_target_leap: continue
            try:
                new_dates.append(pd.Timestamp(year=target_year, month=d.month, day=d.day))
                valid_indices.append(idx)
            except ValueError:
                continue

        mapped_data = daily_data.loc[valid_indices].copy()
        mapped_data['date'] = new_dates
        mapped_data['year'] = target_year
        mapped_data['member'] = member_id
        mapped_data['analog_source_year'] = analog_year
        forecast_rows.append(mapped_data)

    if forecast_rows:
        result_df = pd.concat(forecast_rows, ignore_index=True)
        if 'prec' in result_df.columns: result_df.rename(columns={'prec': 'precip'}, inplace=True)
        if 'rad' in result_df.columns: result_df.rename(columns={'rad': 'srad'}, inplace=True)
        if 'wind' not in result_df.columns: result_df['wind'] = 2.0
        if 'vap' not in result_df.columns: result_df['vap'] = 1.0

        output_path = Path(output_dir) / f'forecast_{district_no}_{target_year}.parquet'
        result_df.to_parquet(output_path, index=False)


def main():
    logging.info("--- Building Smart-ESP Forecast Weather (Optimized) ---")

    if FORECAST_PARTS_DIR.exists():
        shutil.rmtree(FORECAST_PARTS_DIR)
    FORECAST_PARTS_DIR.mkdir(parents=True)

    try:
        # 1. Load Forecasts & Filter
        logging.info("Loading ECMWF Forecast Members...")
        df_seas5 = pd.read_csv(CONFIG['FILE_PATHS']['SEAS5_MEMBER_FEATURES'], dtype={'district_no': str})
        df_seas5['district_no'] = df_seas5['district_no'].str.zfill(5)

        # --- OPTIMIZATION ---
        limit = CONFIG.get('DISTRICT_LIMIT')
        if limit:
            logging.warning(f"!!! TEST MODE: Limiting to top {limit} districts !!!")
            target_districts = sorted(df_seas5['district_no'].unique())[:limit]
            df_seas5 = df_seas5[df_seas5['district_no'].isin(target_districts)]
        else:
            target_districts = None
        # --------------------

        # 2. Load History & Filter
        logging.info("Loading Historical Weather Database...")
        weather_path = Path(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'])
        files = list(weather_path.glob("*.csv"))
        full_hist_df = pd.concat(
            (pd.read_csv(f, parse_dates=['date'], dtype={'district_no': str}) for f in
             tqdm(files, desc="Loading History")),
            ignore_index=True
        )
        full_hist_df['district_no'] = full_hist_df['district_no'].str.zfill(5)

        # --- OPTIMIZATION ---
        if target_districts:
            full_hist_df = full_hist_df[full_hist_df['district_no'].isin(target_districts)]
        # --------------------

        full_hist_df['year'] = full_hist_df['date'].dt.year
        if 'tmean' not in full_hist_df.columns:
            full_hist_df['tmean'] = (full_hist_df['tmin'] + full_hist_df['tmax']) / 2
        if 'precip' in full_hist_df.columns and 'prec' not in full_hist_df.columns:
            full_hist_df.rename(columns={'precip': 'prec'}, inplace=True)

        engine = AnalogYearSelector(full_hist_df)

        # 3. Run
        logging.info("Generating Analog Ensembles...")
        grouped_tasks = df_seas5.groupby(['district_no', 'year'])

        Parallel(n_jobs=-1)(
            delayed(generate_forecasts_for_group)(
                name, group, engine, FORECAST_PARTS_DIR
            ) for name, group in tqdm(grouped_tasks, desc="Processing Districts")
        )

        logging.info(f"--- Success. Forecasts saved to {FORECAST_PARTS_DIR} ---")

    except Exception as e:
        logging.error(f"FATAL: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()