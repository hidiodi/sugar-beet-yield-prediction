# File: src/02_models/Wofost7.1/build_forecast_weather.py
# Description: FINAL VERSION. Writes partitioned parquet files, does NOT create a giant CSV.

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from scipy.stats import gamma
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import shutil

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG
# --- NEW: Define a permanent output directory for the partitioned data ---
FORECAST_PARTS_DIR = config.PROCESSED_DATA_DIR / 'forecast_weather_parts'


# --- WeatherGenerator Class (Unchanged) ---
class WeatherGenerator:
    def __init__(self):
        self.stats = defaultdict(dict)
        self.PRECIP_THRESHOLD_MM = CONFIG['WEATHER_GENERATOR']['PRECIP_THRESHOLD_MM']
        self.MIN_SRAD = CONFIG['WEATHER_GENERATOR']['MIN_SRAD']

    def fit(self, daily_df: pd.DataFrame):
        daily_df = daily_df.copy()
        daily_df['district_no'] = daily_df['district_no'].astype(str).str.zfill(5)
        daily_df['month'] = daily_df['date'].dt.month
        if 'precip' not in daily_df.columns and 'prec' in daily_df.columns: daily_df.rename(columns={'prec': 'precip'},
                                                                                            inplace=True)
        daily_df['is_wet'] = (daily_df['precip'] > self.PRECIP_THRESHOLD_MM).astype(int)
        if 'vap' not in daily_df.columns: daily_df['vap'] = 1.0
        daily_df['vap'] = daily_df['vap'].fillna(1.0)
        if 'wind' not in daily_df.columns: daily_df['wind'] = 2.0
        daily_df['wind'] = daily_df['wind'].fillna(2.0)
        for (district_no, month), group in tqdm(daily_df.groupby(['district_no', 'month']),
                                                desc="Learning Weather Patterns"):
            p01 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 1)).sum();
            p00 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 0)).sum()
            p11 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 1)).sum();
            p10 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 0)).sum()
            prob_wet_given_dry = p01 / (p01 + p00) if (p01 + p00) > 0 else 0.1;
            prob_wet_given_wet = p11 / (p11 + p10) if (p11 + p10) > 0 else 0.5
            wet_day_precip = group[group['is_wet'] == 1]['precip']
            if len(wet_day_precip) > 2:
                a, loc, b = gamma.fit(wet_day_precip, floc=0); gamma_shape, gamma_scale = a, b
            else:
                gamma_shape, gamma_scale = (1.0, wet_day_precip.mean() or 1.0)
            self.stats[(district_no, month)] = {'p_wet_given_dry': prob_wet_given_dry,
                                                'p_wet_given_wet': prob_wet_given_wet,
                                                'precip_gamma_shape': gamma_shape, 'precip_gamma_scale': gamma_scale,
                                                'precip_mean': group['precip'].mean(),
                                                'tmin_mean': group['tmin'].mean(),
                                                'tmin_std': max(group['tmin'].std(), 0.5),
                                                'tmax_mean': group['tmax'].mean(),
                                                'tmax_std': max(group['tmax'].std(), 0.5),
                                                'srad_mean': group['srad'].mean(),
                                                'srad_std': max(group['srad'].std(), 0.5),
                                                'vap_mean': group['vap'].mean(),
                                                'vap_std': max(group['vap'].std(), 0.1),
                                                'wind_mean': group['wind'].mean(),
                                                'wind_std': max(group['wind'].std(), 0.5)}

    def generate(self, district_no: str, start_date_str: str, end_date_str: str, monthly_anomalies: dict):
        dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D');
        generated_data = [];
        yesterday_was_wet = np.random.rand() < 0.5
        for date in dates:
            month, key = date.month, (str(district_no).zfill(5), date.month)
            if key not in self.stats: continue
            month_stats = self.stats[key];
            transition_prob = month_stats['p_wet_given_wet'] if yesterday_was_wet else month_stats['p_wet_given_dry']
            today_is_wet = np.random.rand() < transition_prob;
            precip = 0.0
            if today_is_wet: precip = max(0, gamma.rvs(a=month_stats['precip_gamma_shape'],
                                                       scale=month_stats['precip_gamma_scale'], size=1)[0])
            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std']);
            tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin: tmax = tmin + abs(np.random.normal(0, 1.0))
            srad = max(self.MIN_SRAD, np.random.normal(month_stats['srad_mean'], month_stats['srad_std']));
            vap = max(0.1, np.random.normal(month_stats['vap_mean'], month_stats['vap_std']));
            wind = max(0.0, np.random.normal(month_stats['wind_mean'], month_stats['wind_std']))
            generated_data.append(
                {'date': date, 'tmin': tmin, 'tmax': tmax, 'precip': precip, 'srad': srad, 'vap': vap, 'wind': wind});
            yesterday_was_wet = today_is_wet
        if not generated_data: return pd.DataFrame()
        synthetic_df = pd.DataFrame(generated_data);
        synthetic_df['month'] = synthetic_df['date'].dt.month
        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month;
            key = (str(district_no).zfill(5), month)
            if key not in self.stats: continue
            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0);
            hist_tmean = (self.stats[key]['tmin_mean'] + self.stats[key]['tmax_mean']) / 2
            synth_tmean = (synthetic_df.loc[month_mask, 'tmin'].mean() + synthetic_df.loc[
                month_mask, 'tmax'].mean()) / 2
            temp_correction = (hist_tmean + temp_anomaly) - synth_tmean;
            synthetic_df.loc[month_mask, ['tmin', 'tmax']] += temp_correction
            hist_precip_mean_daily = self.stats[key].get('precip_mean', 0);
            forecast_precip_anomaly_daily = monthly_anomalies.get(f'precip_anomaly_{month}', 0)
            target_precip_daily = max(0.0, hist_precip_mean_daily + forecast_precip_anomaly_daily);
            target_precip_total_month = target_precip_daily * month_mask.sum()
            synth_precip = synthetic_df.loc[month_mask, 'precip'].sum()
            if synth_precip > 0:
                scaling_factor = target_precip_total_month / synth_precip; synthetic_df.loc[
                    month_mask, 'precip'] *= scaling_factor
            elif target_precip_total_month > 0:
                synthetic_df.loc[
                    month_mask & (synthetic_df['precip'] == 0), 'precip'] = target_precip_total_month / month_mask.sum()
        return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad', 'vap', 'wind']]


def generate_for_district_year(group_name, group_df, wg, output_dir):
    district_no, year = group_name;
    all_members_for_group = [];
    output_path = Path(output_dir) / f'forecast_{district_no}_{year}.parquet'
    month_map = {3: 'march', 4: 'april', 5: 'may', 6: 'june', 7: 'july', 8: 'august', 9: 'september', 10: 'october'}
    start_date, end_date = f'{year}-03-01', f'{year}-11-30'
    for _, member_row in group_df.iterrows():
        member = member_row['seas5_member'];
        monthly_anomalies = {}
        for month_num, month_name in month_map.items():
            temp_col = f'{month_name}_temp_anomaly_forecast';
            precip_col = f'{month_name}_precip_anomaly_forecast'
            monthly_anomalies[f'temp_anomaly_{month_num}'] = member_row.get(temp_col, 0)
            monthly_anomalies[f'precip_anomaly_{month_num}'] = member_row.get(precip_col, 0)
        synth_weather = wg.generate(district_no, start_date, end_date, monthly_anomalies)
        if not synth_weather.empty:
            synth_weather['district_no'] = district_no;
            synth_weather['year'] = year;
            synth_weather['member'] = member
            all_members_for_group.append(synth_weather)
    if all_members_for_group:
        result_df = pd.concat(all_members_for_group, ignore_index=True);
        result_df.to_parquet(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate forecast weather ensembles.")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Limit the run to a specific number of district-year groups for testing.")
    args = parser.parse_args()
    logging.info("--- Building Forecast Weather (Writing Partitioned Parquet Files) ---")

    # --- SETUP OUTPUT DIRECTORY ---
    if FORECAST_PARTS_DIR.exists():
        logging.warning(f"Output directory '{FORECAST_PARTS_DIR}' exists. Deleting and regenerating.")
        shutil.rmtree(FORECAST_PARTS_DIR)
    FORECAST_PARTS_DIR.mkdir(parents=True)

    try:
        # Load inputs
        weather_path = Path(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR']);
        weather_files = list(weather_path.glob("*.csv"))
        historical_weather_df = pd.concat(
            (pd.read_csv(f, parse_dates=['date']) for f in tqdm(weather_files, desc="Loading Historical Weather")),
            ignore_index=True)
        df_seas5_all = pd.read_csv(CONFIG['FILE_PATHS']['SEAS5_MEMBER_FEATURES'], dtype={'district_no': str})

        if args.limit:
            logging.warning(f"--- !!! TEST RUN !!! Limiting to {args.limit} district-year groups. ---")
            unique_groups = df_seas5_all[['district_no', 'year']].drop_duplicates().head(args.limit)
            df_seas5_all = pd.merge(df_seas5_all, unique_groups, on=['district_no', 'year'], how='inner')

        wg = WeatherGenerator();
        wg.fit(historical_weather_df)
        logging.info("Grouping forecast tasks by district and year...")
        grouped_tasks = df_seas5_all.groupby(['district_no', 'year'])
        logging.info(f"Generating forecast ensembles in parallel for {len(grouped_tasks)} groups...")

        # --- EXECUTE and write partitioned files ---
        Parallel(n_jobs=-1)(
            delayed(generate_for_district_year)(group_name, group_df, wg, FORECAST_PARTS_DIR)
            for group_name, group_df in tqdm(grouped_tasks)
        )

        # --- NO AGGREGATION. WE ARE DONE. ---
        logging.info(f"--- SUCCESS: Partitioned forecast weather files saved to '{FORECAST_PARTS_DIR}' ---")

    except Exception as e:
        logging.error(f"FATAL: An error occurred. Error: {e}", exc_info=True);
        sys.exit(1)


if __name__ == "__main__":
    main()