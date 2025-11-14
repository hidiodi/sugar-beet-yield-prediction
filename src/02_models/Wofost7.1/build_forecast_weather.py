# File: src/02_models/Wofost7.1/build_forecast_weather.py
# Description: Pre-generates all synthetic weather for all simulations.

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from scipy.stats import gamma
from collections import defaultdict
from tqdm import tqdm

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG


class WeatherGenerator:
    def __init__(self):
        self.stats = defaultdict(dict)
        self.PRECIP_THRESHOLD_MM = CONFIG['WEATHER_GENERATOR']['PRECIP_THRESHOLD_MM']
        self.MIN_SRAD = CONFIG['WEATHER_GENERATOR']['MIN_SRAD']

    def fit(self, daily_df: pd.DataFrame):
        daily_df = daily_df.copy()
        daily_df['district_no'] = daily_df['district_no'].astype(str).str.zfill(5)
        daily_df['month'] = daily_df['date'].dt.month
        daily_df['is_wet'] = (daily_df['precip'] > self.PRECIP_THRESHOLD_MM).astype(int)

        if 'vap' not in daily_df.columns: daily_df['vap'] = 1.0
        daily_df['vap'] = daily_df['vap'].fillna(1.0)
        if 'wind' not in daily_df.columns: daily_df['wind'] = 2.0
        daily_df['wind'] = daily_df['wind'].fillna(2.0)

        for (district_no, month), group in tqdm(daily_df.groupby(['district_no', 'month']), desc="Learning Weather Patterns"):
            p01 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 1)).sum()
            p00 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 0)).sum()
            p11 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 1)).sum()
            p10 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 0)).sum()
            prob_wet_given_dry = p01 / (p01 + p00) if (p01 + p00) > 0 else 0.1
            prob_wet_given_wet = p11 / (p11 + p10) if (p11 + p10) > 0 else 0.5
            wet_day_precip = group[group['is_wet'] == 1]['precip']
            if len(wet_day_precip) > 2:
                a, loc, b = gamma.fit(wet_day_precip, floc=0)
                gamma_shape, gamma_scale = a, b
            else:
                gamma_shape, gamma_scale = (1.0, wet_day_precip.mean() or 1.0)

            self.stats[(district_no, month)] = {
                'p_wet_given_dry': prob_wet_given_dry, 'p_wet_given_wet': prob_wet_given_wet,
                'precip_gamma_shape': gamma_shape, 'precip_gamma_scale': gamma_scale,
                'precip_mean': group['precip'].mean(),
                'tmin_mean': group['tmin'].mean(), 'tmin_std': max(group['tmin'].std(), 0.5),
                'tmax_mean': group['tmax'].mean(), 'tmax_std': max(group['tmax'].std(), 0.5),
                'srad_mean': group['srad'].mean(), 'srad_std': max(group['srad'].std(), 0.5),
                'vap_mean': group['vap'].mean(), 'vap_std': max(group['vap'].std(), 0.1),
                'wind_mean': group['wind'].mean(), 'wind_std': max(group['wind'].std(), 0.5)
            }

    def generate(self, district_no: str, start_date_str: str, end_date_str: str, monthly_anomalies: dict):
        dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
        generated_data = []
        yesterday_was_wet = np.random.rand() < 0.5
        for date in dates:
            month, key = date.month, (str(district_no).zfill(5), date.month)
            if key not in self.stats: continue
            month_stats = self.stats[key]

            transition_prob = month_stats['p_wet_given_wet'] if yesterday_was_wet else month_stats['p_wet_given_dry']
            today_is_wet = np.random.rand() < transition_prob
            precip = 0.0
            if today_is_wet: precip = max(0, gamma.rvs(a=month_stats['precip_gamma_shape'], scale=month_stats['precip_gamma_scale'], size=1)[0])

            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std'])
            tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin: tmax = tmin + abs(np.random.normal(0, 1.0))
            srad = max(self.MIN_SRAD, np.random.normal(month_stats['srad_mean'], month_stats['srad_std']))
            vap = max(0.1, np.random.normal(month_stats['vap_mean'], month_stats['vap_std']))
            wind = max(0.0, np.random.normal(month_stats['wind_mean'], month_stats['wind_std']))

            generated_data.append({'date': date, 'tmin': tmin, 'tmax': tmax, 'precip': precip, 'srad': srad, 'vap': vap, 'wind': wind})
            yesterday_was_wet = today_is_wet

        if not generated_data: return pd.DataFrame()

        synthetic_df = pd.DataFrame(generated_data)
        synthetic_df['month'] = synthetic_df['date'].dt.month

        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month
            key = (str(district_no).zfill(5), month)
            if key not in self.stats: continue

            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0)
            hist_tmean = (self.stats[key]['tmin_mean'] + self.stats[key]['tmax_mean']) / 2
            synth_tmean = (synthetic_df.loc[month_mask, 'tmin'].mean() + synthetic_df.loc[month_mask, 'tmax'].mean()) / 2
            temp_correction = (hist_tmean + temp_anomaly) - synth_tmean
            synthetic_df.loc[month_mask, ['tmin', 'tmax']] += temp_correction

            hist_precip_mean_daily = self.stats[key].get('precip_mean', 0)
            forecast_precip_anomaly_daily = monthly_anomalies.get(f'precip_anomaly_{month}', 0)
            target_precip_daily = max(0.0, hist_precip_mean_daily + forecast_precip_anomaly_daily)
            target_precip_total_month = target_precip_daily * month_mask.sum()
            synth_precip = synthetic_df.loc[month_mask, 'precip'].sum()
            if synth_precip > 0:
                scaling_factor = target_precip_total_month / synth_precip
                synthetic_df.loc[month_mask, 'precip'] *= scaling_factor
            elif target_precip_total_month > 0:
                synthetic_df.loc[month_mask & (synthetic_df['precip'] == 0), 'precip'] = target_precip_total_month / month_mask.sum()

        return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad', 'vap', 'wind']]


def main():
    """
    Main function to build the forecast weather file.
    """
    logging.info("--- Building Forecast Weather ---")

    # Load inputs
    try:
        historical_weather_df = pd.read_csv(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'], parse_dates=['date'], dtype={'district_no': str})
        df_seas5_all = pd.read_csv(CONFIG['FILE_PATHS']['SEAS5_MEMBER_FEATURES'], dtype={'district_no': str})
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required data file was not found. Error: {e}"); sys.exit(1)

    # Fit WeatherGenerator
    wg = WeatherGenerator()
    wg.fit(historical_weather_df)

    all_forecasts = []
    # Loop through every district, year, and member
    for _, member_row in tqdm(df_seas5_all.iterrows(), total=len(df_seas5_all), desc="Generating Forecast Ensembles"):
        year = member_row['year']
        district_no = member_row['district_no']
        member = member_row['seas5_member']

        monthly_anomalies = {}
        spring_temp_anomaly = member_row.get('spring_temp_anomaly_forecast', 0)
        summer_temp_anomaly = member_row.get('summer_temp_anomaly_forecast', 0)
        spring_precip_anomaly = member_row.get('spring_precip_anomaly_forecast', 0)
        summer_precip_anomaly = member_row.get('summer_precip_anomaly_forecast', 0)

        for month in range(3, 11): # March to October
            if month in [3, 4, 5]: # Spring
                monthly_anomalies[f'temp_anomaly_{month}'] = spring_temp_anomaly
                monthly_anomalies[f'precip_anomaly_{month}'] = spring_precip_anomaly
            elif month in [6, 7, 8]: # Summer
                monthly_anomalies[f'temp_anomaly_{month}'] = summer_temp_anomaly
                monthly_anomalies[f'precip_anomaly_{month}'] = summer_precip_anomaly
            else: # Autumn
                monthly_anomalies[f'temp_anomaly_{month}'] = summer_temp_anomaly
                monthly_anomalies[f'precip_anomaly_{month}'] = summer_precip_anomaly

        start_date, end_date = f'{year}-03-01', f'{year}-11-30'
        synth_weather = wg.generate(district_no, start_date, end_date, monthly_anomalies)

        if not synth_weather.empty:
            synth_weather['district_no'] = district_no
            synth_weather['year'] = year
            synth_weather['member'] = member
            all_forecasts.append(synth_weather)

    output_df = pd.concat(all_forecasts, ignore_index=True)
    output_path = config.PROCESSED_DATA_DIR / 'ForecastedWeather.csv'
    output_df.to_csv(output_path, index=False)
    logging.info(f"--- ForecastedWeather.csv saved to {output_path} ---")
    logging.info("Data Preview:\n" + output_df.head().to_string())

if __name__ == "__main__":
    main()
