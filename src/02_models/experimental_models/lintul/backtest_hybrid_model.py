# File: src/features/build_weather_generator.py
# Description: A complete, runnable script to train and save a stochastic weather generator.
# This version includes a robust generation method with bias correction.
# NOTE: This file is CROP-AGNOSTIC and requires NO CHANGES for the sugar beet model update.

import pandas as pd
import numpy as np
import joblib
import os
import logging
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')

# --- Configuration ---
# You must provide a CSV with historical DAILY weather data for this to work.
# Expected columns: 'date', 'district_no', 'tmin', 'tmax', 'precip', 'srad' (solar radiation)
HISTORICAL_DAILY_WEATHER_PATH = 'data/02_intermediate/historical_daily_weather_era5.csv'
OUTPUT_WG_PATH = 'src/models/weather_generator.joblib'
PRECIP_THRESHOLD_MM = 0.3  # A sensitive threshold for what constitutes a "wet" day


class WeatherGenerator:
    """
    A stochastic weather generator that learns from historical daily data and generates
    synthetic daily sequences conditioned on monthly climate model forecasts. This version
    uses a first-order Markov chain for precipitation and a normal distribution for other
    variables, with a final bias correction step.
    """

    def __init__(self):
        self.stats = defaultdict(dict)  # Stores learned stats for each district-month

    def fit(self, daily_df: pd.DataFrame):
        """Learns the monthly statistical properties of weather for each district."""
        logging.info("Fitting Weather Generator by learning historical monthly stats...")
        daily_df['month'] = daily_df['date'].dt.month
        daily_df['is_wet'] = (daily_df['precip'] > PRECIP_THRESHOLD_MM).astype(int)

        # Calculate historical monthly precip totals needed for bias correction later
        monthly_precip = daily_df.groupby(['district_no', 'month', daily_df['date'].dt.year])['precip'].sum().reset_index()
        avg_monthly_precip = monthly_precip.groupby(['district_no', 'month'])['precip'].mean().reset_index()

        for (district_no, month), group in tqdm(daily_df.groupby(['district_no', 'month']),
                                                desc="Learning Weather Patterns"):
            # Precipitation Markov Chain (transition probabilities)
            p01 = (group['is_wet'].shift(1) == 0) & (group['is_wet'] == 1)
            p11 = (group['is_wet'].shift(1) == 1) & (group['is_wet'] == 1)
            p00 = (group['is_wet'].shift(1) == 0) & (group['is_wet'] == 0)
            p10 = (group['is_wet'].shift(1) == 1) & (group['is_wet'] == 0)

            prob_wet_given_dry = p01.sum() / (p01.sum() + p00.sum()) if (p01.sum() + p00.sum()) > 0 else 0.1
            prob_wet_given_wet = p11.sum() / (p11.sum() + p10.sum()) if (p11.sum() + p10.sum()) > 0 else 0.5

            # Get the historical average precip total for this district and month for correction step
            hist_precip_total = avg_monthly_precip[
                (avg_monthly_precip['district_no'] == district_no) & (avg_monthly_precip['month'] == month)
            ]['precip'].values[0]

            # Parameters for other variables
            self.stats[(district_no, month)] = {
                'p_wet_given_dry': prob_wet_given_dry,
                'p_wet_given_wet': prob_wet_given_wet,
                'precip_wet_day_mean': group[group['is_wet'] == 1]['precip'].mean(),
                'precip_wet_day_std': group[group['is_wet'] == 1]['precip'].std(),
                'historical_precip_total': hist_precip_total, # Store for bias correction
                'tmin_mean': group['tmin'].mean(), 'tmin_std': group['tmin'].std(),
                'tmax_mean': group['tmax'].mean(), 'tmax_std': group['tmax'].std(),
                'srad_mean': group['srad'].mean(), 'srad_std': group['srad'].std(),
            }

    def generate(self, district_no: str, start_date_str: str, end_date_str: str, monthly_anomalies: dict):
        """
        Generates a synthetic daily weather sequence, including a bias correction step
        to match the monthly forecast anomalies.
        """
        dates = pd.to_datetime(pd.date_range(start=start_date_str, end=end_date_str, freq='D'))
        generated_data = []

        # Start with a 50/50 chance of the first day being wet
        yesterday_was_wet = np.random.rand() < 0.5

        # --- Step 1: Generate a raw, uncorrected daily sequence ---
        for date in dates:
            month = date.month
            key = (district_no, month)
            if key not in self.stats: continue
            month_stats = self.stats[key]

            # Determine precipitation state for today
            transition_prob = month_stats['p_wet_given_wet'] if yesterday_was_wet else month_stats['p_wet_given_dry']
            today_is_wet = np.random.rand() < transition_prob

            # Generate precipitation amount
            precip = 0.0
            if today_is_wet:
                precip = np.random.normal(month_stats['precip_wet_day_mean'], month_stats['precip_wet_day_std'])
                precip = max(0, precip)

            # Generate other variables
            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std'])
            tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin: tmax = tmin + np.abs(np.random.normal(0.5, month_stats['tmax_std']))
            srad = np.random.normal(month_stats['srad_mean'], month_stats['srad_std'])
            srad = max(0, srad)

            generated_data.append({'date': date, 'tmin': tmin, 'tmax': tmax, 'precip': precip, 'srad': srad})
            yesterday_was_wet = today_is_wet

        if not generated_data:
            return pd.DataFrame()

        synthetic_df = pd.DataFrame(generated_data)
        synthetic_df['month'] = synthetic_df['date'].dt.month

        # --- Step 2: Apply Bias Correction ---
        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month
            month_stats = self.stats.get((district_no, month))
            if not month_stats: continue

            # Temperature Correction (Additive)
            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0)
            generated_t_mean = (synthetic_df.loc[month_mask, 'tmin'].mean() + synthetic_df.loc[
                month_mask, 'tmax'].mean()) / 2
            historical_t_mean = (month_stats['tmin_mean'] + month_stats['tmax_mean']) / 2
            temp_correction = (historical_t_mean + temp_anomaly) - generated_t_mean
            synthetic_df.loc[month_mask, 'tmin'] += temp_correction
            synthetic_df.loc[month_mask, 'tmax'] += temp_correction

            # Precipitation Correction (Multiplicative)
            precip_anomaly_factor = 1 + monthly_anomalies.get(f'precip_anomaly_{month}', 0)
            generated_precip_total = synthetic_df.loc[month_mask, 'precip'].sum()
            target_precip_total = month_stats['historical_precip_total'] * precip_anomaly_factor

            if generated_precip_total > 0.1: # Avoid division by zero
                precip_correction_factor = target_precip_total / generated_precip_total
                synthetic_df.loc[month_mask, 'precip'] *= precip_correction_factor
                synthetic_df['precip'] = synthetic_df['precip'].clip(lower=0) # Ensure no negative precip

        return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad']]


if __name__ == "__main__":
    logging.info("===== Training and Saving Weather Generator =====")
    try:
        if not os.path.exists(HISTORICAL_DAILY_WEATHER_PATH):
            raise FileNotFoundError(f"Required daily weather data not found at '{HISTORICAL_DAILY_WEATHER_PATH}'")

        df_daily = pd.read_csv(HISTORICAL_DAILY_WEATHER_PATH, parse_dates=['date'])
        df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)

        wg = WeatherGenerator()
        wg.fit(df_daily)

        os.makedirs(os.path.dirname(OUTPUT_WG_PATH), exist_ok=True)
        joblib.dump(wg, OUTPUT_WG_PATH)

        logging.info(f"✅ Weather Generator trained and saved successfully to {OUTPUT_WG_PATH}")

    except FileNotFoundError as e:
        logging.error(f"❌ FATAL: {e}")
        logging.error("Please provide the daily weather data file to train the generator.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)