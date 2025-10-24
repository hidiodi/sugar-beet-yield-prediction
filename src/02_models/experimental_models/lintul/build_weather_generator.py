# File: src/features/build_weather_generator.py
# Description: A complete, runnable script to train and save a stochastic weather generator.
# This version includes a robust generation method with bias correction.

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
PRECIP_THRESHOLD_MM = 0.3  # A more sensitive threshold for what constitutes a "wet" day


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

        for (district_no, month), group in tqdm(daily_df.groupby(['district_no', 'month']),
                                                desc="Learning Weather Patterns"):
            # Precipitation Markov Chain (transition probabilities)
            p01 = (group['is_wet'].shift(1) == 0) & (group['is_wet'] == 1)
            p11 = (group['is_wet'].shift(1) == 1) & (group['is_wet'] == 1)
            p00 = (group['is_wet'].shift(1) == 0) & (group['is_wet'] == 0)
            p10 = (group['is_wet'].shift(1) == 1) & (group['is_wet'] == 0)

            prob_wet_given_dry = p01.sum() / (p01.sum() + p00.sum()) if (p01.sum() + p00.sum()) > 0 else 0.1
            prob_wet_given_wet = p11.sum() / (p11.sum() + p10.sum()) if (p11.sum() + p10.sum()) > 0 else 0.5

            # Parameters for other variables
            self.stats[(district_no, month)] = {
                'p_wet_given_dry': prob_wet_given_dry,
                'p_wet_given_wet': prob_wet_given_wet,
                'precip_wet_day_mean': group[group['is_wet'] == 1]['precip'].mean(),
                'precip_wet_day_std': group[group['is_wet'] == 1]['precip'].std(),
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
                precip = max(0, precip)  # Ensure non-negative

            # Generate other variables
            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std'])
            tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin: tmax = tmin + np.abs(np.random.normal(0, month_stats['tmax_std']))  # Ensure tmax > tmin
            srad = np.random.normal(month_stats['srad_mean'], month_stats['srad_std'])
            srad = max(0, srad)

            generated_data.append({'date': date, 'tmin': tmin, 'tmax': tmax, 'precip': precip, 'srad': srad})
            yesterday_was_wet = today_is_wet

        if not generated_data:
            return pd.DataFrame()

        synthetic_df = pd.DataFrame(generated_data)
        synthetic_df['month'] = synthetic_df['date'].dt.month

        # --- Step 2: Apply Bias Correction ---
        # Adjust the generated sequence so its monthly means match the SEAS5 recipe.
        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month

            # Temperature Correction (Additive)
            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0)
            generated_t_mean = (synthetic_df.loc[month_mask, 'tmin'].mean() + synthetic_df.loc[
                month_mask, 'tmax'].mean()) / 2
            historical_t_mean = (self.stats[(district_no, month)]['tmin_mean'] + self.stats[(district_no, month)][
                'tmax_mean']) / 2

            # The correction factor is the forecast anomaly plus the difference needed to bring our random sample back to the historical mean.
            temp_correction = (historical_t_mean + temp_anomaly) - generated_t_mean
            synthetic_df.loc[month_mask, 'tmin'] += temp_correction
            synthetic_df.loc[month_mask, 'tmax'] += temp_correction

            # Precipitation Correction (Multiplicative)
            precip_anomaly_factor = 1 + monthly_anomalies.get(f'precip_anomaly_{month}',
                                                              0)  # e.g., anomaly of -0.2 -> factor of 0.8
            generated_precip_total = synthetic_df.loc[month_mask, 'precip'].sum()
            historical_precip_total = self.stats[(district_no, month)]['precip_mean'] * synthetic_df[month_mask].shape[
                0]

            target_precip_total = historical_precip_total * precip_anomaly_factor

            if generated_precip_total > 0:
                precip_correction_factor = target_precip_total / generated_precip_total
                synthetic_df.loc[month_mask, 'precip'] *= precip_correction_factor

        return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad']]


if __name__ == "__main__":
    logging.info("===== Training and Saving Weather Generator =====")
    try:
        if not os.path.exists(HISTORICAL_DAILY_WEATHER_PATH):
            raise FileNotFoundError(f"Required daily weather data not found at '{HISTORICAL_DAILY_WEATHER_PATH}'")

        df_daily = pd.read_csv(HISTORICAL_DAILY_WEATHER_PATH, parse_dates=['date'])
        # Ensure data types are correct
        df_daily['district_no'] = df_daily['district_no'].astype(str).str.zfill(5)

        wg = WeatherGenerator()
        wg.fit(df_daily)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(OUTPUT_WG_PATH), exist_ok=True)
        joblib.dump(wg, OUTPUT_WG_PATH)

        logging.info(f"✅ Weather Generator trained and saved successfully to {OUTPUT_WG_PATH}")

        # --- Optional: Run a test generation ---
        logging.info("\n--- Running a test generation for a sample district ---")
        test_district = df_daily['district_no'].iloc[0]
        test_anomalies = {
            'temp_anomaly_7': 2.5,  # Forecast a July that is 2.5 C warmer
            'precip_anomaly_7': -0.5,  # and 50% drier
        }
        test_weather = wg.generate(test_district, '2024-03-01', '2024-10-31', test_anomalies)

        logging.info(f"Generated {len(test_weather)} days for district {test_district}.")
        july_weather = test_weather[test_weather['date'].dt.month == 7]
        logging.info("Sample of generated July data:")
        print(july_weather.head())
        logging.info("Verification of July bias correction:")
        original_july_t_mean = (wg.stats[(test_district, 7)]['tmin_mean'] + wg.stats[(test_district, 7)][
            'tmax_mean']) / 2
        generated_july_t_mean = (july_weather['tmin'].mean() + july_weather['tmax'].mean()) / 2
        logging.info(f"Historical July T_Mean: {original_july_t_mean:.2f} C")
        logging.info(
            f"Target July T_Mean (Historical + Anomaly): {original_july_t_mean + test_anomalies['temp_anomaly_7']:.2f} C")
        logging.info(f"Generated July T_Mean: {generated_july_t_mean:.2f} C  <-- Should be very close to Target")

    except FileNotFoundError as e:
        logging.error(f"❌ FATAL: {e}")
        logging.error("Please provide the daily weather data file to train the generator.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)