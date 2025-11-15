# File: src/02_models/Wofost7.1/build_forecast_weather.py
# Description: FINAL CORRECTED VERSION 4. Fixes the analog year logic by perfectly
# replicating the anomaly handling and conditional application from the parent pipeline.

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
FORECAST_PARTS_DIR = config.PROCESSED_DATA_DIR / 'forecast_weather_parts'


# --- WeatherGenerator Class (Unchanged from original) ---
class WeatherGenerator:
    """
    A class to generate synthetic daily weather data based on learned monthly statistics from historical data.
    It can generate baseline weather and then apply monthly anomalies for temperature and precipitation.
    """

    def __init__(self):
        self.stats = defaultdict(dict)
        self.PRECIP_THRESHOLD_MM = CONFIG['WEATHER_GENERATOR']['PRECIP_THRESHOLD_MM']
        self.MIN_SRAD = CONFIG['WEATHER_GENERATOR']['MIN_SRAD']

    def fit(self, daily_df: pd.DataFrame):
        """
        Learns the statistical properties of weather for each district and month from historical data.
        """
        daily_df = daily_df.copy()
        # Pre-computation and validation
        if 'precip' in daily_df.columns and daily_df['precip'].mean() < 1.0 and daily_df['precip'].mean() != 0:
            daily_df['precip'] *= 10  # Ad-hoc fix for unit issues if detected
        daily_df['district_no'] = daily_df['district_no'].astype(str).str.zfill(5)
        daily_df['month'] = daily_df['date'].dt.month
        daily_df['is_wet'] = (daily_df['precip'] > self.PRECIP_THRESHOLD_MM).astype(int)
        if 'vap' not in daily_df.columns: daily_df['vap'] = 1.0
        daily_df['vap'] = daily_df['vap'].fillna(1.0)
        if 'wind' not in daily_df.columns: daily_df['wind'] = 2.0
        daily_df['wind'] = daily_df['wind'].fillna(2.0)

        # Group and compute stats
        for (district_no, month), group in daily_df.groupby(['district_no', 'month']):
            # Precipitation transition probabilities (Markov chain)
            p01 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 1)).sum()
            p00 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 0)).sum()
            p11 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 1)).sum()
            p10 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 0)).sum()
            prob_wet_given_dry = p01 / (p01 + p00) if (p01 + p00) > 0 else 0.1
            prob_wet_given_wet = p11 / (p11 + p10) if (p11 + p10) > 0 else 0.5

            # Precipitation amount on wet days (Gamma distribution)
            wet_day_precip = group[group['is_wet'] == 1]['precip']
            if len(wet_day_precip) > 2:
                a, loc, b = gamma.fit(wet_day_precip, floc=0)
                gamma_shape, gamma_scale = a, b
            else:
                gamma_shape, gamma_scale = (1.0, wet_day_precip.mean() or 1.0)

            # Store all statistics
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
        """
        Generates a synthetic weather timeseries for a given period and applies specified anomalies.
        """
        dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
        generated_data = []
        yesterday_was_wet = np.random.rand() < 0.5

        # --- 1. Generate Baseline Synthetic Weather ---
        for date in dates:
            month, key = date.month, (str(district_no).zfill(5), date.month)
            if key not in self.stats: continue
            month_stats = self.stats[key]
            transition_prob = month_stats['p_wet_given_wet'] if yesterday_was_wet else month_stats['p_wet_given_dry']
            today_is_wet = np.random.rand() < transition_prob
            precip = 0.0
            if today_is_wet:
                precip = max(0, gamma.rvs(a=month_stats['precip_gamma_shape'], scale=month_stats['precip_gamma_scale'],
                                          size=1)[0])
            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std'])
            tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin: tmax = tmin + abs(np.random.normal(0, 1.0))  # Ensure Tmax > Tmin
            srad = max(self.MIN_SRAD, np.random.normal(month_stats['srad_mean'], month_stats['srad_std']))
            vap = max(0.1, np.random.normal(month_stats['vap_mean'], month_stats['vap_std']))
            wind = max(0.0, np.random.normal(month_stats['wind_mean'], month_stats['wind_std']))
            generated_data.append(
                {'date': date, 'tmin': tmin, 'tmax': tmax, 'precip': precip, 'srad': srad, 'vap': vap, 'wind': wind})
            yesterday_was_wet = today_is_wet

        if not generated_data: return pd.DataFrame()
        synthetic_df = pd.DataFrame(generated_data)
        synthetic_df['month'] = synthetic_df['date'].dt.month

        # --- 2. Apply Anomalies (if provided) ---
        if not monthly_anomalies:
            return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad', 'vap', 'wind']]

        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month
            key = (str(district_no).zfill(5), month)
            if key not in self.stats: continue

            # Temperature Anomaly Correction
            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0)
            hist_tmean = (self.stats[key]['tmin_mean'] + self.stats[key]['tmax_mean']) / 2
            synth_tmean = (synthetic_df.loc[month_mask, 'tmin'].mean() + synthetic_df.loc[
                month_mask, 'tmax'].mean()) / 2
            temp_correction = (hist_tmean + temp_anomaly) - synth_tmean
            synthetic_df.loc[month_mask, ['tmin', 'tmax']] += temp_correction

            # Precipitation Anomaly Correction
            hist_precip_mean_daily = self.stats[key].get('precip_mean', 0)
            forecast_precip_anomaly_daily = monthly_anomalies.get(f'precip_anomaly_{month}', 0)
            target_precip_daily = max(0.0, hist_precip_mean_daily + forecast_precip_anomaly_daily)
            target_precip_total_month = target_precip_daily * month_mask.sum()
            synth_precip = synthetic_df.loc[month_mask, 'precip'].sum()
            if synth_precip > 0:
                scaling_factor = target_precip_total_month / synth_precip
                synthetic_df.loc[month_mask, 'precip'] *= scaling_factor
            elif target_precip_total_month > 0:
                synthetic_df.loc[
                    month_mask & (synthetic_df['precip'] == 0), 'precip'] = target_precip_total_month / month_mask.sum()

        return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad', 'vap', 'wind']]


def generate_forecasts_for_group(group_name, group_df, full_hist_weather_df, analog_config, output_dir):
    """
    Main worker function to generate weather for a specific district-year group.
    Contains the corrected logic for analog year selection and conditional anomaly application.
    """
    district_no, year = group_name

    district_past_weather = full_hist_weather_df[
        (full_hist_weather_df['district_no'] == district_no) &
        (full_hist_weather_df['year'] < year)
        ].copy()

    wg = WeatherGenerator()
    apply_anomalies = True  # Default to applying anomalies (fallback case)

    # ==========================================================================
    # === START: ROBUST ANALOG YEAR & FALLBACK LOGIC (REPLICATED FROM PIPELINE) ===
    # ==========================================================================
    if district_past_weather.empty or len(district_past_weather['year'].unique()) < analog_config['MIN_YEARS_FOR_FIT']:
        logging.warning(f"Not enough past data for analog search for {district_no}-{year}. "
                        "Using generic climatology for the district as a fallback.")
        # Fallback: Use all available history for this district to build a generic model.
        generic_district_weather = full_hist_weather_df[full_hist_weather_df['district_no'] == district_no].copy()
        if generic_district_weather.empty:
            logging.error(f"CRITICAL: No weather data found AT ALL for district {district_no}. Cannot generate.")
            return
        wg.fit(generic_district_weather)
        apply_anomalies = True  # Explicitly set to apply anomalies for this fallback WG
    else:
        # --- This is the "expert" logic, which runs when enough past data exists ---
        apply_anomalies = False  # If this logic succeeds, we use an expert WG and don't apply anomalies

        # 1. Calculate historical climatology
        climatology = district_past_weather.groupby('month')[['tmin', 'tmax', 'precip']].mean()
        climatology['temp'] = (climatology['tmin'] + climatology['tmax']) / 2

        # 2. **FIXED**: Create the target forecast profile using seasonal anomalies
        target_forecast = group_df.iloc[0]  # Use first member as reference for forecast
        target_anomalies = {}
        for month in range(3, 11):  # March to October
            if month in [3, 4, 5]:  # Spring
                target_anomalies[month] = {
                    'temp': target_forecast.get('spring_temp_anomaly_forecast', 0.0),
                    'precip': target_forecast.get('spring_precip_anomaly_forecast', 0.0)
                }
            else:  # Summer & Autumn
                target_anomalies[month] = {
                    'temp': target_forecast.get('summer_temp_anomaly_forecast', 0.0),
                    'precip': target_forecast.get('summer_precip_anomaly_forecast', 0.0)
                }

        # 3. Build the target weather profile by applying anomalies to climatology
        target_weather = climatology.copy()
        for month, anomalies in target_anomalies.items():
            if month in target_weather.index:
                target_weather.loc[month, 'temp'] += anomalies['temp']
                # Note: This is anomaly in daily average, not total
                target_weather.loc[month, 'precip'] += anomalies['precip']

                # 4. Find the closest historical "analog" years
        yearly_avg = district_past_weather.groupby(['year', 'month'])[['precip', 'tmin', 'tmax']].mean().reset_index()
        yearly_avg['temp'] = (yearly_avg['tmin'] + yearly_avg['tmax']) / 2
        hist_pivot = yearly_avg.pivot_table(index='year', columns='month', values=['temp', 'precip'])
        hist_pivot.columns = [f'{val}_{month}' for val, month in hist_pivot.columns]

        target_series = {}
        for month in range(3, 11):
            if month in target_weather.index:
                target_series[f'temp_{month}'] = target_weather.loc[month, 'temp']
                target_series[f'precip_{month}'] = target_weather.loc[month, 'precip']

        common_cols = hist_pivot.columns.intersection(target_series.keys())
        if not common_cols.any():
            # This is a sub-fallback, if something is wrong with the data.
            logging.warning(
                f"Could not find common columns for analog search for {district_no}-{year}. Reverting to generic climatology.")
            wg.fit(district_past_weather)
            apply_anomalies = True
        else:
            aligned_target = pd.Series(target_series)[common_cols]
            distances = np.sqrt(np.sum((hist_pivot[common_cols].dropna() - aligned_target) ** 2, axis=1)).sort_values()
            analog_years = distances.head(analog_config['NUM_ANALOGS']).index.tolist()
            analog_weather_data = district_past_weather[district_past_weather['year'].isin(analog_years)]

            # Fit the "expert" weather generator on only the best analog years
            wg.fit(analog_weather_data)
    # ==========================================================================
    # === END: LOGIC BLOCK ===
    # ==========================================================================

    # --- Generation and Saving ---
    all_members_for_group = []
    start_date, end_date = f'{year}-03-01', f'{year}-11-30'

    for _, member_row in group_df.iterrows():
        member = member_row['seas5_member']
        monthly_anomalies_to_apply = {}

        # **FIXED**: Only prepare anomalies if the flag is set (i.e., we are using the fallback WG)
        if apply_anomalies:
            for month in range(3, 11):
                if month in [3, 4, 5]:  # Spring
                    monthly_anomalies_to_apply[f'temp_anomaly_{month}'] = member_row.get('spring_temp_anomaly_forecast',
                                                                                         0)
                    monthly_anomalies_to_apply[f'precip_anomaly_{month}'] = member_row.get(
                        'spring_precip_anomaly_forecast', 0)
                else:  # Summer & Autumn
                    monthly_anomalies_to_apply[f'temp_anomaly_{month}'] = member_row.get('summer_temp_anomaly_forecast',
                                                                                         0)
                    monthly_anomalies_to_apply[f'precip_anomaly_{month}'] = member_row.get(
                        'summer_precip_anomaly_forecast', 0)

        # The 'monthly_anomalies_to_apply' dictionary will be empty for the expert/analog case
        synth_weather = wg.generate(district_no, start_date, end_date, monthly_anomalies_to_apply)

        if not synth_weather.empty:
            synth_weather['district_no'] = district_no
            synth_weather['year'] = year
            synth_weather['member'] = member
            all_members_for_group.append(synth_weather)

    if all_members_for_group:
        result_df = pd.concat(all_members_for_group, ignore_index=True)
        output_path = Path(output_dir) / f'forecast_{district_no}_{year}.parquet'
        result_df.to_parquet(output_path, index=False)


# --- Main function (unchanged) ---
def main():
    parser = argparse.ArgumentParser(description="Generate forecast weather ensembles using analog year logic.")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Limit the run to a specific number of district-year groups for testing.")
    args = parser.parse_args()
    logging.info("--- Building Forecast Weather (with Analog Year Logic) ---")
    if FORECAST_PARTS_DIR.exists():
        logging.warning(f"Output directory '{FORECAST_PARTS_DIR}' exists. Deleting and regenerating.")
        shutil.rmtree(FORECAST_PARTS_DIR)
    FORECAST_PARTS_DIR.mkdir(parents=True)
    try:
        logging.info("Loading all historical weather files into memory...")
        weather_path = Path(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'])
        weather_files = list(weather_path.glob("*.csv"))
        if not weather_files: raise FileNotFoundError("No historical weather CSV files found.")
        full_hist_weather_df = pd.concat((pd.read_csv(f, parse_dates=['date'], dtype={'district_no': str}) for f in
                                          tqdm(weather_files, desc="Loading Historical Weather")), ignore_index=True)
        full_hist_weather_df['year'] = full_hist_weather_df['date'].dt.year
        full_hist_weather_df['month'] = full_hist_weather_df['date'].dt.month
        full_hist_weather_df['district_no'] = full_hist_weather_df['district_no'].str.zfill(5)
        if 'prec' in full_hist_weather_df.columns: full_hist_weather_df.rename(columns={'prec': 'precip'}, inplace=True)

        logging.info("Loading SEAS5 forecast member data...")
        df_seas5_all = pd.read_csv(CONFIG['FILE_PATHS']['SEAS5_MEMBER_FEATURES'], dtype={'district_no': str})
        df_seas5_all['district_no'] = df_seas5_all['district_no'].str.zfill(5)

        if args.limit:
            logging.warning(f"--- !!! TEST RUN !!! Limiting to {args.limit} district-year groups. ---")
            unique_groups = df_seas5_all[['district_no', 'year']].drop_duplicates().head(args.limit)
            df_seas5_all = pd.merge(df_seas5_all, unique_groups, on=['district_no', 'year'], how='inner')

        logging.info("Grouping forecast tasks by district and year...")
        grouped_tasks = df_seas5_all.groupby(['district_no', 'year'])
        logging.info(f"Generating forecast ensembles in parallel for {len(grouped_tasks)} groups...")

        Parallel(n_jobs=-1)(delayed(generate_forecasts_for_group)(
            group_name, group_df, full_hist_weather_df, CONFIG['ANALOG_YEAR_CONFIG'], FORECAST_PARTS_DIR
        ) for group_name, group_df in tqdm(grouped_tasks))

        logging.info(f"--- SUCCESS: Partitioned forecast weather files saved to '{FORECAST_PARTS_DIR}' ---")
    except Exception as e:
        logging.error(f"FATAL: An error occurred. Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()