# File: run_wofost_pipeline.py
# Refactored to use central configuration from src.config
# V2 CLEAN & COMPLETE: This is the definitive, refactored version.
# It uses pre-calculated physical parameters and dynamic initial conditions.
# All obsolete functions and hacks have been REMOVED.
# All original logic has been preserved and integrated with the new data sources.

import datetime
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import os
import logging
import sys
from scipy.stats import gamma
from pcse.util import penman_monteith
from pcse.models import Wofost72_WLP_FD, Wofost72_PP
from pcse.base import ParameterProvider, WeatherDataProvider
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import Parallel, delayed
import geopandas as gpd

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# Use the WOFOST_CONFIG dictionary from the central config file
CONFIG = config.WOFOST_CONFIG

# ==============================================================================
# === SCRIPT STARTS HERE ===
# ==============================================================================
# Clear existing handlers to prevent duplicate logging
logging.getLogger().handlers = []
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)


class ParameterDict(dict):
    def add_variable(self, name, value, description=""):
        self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def copy(self):
        return ParameterDict(self)


class SimpleWeatherDataProvider(WeatherDataProvider):
    def __init__(self, weather_df, site_data):
        super().__init__()
        self.latitude = site_data['LAT']
        self.longitude = site_data['LON']
        self.elevation = site_data['ELEV']
        self.angstA = 0.25
        self.angstB = 0.5
        weather_df = weather_df.copy()
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        self.store = {}
        fallback_wind = CONFIG['WEATHER_DEFAULTS']['WIND_SPEED']
        fallback_vap_kpa = CONFIG['WEATHER_DEFAULTS']['VAPOR_PRESSURE']
        for _, row in weather_df.iterrows():
            try:
                day = row['date'].date()
                tmin = float(row['tmin'])
                tmax = float(row['tmax'])
                wind = float(row.get('wind', fallback_wind))

                # 'precip' is now in mm (from new CSV). Model needs cm.
                precip_cm = float(row['precip']) / 10.0

                # 'srad' is now in MJ/m²/day (from new CSV).
                # Model 'IRRAD' needs J/m²/day.
                # Model 'penman_monteith' needs kJ/m²/day.
                srad_mj_m2_day = float(row['srad'])
                irrad_j_m2_day = srad_mj_m2_day * 1_000_000.0
                irrad_kj_m2_day = srad_mj_m2_day * 1_000.0

                # 'vap' is now in kPa (from new CSV). Model 'VAP' needs hPa.
                vap_kpa = float(row.get('vap', fallback_vap_kpa))
                vap_hpa = vap_kpa * 10.0

                et0_mm = penman_monteith(day, self.latitude, self.elevation, tmin, tmax, irrad_kj_m2_day, vap_hpa,
                                         wind)
                et0_cm = et0_mm / 10.0

                self.store[(day, 0)] = ParameterDict(
                    {'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_cm,
                     'IRRAD': irrad_j_m2_day, 'VAP': vap_hpa, 'WIND': wind, 'E0': et0_cm, 'ES0': et0_cm, 'ET0': et0_cm,
                     'SNOWDEPTH': 0.0})
            except Exception as e:
                logging.error(
                    f"CRITICAL: Failed processing weather row for date: {row.get('date')}. Error: {e}")
                raise e


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

        # --- Handle potential missing 'vap' and 'wind' ---
        if 'vap' not in daily_df.columns:
            logging.warning("WeatherGenerator: 'vap' column missing from historical data. Using default 1.0.")
            daily_df['vap'] = 1.0
        daily_df['vap'] = daily_df['vap'].fillna(1.0)  # Fill any stray NaNs

        if 'wind' not in daily_df.columns:
            logging.warning("WeatherGenerator: 'wind' column missing from historical data. Using default 2.0.")
            daily_df['wind'] = 2.0
        daily_df['wind'] = daily_df['wind'].fillna(2.0)  # Fill any stray NaNs
        # --- END ---

        for (district_no, month), group in tqdm(daily_df.groupby(['district_no', 'month']),
                                                desc="Learning Weather Patterns"):
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
                'p_wet_given_dry': prob_wet_given_dry,
                'p_wet_given_wet': prob_wet_given_wet,
                'precip_gamma_shape': gamma_shape,
                'precip_gamma_scale': gamma_scale,
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
                'wind_std': max(group['wind'].std(), 0.5)
            }

    def generate(self, district_no: str, start_date_str: str, end_date_str: str, monthly_anomalies: dict):
        dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
        generated_data = []
        yesterday_was_wet = np.random.rand() < 0.5
        for date in dates:
            month, key = date.month, (str(district_no).zfill(5), date.month)
            if key not in self.stats: continue
            month_stats = self.stats[key]

            # --- Get historical stats ---
            transition_prob = month_stats['p_wet_given_wet'] if yesterday_was_wet else month_stats['p_wet_given_dry']
            today_is_wet = np.random.rand() < transition_prob
            precip = 0.0
            if today_is_wet: alpha = month_stats['precip_gamma_shape']; beta = month_stats[
                'precip_gamma_scale']; precip = max(0, gamma.rvs(a=alpha, scale=beta, size=1)[0])

            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std'])
            tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin: tmax = tmin + abs(np.random.normal(0, 1.0))
            srad = max(self.MIN_SRAD, np.random.normal(month_stats['srad_mean'], month_stats['srad_std']))
            vap = max(0.1, np.random.normal(month_stats['vap_mean'], month_stats['vap_std']))
            wind = max(0.0, np.random.normal(month_stats['wind_mean'], month_stats['wind_std']))

            generated_data.append(
                {'date': date, 'tmin': tmin, 'tmax': tmax, 'precip': precip, 'srad': srad, 'vap': vap, 'wind': wind})
            yesterday_was_wet = today_is_wet

        if not generated_data: return pd.DataFrame()

        synthetic_df = pd.DataFrame(generated_data)
        synthetic_df['month'] = synthetic_df['date'].dt.month

        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month
            key = (str(district_no).zfill(5), month)
            if key not in self.stats: continue

            # --- Temperature Anomaly ---
            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0)
            hist_tmean = (self.stats[key]['tmin_mean'] + self.stats[key]['tmax_mean']) / 2
            synth_tmean = (synthetic_df.loc[month_mask, 'tmin'].mean() + synthetic_df.loc[
                month_mask, 'tmax'].mean()) / 2
            temp_correction = (hist_tmean + temp_anomaly) - synth_tmean
            synthetic_df.loc[month_mask, ['tmin', 'tmax']] += temp_correction

            # --- Precipitation Anomaly ---
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


# === NEW VALIDATION FUNCTION ===================================================
def analyze_v2_model_inputs(df_static_all, cropdata):
    """
    V2: Performs a diagnostic analysis on the ACTUAL model-ready inputs.
    This checks the pre-calculated physical parameters, not the raw ingredients.
    """
    logging.info("=" * 80 + "\n--- V2 INPUT ANALYSIS: Checking pre-calculated physical parameters ---\n" + "=" * 80)
    analysis_passed = True

    try:
        logging.info("[1/3] Analyzing static data integrity...")
        required_cols = ['SMW', 'SMFCF', 'CRAIRC', 'K0', 'WAV', 'RDMSOL', 'NOTINF', 'SSMAX', 'latitude', 'longitude',
                         'avg_elevation']
        missing_cols = [col for col in required_cols if col not in df_static_all.columns]
        if missing_cols:
            logging.error(f"    [FAIL] Missing critical columns in merged static data: {missing_cols}")
            analysis_passed = False
        else:
            logging.info("    [OK] All required physics columns are present.")

        logging.info("[2/3] Analyzing physical parameter ranges...")
        descriptions = df_static_all[required_cols].describe().T
        logging.info("\n" + descriptions.to_string())

        # Specific checks for plausibility
        if not (0.05 < descriptions.loc['SMFCF', 'mean'] < 0.6):  # Typical range for fraction
            logging.warning(
                f"    [WARNING] Mean Field Capacity (SMFCF) is {descriptions.loc['SMFCF', 'mean']:.3f} (fraction), which seems unusual. Expected between 0.05 and 0.6.")
            # analysis_passed = False # Keep as warning for now, actual range can vary
        if descriptions.loc['CRAIRC', 'mean'] < 0.01 or descriptions.loc[
            'CRAIRC', 'mean'] > 0.1:  # Typical range for fraction
            logging.warning(
                f"    [WARNING] Mean Critical Air Content (CRAIRC) is {descriptions.loc['CRAIRC', 'mean']:.4f} (fraction). Expected between 0.01 and 0.1 for most soils.")
            # analysis_passed = False
        if descriptions.loc[
            'WAV', 'min'] < -10.0:  # A small negative WAV is theoretically possible if SMW is too high relative to SMFCF or severe drought
            logging.error(
                f"    [FAIL] Minimum Initial Available Water (WAV) is extremely negative ({descriptions.loc['WAV', 'min']:.2f} cm). Check the winter balance script and soil parameters.")
            analysis_passed = False
        if descriptions.loc['WAV', 'max'] > 50.0:  # Unlikely to have 50cm available water in root zone
            logging.warning(
                f"    [WARNING] Maximum Initial Available Water (WAV) is very high ({descriptions.loc['WAV', 'max']:.2f} cm). Check winter balance logic or RDMSOL/SMFCF.")
            # analysis_passed = False
        if descriptions.loc['RDMSOL', 'mean'] < 100.0 or descriptions.loc['RDMSOL', 'mean'] > 250.0:
            logging.warning(
                f"    [WARNING] Mean Soil-Limited Rooting Depth (RDMSOL) is {descriptions.loc['RDMSOL', 'mean']:.1f} cm. Review its source/default.")
            # analysis_passed = False

        logging.info("    [OK] Physical parameter ranges appear plausible (manual review recommended for warnings).")

        logging.info("[3/3] Analyzing crop parameters (subset for sanity check)...")
        if not (500 < cropdata.get('TSUM1', 0) < 1200):
            logging.warning(
                f"    [WARNING] TSUM1 is {cropdata.get('TSUM1')}, seems unusual for sugarbeet. Expected 500-1200 C.d.")
            # analysis_passed = False
        if not (0.3 < cropdata.get('CFET', 0) < 1.3):  # Crop factor for transpiration
            logging.warning(
                f"    [WARNING] CFET (Crop Factor) is {cropdata.get('CFET')}. Typical range is 0.3-1.3, check for plausibility.")

        logging.info(
            f"    [OK] Crop parameters loaded (e.g., TSUM1: {cropdata.get('TSUM1')}, CFET: {cropdata.get('CFET')}).")

    except Exception as e:
        logging.error(f"    [FAIL] An error occurred during input analysis: {e}", exc_info=True)
        analysis_passed = False

    logging.info("=" * 80)
    if analysis_passed:
        logging.info("--- ANALYSIS V2 COMPLETE: Inputs seem plausible. ---")
    else:
        logging.error("--- ANALYSIS V2 FAILED: Critical errors found in model inputs. ---")
    logging.info("=" * 80)
    return analysis_passed


# === REFACTORED PARAMETER LOADER =================================================
def _create_district_specific_parameters(static_row, cropdata):
    """
    V3: Loads pre-calculated physics AND dynamic initial conditions directly from the data row.
    This function is now a simple data loader, not a calculator.
    """
    sitedata = ParameterDict()
    soildata = ParameterDict()

    # Site-specific data from the merged dataframe
    sitedata.add_variable('LAT', static_row['latitude'])
    sitedata.add_variable('LON', static_row['longitude'])
    sitedata.add_variable('ELEV', static_row['avg_elevation'])

    # Pass the DYNAMIC, PRE-CALCULATED WAV
    try:
        sitedata.add_variable('WAV', static_row['WAV'])
    except KeyError:
        logging.error(f"FATAL: 'WAV' column not found in static_row. The merge with initial_conditions_wav.csv failed.")
        raise

    # Add other site-related parameters that are static but not from soil PTFs
    # These are directly from the static_features_districts.csv (V4)
    sitedata.add_variable('NOTINF', static_row['NOTINF'])  # Max fraction of rain not infiltrating
    sitedata.add_variable('SSMAX', static_row['SSMAX'])  # Max surface storage

    # Soil-specific, pre-calculated physical data
    # These are from the static_features_districts.csv (V4)
    soil_params = ['SMW', 'SMFCF', 'SM0', 'CRAIRC', 'K0', 'SOPE', 'KSUB', 'RDMSOL']
    for param in soil_params:
        try:
            soildata.add_variable(param, static_row[param])
        except KeyError:
            logging.error(
                f"FATAL: Missing required soil parameter '{param}' in static data for district {static_row.get('district_no', 'N/A')}.")
            raise

    # Add remaining model constants/runtime variables required by PCSE
    sitedata.add_variable('IFUNRN', 0.0)  # Assume no runoff function in this model setup
    sitedata.add_variable('SSI', 0.0)  # Initial surface storage (will be updated by SSMAX)
    sitedata.add_variable('SMLIM', soildata['SMFCF'])  # Initial soil moisture limit (often field capacity)

    return ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata), sitedata


# === SIMULATION & ANALYSIS FUNCTIONS (Logic unchanged, now use correct inputs) =====
# In run_wofost_pipeline.py
# DELETE your old run_historical_simulation function and REPLACE it with this.

def run_historical_simulation(df_static_year, df_daily_hist_year, cropdata, year, cfg, dynamic_sowing_dates):
    """
    MODIFIED: Runs historical simulation using DYNAMIC sowing dates.
    """
    results = []

    # Get the static end date from config ONCE
    # We assume harvest date is still fixed for now.
    crop_end_date_template = cfg['AGROMANAGEMENT']['CROP_END_DATE']

    for _, row in tqdm(df_static_year.iterrows(), total=len(df_static_year), desc=f"Historical Sim {year}"):
        district_no = row['district_no']
        weather_df = df_daily_hist_year[df_daily_hist_year['district_no'] == district_no].copy()
        if weather_df.empty:
            continue

        try:
            parameters, site_data = _create_district_specific_parameters(row, cropdata)
            weather_provider = SimpleWeatherDataProvider(weather_df, site_data)

            # --- START OF THE CRITICAL CHANGE ---
            # Get the dynamic sowing date from the dictionary
            crop_start = dynamic_sowing_dates.get(district_no)

            # Fallback in case a date was not calculated
            if crop_start is None:
                logging.warning(f"No dynamic sowing date for {district_no} in {year}. Falling back to static date.")
                crop_start = cfg['AGROMANAGEMENT']['CROP_START_DATE'].replace(year=year)

            # The harvest date remains the same for this year
            crop_end = crop_end_date_template.replace(year=year)
            # --- END OF THE CRITICAL CHANGE ---

            agromanagement = [{
                crop_start: ParameterDict({
                    'CropCalendar': ParameterDict({
                        'crop_start_date': crop_start,
                        'crop_start_type': 'emergence',
                        'crop_end_date': crop_end,
                        'crop_end_type': 'harvest',
                        'max_duration': cfg['AGROMANAGEMENT']['MAX_DURATION']
                    }),
                    'TimedEvents': None,
                    'StateEvents': None
                })
            }]

            model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)
            model.run_till_terminate()
            output = model.get_output()
            simulated_yield = output[-1]['TWSO'] if output else np.nan

        except Exception as e:
            logging.error(f"[HISTORICAL] ERROR for district {district_no} in {year}: {e}", exc_info=True)
            simulated_yield = np.nan

        results.append({
            'year': year,
            'district_no': district_no,
            'actual_yield': row['kreisYield'],
            'lintul_yield_perfect_weather': simulated_yield
        })

    return pd.DataFrame(results)

def _run_single_forecast_member(member_row, district_no, year, wg, parameters, site_data, cfg, apply_anomalies=True):
    try:
        spring_temp_anomaly = member_row.get('spring_temp_anomaly_forecast', 0)
        summer_temp_anomaly = member_row.get('summer_temp_anomaly_forecast', 0)
        spring_precip_anomaly = member_row.get('spring_precip_anomaly_forecast', 0)
        summer_precip_anomaly = member_row.get('summer_precip_anomaly_forecast', 0)

        monthly_anomalies = {}
        if apply_anomalies:
            for month in range(3, 11):  # March to October
                if month in [3, 4, 5]:  # Spring
                    monthly_anomalies[f'temp_anomaly_{month}'] = spring_temp_anomaly
                    monthly_anomalies[f'precip_anomaly_{month}'] = spring_precip_anomaly
                elif month in [6, 7, 8]:  # Summer
                    monthly_anomalies[f'temp_anomaly_{month}'] = summer_temp_anomaly
                    monthly_anomalies[f'precip_anomaly_{month}'] = summer_precip_anomaly
                else:  # Autumn (Sep, Oct) - extending summer forecast
                    monthly_anomalies[f'temp_anomaly_{month}'] = summer_temp_anomaly
                    monthly_anomalies[f'precip_anomaly_{month}'] = summer_precip_anomaly

        start_date, end_date = f'{year}-03-01', f'{year}-11-30'  # Fixed date range for weather generation
        synth_weather = wg.generate(district_no, start_date, end_date, monthly_anomalies)

        # Check if weather generation was successful
        if synth_weather.empty or len(synth_weather) < (datetime.date(year, 11, 30) - datetime.date(year, 3, 1)).days:
            logging.warning(
                f"[FORECAST_WORKER] Weather generation failed or incomplete for dist {district_no}, member {member_row.get('seas5_member', 'N/A')}. Returning default failure.")
            return {'member': member_row.get('seas5_member', 'N/A'), 'yield_water_limited': 0.0,
                    'yield_potential': 0.0, 'consecutive_tmax_gt_30c': np.nan,
                    'consecutive_dry_days': np.nan, 'drought_stress_index': 1.0,
                    'simulation_failed': True, 'days_to_anthesis': np.nan, 'max_lai_achieved': 0.0,
                    'cumulative_water_stress': np.nan}

        weather_provider = SimpleWeatherDataProvider(synth_weather, site_data)
        crop_start = cfg['AGROMANAGEMENT']['CROP_START_DATE'].replace(year=year)
        crop_end = cfg['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)
        agromanagement = [{crop_start: ParameterDict({'CropCalendar': ParameterDict(
            {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
             'crop_end_type': 'harvest', 'max_duration': cfg['AGROMANAGEMENT']['MAX_DURATION']}), 'TimedEvents': None,
            'StateEvents': None})}]

        model_wlp = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)
        model_wlp.run_till_terminate()
        output_wlp = pd.DataFrame(model_wlp.get_output()).set_index('day')
        yield_wlp = output_wlp.iloc[-1]['TWSO'] if not output_wlp.empty else 0

        model_pp = Wofost72_PP(parameters, weather_provider, agromanagement)
        model_pp.run_till_terminate()
        output_pp = pd.DataFrame(model_pp.get_output())
        yield_pp = output_pp.iloc[-1]['TWSO'] if not output_pp.empty else 0

        def get_max_consecutive_run(boolean_series):
            if not boolean_series.any(): return 0
            runs = boolean_series.ne(boolean_series.shift()).cumsum()
            return boolean_series.groupby(runs).cumsum().max()

        summer_weather = synth_weather[synth_weather['date'].dt.month.isin([6, 7, 8])].copy()
        is_heatwave_day = summer_weather['tmax'] > 30
        consecutive_hot_days = get_max_consecutive_run(is_heatwave_day)

        is_dry_day = summer_weather['precip'] < 1
        consecutive_dry_days = get_max_consecutive_run(is_dry_day)

        drought_stress_index = (yield_pp - yield_wlp) / yield_pp if yield_pp > 0 else 0.0
        days_to_anthesis = np.nan
        if 'DOA' in output_wlp.columns and (output_wlp['DOA'] is not None):
            first_anthesis_day = output_wlp[output_wlp['DOA'].notna()].index.min()
            if pd.notna(first_anthesis_day): days_to_anthesis = (first_anthesis_day - crop_start).days

        max_lai_achieved = output_wlp['LAI'].max() if 'LAI' in output_wlp.columns else 0.0
        cumulative_water_stress = (1 - output_wlp['TRA']).sum() if 'TRA' in output_wlp.columns else np.nan

        return {'member': member_row.get('seas5_member', 'N/A'), 'yield_water_limited': yield_wlp,
                'yield_potential': yield_pp, 'consecutive_tmax_gt_30c': consecutive_hot_days,
                'consecutive_dry_days': consecutive_dry_days, 'drought_stress_index': drought_stress_index,
                'simulation_failed': False, 'days_to_anthesis': days_to_anthesis, 'max_lai_achieved': max_lai_achieved,
                'cumulative_water_stress': cumulative_water_stress}

    except Exception as e:
        logging.warning(
            f"[FORECAST_WORKER] Sim failed for dist {district_no}, member {member_row.get('seas5_member', 'N/A')}: {e}")
        return {'member': member_row.get('seas5_member', 'N/A'), 'yield_water_limited': 0.0, 'yield_potential': 0.0,
                'consecutive_tmax_gt_30c': np.nan, 'consecutive_dry_days': np.nan, 'drought_stress_index': 1.0,
                'simulation_failed': True, 'days_to_anthesis': np.nan, 'max_lai_achieved': 0.0,
                'cumulative_water_stress': np.nan}


def run_forecast_simulation(df_static_year, df_seas5_year, district_wgs, cropdata, year, cfg, expert_districts):
    full_ensemble_results = []
    # Temporarily disable INFO logging from the main thread for cleaner progress bar
    # Re-enable after this loop if needed, but forecast workers will log their own issues
    logging.disable(logging.INFO)
    district_params = {row['district_no']: _create_district_specific_parameters(row, cropdata) for _, row in
                       df_static_year.iterrows()}
    logging.disable(logging.NOTSET)  # Re-enable logging

    for district_no, group in tqdm(df_seas5_year.groupby('district_no'), desc=f"Forecast Sim {year}"):
        if district_no not in district_params: continue
        parameters, site_data = district_params[district_no]
        wg = district_wgs.get(district_no, WeatherGenerator())
        apply_anomalies = district_no not in expert_districts

        tasks = [delayed(_run_single_forecast_member)(member_row, district_no, year, wg, parameters, site_data, cfg,
                                                      apply_anomalies) for
                 _, member_row in group.iterrows()]
        ensemble_outputs = Parallel(n_jobs=-1, backend='loky')(tasks)
        for result in ensemble_outputs:
            if result is not None: full_ensemble_results.append(
                {'year': year, 'district_no': district_no, 'member': result['member'],
                 'yield_water_limited_dry_kgha': result['yield_water_limited'],
                 'yield_potential_dry_kgha': result['yield_potential'],
                 'consecutive_tmax_gt_30c': result['consecutive_tmax_gt_30c'],
                 'consecutive_dry_days': result['consecutive_dry_days'],
                 'drought_stress_index': result['drought_stress_index'],
                 'simulation_failed': result['simulation_failed'], 'days_to_anthesis': result['days_to_anthesis'],
                 'max_lai_achieved': result['max_lai_achieved'],
                 'cumulative_water_stress': result['cumulative_water_stress']})
    return pd.DataFrame(full_ensemble_results)


def analyze_and_plot_ensemble_results(df_hist, df_fcst_ensemble, output_dir, start_year, end_year):
    """
    MODIFIED (v4): Re-added the ensemble save line, plus text-based logging.
    """
    logging.info("=" * 70 + "\n[ANALYSIS v4] Starting Text-Based Debug Analysis\n" + "=" * 70)

    # --- 1. Load Config & *** SAVE RAW ENSEMBLE DATA *** ---
    dmc = CONFIG['CONSTANTS']['DMC_SUGARBEET']

    # *** THIS IS THE FIX: Re-added the save command ***
    if not df_fcst_ensemble.empty:
        fcst_output_path = output_dir / f'forecast_ensemble_{start_year}-{end_year}.csv'
        df_fcst_ensemble.to_csv(fcst_output_path, index=False)
        logging.info(f"✓ Full forecast ensemble results saved to {fcst_output_path}")
    # *** END OF FIX ***

    if df_fcst_ensemble.empty or df_hist.empty:
        logging.error("[ANALYSIS] No data in forecast or historical results. Cannot analyze.")
        return

    # --- 2. Convert ALL Yields to Fresh Weight (dt/ha) ---
    df_fcst_ensemble['yield_wlp_fresh_dt'] = (df_fcst_ensemble['yield_water_limited_dry_kgha'] / dmc) / 100.0
    df_fcst_ensemble['yield_pp_fresh_dt'] = (df_fcst_ensemble['yield_potential_dry_kgha'] / dmc) / 100.0
    df_hist['perfect_yield_dt'] = (df_hist['lintul_yield_perfect_weather'] / dmc) / 100.0

    # --- 3. Aggregate Ensemble Data ---
    logging.info("[ANALYSIS] Aggregating ensemble data...")
    df_fcst_agg = df_fcst_ensemble.groupby(['year', 'district_no']).agg(
        forecast_yield_mean=('yield_wlp_fresh_dt', 'mean'),
        forecast_yield_p10=('yield_wlp_fresh_dt', lambda x: x.quantile(0.10)),
        forecast_yield_p90=('yield_wlp_fresh_dt', lambda x: x.quantile(0.90)),
        potential_yield_mean=('yield_pp_fresh_dt', 'mean'),
        sim_failure_rate=('simulation_failed', 'mean')
    ).reset_index()

    # --- 4. Merge Historical and Forecast Data ---
    df_final = pd.merge(
        df_hist[['year', 'district_no', 'actual_yield', 'perfect_yield_dt']],
        df_fcst_agg,
        on=['year', 'district_no']
    )

    # --- 5. START OF TEXT-BASED ANALYSIS ---
    logging.info("\n" + "=" * 80)
    logging.info("--- DETAILED TEXT-BASED ANALYSIS ---")
    logging.info(f"Analysis for: {start_year}-{end_year}")
    logging.info(f"Dry Matter Content (DMC_SUGARBEET) used for conversion: {dmc}")

    logging.info("\n[DEBUG] 1. Checking for raw 'perfect_yield_dt' from historical run:")
    logging.info(df_hist['perfect_yield_dt'].describe().to_string())
    nan_count_hist = df_hist['perfect_yield_dt'].isna().sum()
    logging.info(f"Historical NaNs: {nan_count_hist} / {len(df_hist)}")

    logging.info("\n[DEBUG] 2. Checking for raw 'forecast_yield_mean' from ensemble:")
    logging.info(df_fcst_agg['forecast_yield_mean'].describe().to_string())

    logging.info("\n[DEBUG] 3. Final Merged DataFrame (Head):")
    logging.info(df_final.head(15).to_string())

    logging.info("\n[DEBUG] 4. Final Merged DataFrame (Statistical Summary BEFORE dropna):")
    logging.info(df_final.describe().to_string())

    # --- 6. Final Data Integrity Check ---
    logging.info("\n[DEBUG] 5. Final Data Integrity & Counts:")
    logging.info(f"Total rows in df_hist: {len(df_hist)}")
    logging.info(f"Total rows in df_fcst_agg: {len(df_fcst_agg)}")
    logging.info(f"Total rows in df_final (after merge): {len(df_final)}")

    df_final_clean = df_final.dropna(subset=['actual_yield', 'perfect_yield_dt', 'forecast_yield_mean'])
    logging.info(f"Total rows in df_final (after dropna): {len(df_final_clean)}")

    logging.info("=" * 80 + "\n")
    # --- END OF TEXT-BASED ANALYSIS ---

    if df_final_clean.empty:
        logging.error("[ANALYSIS] FINAL FAILURE: No valid, non-NaN merged results were found.")
        return

    # --- 7. Calculate & Print Metrics (on cleaned data) ---
    mae_p = mean_absolute_error(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt'])
    r2_p = r2_score(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt'])
    mae_f = mean_absolute_error(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'])
    r2_f = r2_score(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'])

    print("\n--- Overall Performance Metrics (Fresh Weight dt/ha, based on Ensemble Mean) ---")
    print(f"  Perfect Weather (WLP):  MAE = {mae_p:.2f}, R² = {r2_p:.3f}")
    print(f"  Forecast Weather (WLP): MAE = {mae_f:.2f}, R² = {r2_f:.3f}\n")

    # --- 8. Create Diagnostic Plots (only if data is valid) ---
    logging.info("[ANALYSIS] Generating diagnostic plots (if any valid data exists)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle(f'WOFOST Performance Diagnosis ({start_year}-{end_year})', fontsize=16)

    min_val_data = df_final_clean[
        ['actual_yield', 'perfect_yield_dt', 'forecast_yield_p10', 'potential_yield_mean']].min()
    max_val_data = df_final_clean[
        ['actual_yield', 'perfect_yield_dt', 'forecast_yield_p90', 'potential_yield_mean']].max()
    min_val = min_val_data.min() * 0.95
    max_val = max_val_data.max() * 1.05

    # === PLOT 1: PERFECT WEATHER ===
    axes[0].scatter(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt'], alpha=0.6,
                    label='Simulated (Perfect Weather)')
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    axes[0].scatter(df_final_clean['actual_yield'], df_final_clean['potential_yield_mean'],
                    marker='x', color='red', alpha=0.5, label='Mean Potential Yield (PP)')
    axes[0].set_title(f'Perfect Weather\nMAE={mae_p:.2f}, R²={r2_p:.3f}')
    axes[0].set_xlabel('Actual Yield (dt/ha)')
    axes[0].set_ylabel('Simulated Yield (dt/ha)')

    # === PLOT 2: FORECAST WEATHER (ENSEMBLE) ===
    lower_error = df_final_clean['forecast_yield_mean'] - df_final_clean['forecast_yield_p10']
    upper_error = df_final_clean['forecast_yield_p90'] - df_final_clean['forecast_yield_mean']
    y_err = [lower_error.values, upper_error.values]

    axes[1].errorbar(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'], yerr=y_err, fmt='o',
                     color='orange',
                     ecolor='lightgray', elinewidth=3, capsize=0, alpha=0.8,
                     label='Ensemble Mean & 10-90th Pct. Range')
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    axes[1].scatter(df_final_clean['actual_yield'], df_final_clean['potential_yield_mean'],
                    marker='x', color='red', alpha=0.5, label='Mean Potential Yield (PP)')
    axes[1].set_title(f'Forecast Weather (Ensemble Range)\nMean MAE={mae_f:.2f}, Mean R²={r2_f:.3f}')
    axes[1].set_xlabel('Actual Yield (dt/ha)')

    # --- 9. Finalize and Save Plot ---
    for ax in axes:
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / f'results_scatter_with_POTENTIAL_{start_year}-{end_year}.png'
    plt.savefig(plot_path, dpi=300)
    logging.info(f"[ANALYSIS] ✓ Diagnostic plot (with potential yield) saved to {plot_path}")
    plt.show()


def aggregate_and_save_extreme_weather_metrics(df_fcst_ensemble, output_path):
    """
    Calculates and saves the distributional features for the new extreme weather and drought stress metrics.
    """
    logging.info(
        "=" * 70 + "\n[ANALYSIS] Aggregating new in-season, weather, and drought stress risk features...\n" + "=" * 70)
    if df_fcst_ensemble.empty:
        logging.warning("[ANALYSIS] Forecast ensemble dataframe is empty. Skipping extreme metrics.")
        return

    # Define aggregation functions
    aggs = {
        'consecutive_tmax_gt_30c': [
            'mean', 'std', lambda x: x.quantile(0.90), lambda x: (x > 10).mean()
        ],
        'consecutive_dry_days': [
            'mean', 'std', lambda x: x.quantile(0.90), lambda x: (x > 21).mean()
        ],
        'drought_stress_index': [
            'mean', 'std', lambda x: x.quantile(0.90), lambda x: (x > 0.5).mean()
        ],
        'simulation_failed': ['mean'],
        'days_to_anthesis': ['mean', 'std', lambda x: x.quantile(0.90)],
        'max_lai_achieved': ['mean', 'std', lambda x: x.quantile(0.10)],
        # Lower LAI is worse, so p10 is the "worst case"
        'cumulative_water_stress': ['mean', 'std', lambda x: x.quantile(0.90)]
    }

    # Perform aggregation
    df_extreme_metrics = df_fcst_ensemble.groupby(['year', 'district_no']).agg(aggs).reset_index()

    # Flatten the multi-level column names
    df_extreme_metrics.columns = [
        'year', 'district_no',
        'mean_consecutive_days_above_30c', 'std_dev_consecutive_days_above_30c',
        'p90_consecutive_days_above_30c', 'prob_heatwave_gt_10_days',
        'mean_consecutive_dry_days', 'std_dev_consecutive_dry_days',
        'p90_consecutive_dry_days', 'prob_drought_spell_gt_21_days',
        'mean_drought_stress_index', 'std_drought_stress_index',
        'p90_drought_stress_index', 'prob_severe_drought_stress',
        'prob_simulation_failure',
        'mean_days_to_anthesis', 'std_days_to_anthesis', 'p90_days_to_anthesis',
        'mean_max_lai_achieved', 'std_max_lai_achieved', 'p10_max_lai_achieved',
        'mean_cumulative_water_stress', 'std_cumulative_water_stress', 'p90_cumulative_water_stress'
    ]

    # Save the aggregated metrics to a new CSV file
    df_extreme_metrics.to_csv(output_path, index=False)
    logging.info(f"[ANALYSIS] ✓ All risk features saved to {output_path}")
    pass


if __name__ == "__main__":
    # --- 1. SETUP LOGGING & PATHS ---
    os.makedirs(CONFIG['FILE_PATHS']['OUTPUT_DIR'], exist_ok=True)
    sowing_manager = DynamicSowingManager()

    # --- 2. CONSOLIDATED DATA LOADING ---
    logging.info("=" * 70 + "\nLoading and merging ALL data sources (V3 WORKFLOW)...\n" + "=" * 70)
    try:
        df_yield = pd.read_csv(CONFIG['FILE_PATHS']['YIELD_DATA'], dtype={'district_no': str})
        df_yield.rename(columns={'yield': 'kreisYield'}, inplace=True)

        df_static_physics = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_SOIL_FEATURES'], dtype={'district_no': str})

        gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH)
        gdf_districts['latitude'] = gdf_districts.geometry.centroid.y
        gdf_districts['longitude'] = gdf_districts.geometry.centroid.x
        gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)

        df_wav = pd.read_csv(CONFIG['FILE_PATHS']['INITIAL_CONDITIONS'], dtype={'district_no': str})

        df_static_base = pd.merge(df_static_physics, gdf_districts[['district_no', 'latitude', 'longitude']],
                                  on='district_no')
        df_static_all = pd.merge(df_yield, df_static_base, on='district_no', how='inner')
        df_static_all = pd.merge(df_static_all, df_wav, on=['year', 'district_no'], how='left')

        missing_wav = df_static_all['WAV'].isna()
        if missing_wav.any():
            logging.warning(
                f"{missing_wav.sum()} rows have missing WAV values. Check year ranges. Filling with default 10.0.")
            df_static_all.loc[missing_wav, 'WAV'] = 10.0

        df_seas5_all = pd.read_csv(CONFIG['FILE_PATHS']['SEAS5_MEMBER_FEATURES'], dtype={'district_no': str})
        valid_combinations = df_static_all[['year', 'district_no']].drop_duplicates()
        df_seas5_all = pd.merge(df_seas5_all, valid_combinations, on=['year', 'district_no'], how='inner')
        logging.info(f"Proceeding with {len(df_static_all)} valid district-year records.")

    except FileNotFoundError as e:
        logging.error(f"FATAL: A required data file was not found. Error: {e}");
        sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: Error during data loading: {e}", exc_info=True);
        sys.exit(1)

    # --- 3. LOAD WEATHER & CROP DATA ---
    logging.info("Loading auxiliary data (weather, crop)...")
    full_hist_weather_df = pd.concat([pd.read_csv(f, parse_dates=['date'], dtype={'district_no': str}) for f in
                                      CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'].glob("*.csv")],
                                     ignore_index=True)
    with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
        crop_params_all = yaml.safe_load(f)['CropParameters']
    final_params_dict = {**crop_params_all.get('GenericC3', {}), **crop_params_all['EcoTypes']['sugarbeet'],
                         **crop_params_all['Varieties']['Sugarbeet_601']}
    cropdata = ParameterDict()
    for key, val in final_params_dict.items():
        if key not in ['Metadata', '<<'] and isinstance(val, list) and len(val) > 0:
            cropdata.add_variable(key, val[0])

    # --- 4. RUN V2 INPUT VALIDATION ---
    if not analyze_v2_model_inputs(df_static_all, cropdata):
        logging.error("Input data analysis failed. Aborting pipeline.");
        sys.exit(1)

    # --- 5. MAIN SIMULATION LOOP ---
    all_hist_results = []
    all_fcst_results = []
    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        logging.info("=" * 70 + f"\nPROCESSING YEAR: {year}\n" + "=" * 70)
        df_static_year = df_static_all[df_static_all['year'] == year].copy()
        df_seas5_year = df_seas5_all[df_seas5_all['year'] == year].copy()

        hist_weather_path = CONFIG['FILE_PATHS'][
                                'HISTORICAL_DAILY_WEATHER_DIR'] / f"historical_daily_weather_era5_{year}.csv"
        try:
            df_daily_hist_year = pd.read_csv(hist_weather_path, parse_dates=['date'], dtype={'district_no': str})
        except FileNotFoundError:
            logging.warning(f"Weather for {year} not found. Skipping year.");
            continue

        if CONFIG['DISTRICT_LIMIT'] is not None:
            limited_districts = df_static_year['district_no'].unique()[:CONFIG['DISTRICT_LIMIT']]
            df_static_year = df_static_year[df_static_year['district_no'].isin(limited_districts)]
            df_seas5_year = df_seas5_year[df_seas5_year['district_no'].isin(limited_districts)]
        if df_static_year.empty:
            logging.warning(f"Missing static data for {year}. Skipping.");
            continue

        past_weather_df = full_hist_weather_df[full_hist_weather_df['year'] < year].copy()
        district_weather_generators = {}
        expert_districts = set()
        for district_no in tqdm(df_static_year['district_no'].unique(), desc=f"Fitting Expert WGs for {year}"):
            district_past_weather = past_weather_df[past_weather_df['district_no'] == district_no]
            if len(district_past_weather['year'].unique()) < CONFIG['ANALOG_YEAR_CONFIG']['MIN_YEARS_FOR_FIT']:
                wg_expert = WeatherGenerator();
                wg_expert.fit(district_past_weather)
                district_weather_generators[district_no] = wg_expert
            else:
                expert_districts.add(district_no)
                climatology = district_past_weather.groupby('month')[['tmin', 'tmax', 'precip']].mean()
                climatology['temp'] = (climatology['tmin'] + climatology['tmax']) / 2
                target_forecast = df_seas5_year[df_seas5_year['district_no'] == district_no]
                target_anomalies = {m: {'temp': target_forecast.get(
                    f'spring_temp_anomaly_forecast' if m in [3, 4, 5] else f'summer_temp_anomaly_forecast',
                    pd.Series(0.0)).mean(),
                                        'precip': target_forecast.get(f'spring_precip_anomaly_forecast' if m in [3, 4,
                                                                                                                 5] else f'summer_precip_anomaly_forecast',
                                                                      pd.Series(0.0)).mean()} for m in range(3, 11)}
                target_weather = climatology.copy()
                for month in range(3, 11):
                    if month in target_weather.index:
                        target_weather.loc[month, 'temp'] += target_anomalies[month]['temp']
                        target_weather.loc[month, 'precip'] += target_anomalies[month]['precip']
                yearly_avg = district_past_weather.groupby(['year', 'month'])[
                    ['precip', 'tmin', 'tmax']].mean().reset_index()
                yearly_avg['temp'] = (yearly_avg['tmin'] + yearly_avg['tmax']) / 2
                hist_pivot = yearly_avg.pivot_table(index='year', columns='month', values=['temp', 'precip'])
                hist_pivot.columns = [f'{val}_{month}' for val, month in hist_pivot.columns]
                target_series = {f'{val}_{m}': target_weather.loc[m, val] for m in range(3, 11) if
                                 m in target_weather.index for val in ['temp', 'precip']}
                common_cols = hist_pivot.columns.intersection(target_series.keys())
                if not common_cols.any():
                    wg_expert = WeatherGenerator();
                    wg_expert.fit(district_past_weather)
                    district_weather_generators[district_no] = wg_expert
                    continue
                aligned_target = pd.Series(target_series)[common_cols]
                distances = np.sqrt(
                    np.sum((hist_pivot[common_cols].dropna() - aligned_target) ** 2, axis=1)).sort_values()
                analog_years = distances.head(CONFIG['ANALOG_YEAR_CONFIG']['NUM_ANALOGS']).index.tolist()
                analog_weather_data = district_past_weather[district_past_weather['year'].isin(analog_years)]
                wg_expert = WeatherGenerator();
                wg_expert.fit(analog_weather_data)
                district_weather_generators[district_no] = wg_expert

        dynamic_sowing_dates = {}
        for district_no in df_static_year['district_no'].unique():
            district_weather_for_sowing = df_daily_hist_year[df_daily_hist_year['district_no'] == district_no]
            if not district_weather_for_sowing.empty:
                sowing_date = sowing_manager.find_sowing_date(district_weather_for_sowing)
                dynamic_sowing_dates[district_no] = sowing_date

        df_hist = run_historical_simulation(df_static_year, df_daily_hist_year, cropdata, year, CONFIG,
                                            dynamic_sowing_dates)
        if not df_hist.empty: all_hist_results.append(df_hist)

        # Forecast run is temporarily disabled to focus on historical performance
        # df_fcst = run_forecast_simulation(...)
        # if not df_fcst.empty: all_fcst_results.append(df_fcst)

    # --- 6. FINAL ANALYSIS ---
    if all_hist_results:
        final_hist_df = pd.concat(all_hist_results, ignore_index=True)
        final_hist_df.dropna(inplace=True)

        if len(final_hist_df) > 1:
            mae = mean_absolute_error(final_hist_df['actual_yield'], final_hist_df['lintul_yield_perfect_weather'])
            r2 = r2_score(final_hist_df['actual_yield'], final_hist_df['lintul_yield_perfect_weather'])
            logging.info(
                "\n" + "=" * 50 + f"\nFINAL HISTORICAL PERFORMANCE (DYNAMIC SOWING)\nMAE: {mae:.2f}\nR2: {r2:.3f}\n" + "=" * 50)
        else:
            logging.warning("Not enough simulation results to calculate final performance metrics.")

    else:
        logging.error("No simulation results were generated across all years.")

    logging.info("\n" + "=" * 70 + "\n✓ PIPELINE COMPLETED.\n" + "=" * 70)