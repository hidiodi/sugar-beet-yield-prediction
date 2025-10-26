# File: run_lintul_one_year_test_pipeline.py
# Description: A consolidated script to run the full one-year test pipeline.
#
# REFACTORED & PARALLELIZED VERSION v12 (Production Ready): Implements monthly
# forecast anomalies, dynamic sowing dates, and dynamic initial soil moisture
# for both forecast and historical simulations. Includes final bug fixes and
# quality improvements for robustness.

import datetime
import yaml
import pandas as pd
import numpy as np
import os
import logging
import sys

import geopandas as gpd

from pcse.util import penman_monteith
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider, WeatherDataProvider

from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import Parallel, delayed

# ==============================================================================
# === G L O B A L   C O N F I G U R A T I O N ===
# ==============================================================================
CONFIG = {
    # --- Core Simulation Settings ---
    'TEST_YEAR': 2018,  # What: The single year to run the forecast/hindcast for. Why: Defines the target period. From: User-defined.
    'DISTRICT_LIMIT': 2,    # What: Limits the run to the first N districts. Why: For fast testing and debugging. From: User-defined.

    # --- File Input/Output Locations ---
    'FILE_PATHS': {
        # What: Path to historical daily weather data. Why: Drives the historical simulation and trains the weather generator. From: `03_process_agera5_data.py`.
        'HISTORICAL_DAILY_WEATHER': 'data/02_intermediate/historical_daily_weather_era5_2018_TEST.csv',
        # What: Path to static (non-weather) features per district. Why: Provides soil data, elevation, etc. From: GEE data processing scripts.
        'STATIC_FEATURES': 'data/05_model_input/stage1_preseason_features.csv',
        # What: Path to pre-processed ECMWF forecast anomalies. Why: Drives the forecast weather generator. From: `build_forecast_features.py`.
        'SEAS5_MEMBER_FEATURES': 'data/02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv',
        # What: Official WOFOST crop parameter file. Why: Defines the sugar beet's biological properties. From: PCSE/WOFOST standard data.
        'CROP_YAML': 'data/01_raw/sugarbeet.yaml',
        # What: Directory to save all outputs (CSVs, plots). Why: Organizes simulation results. From: User-defined.
        'OUTPUT_DIR': 'data/06_model_output/one_year_test',
        # What: Geographic boundaries of each district. Why: Used to calculate centroids for weather data lookup. From: External GIS data source.
        'DISTRICTS_GEOJSON': 'data/01_raw/districts_official.geojson',
    },

    # --- Weather Generator Parameters ---
    'WEATHER_GENERATOR': {
        # What: Rainfall amount (mm) to define a day as "wet". Why: Used by the Markov chain for wet/dry day transitions. From: Justified agronomic assumption.
        'PRECIP_THRESHOLD_MM': 0.3,
        # What: The lowest possible solar radiation value (J/m2/day). Why: Prevents the stochastic generator from creating negative values. From: Justified physical assumption.
        'MIN_SRAD': 100.0,
    },

    # --- Crop Management Rules ---
    'AGROMANAGEMENT': {
        # What: Assumed harvest date for all simulations. Why: Provides a fixed end to the growing season. From: Agronomic assumption (can be improved).
        'CROP_END_DATE': datetime.date(2018, 10, 20),
        # What: A safety limit on the simulation length in days. Why: Prevents model run-on errors. From: PCSE standard practice.
        'MAX_DURATION': 250,
    },

    # --- Core Biophysical Constants ---
    'CONSTANTS': {
        # What: Fraction of fresh beet weight that is dry matter. Why: Converts simulated dry biomass to a comparable fresh yield. From: Standard literature value.
        'DMC_SUGARBEET': 0.25,
        # What: The rooting depth of a newly emerged seedling (cm). Why: Required to initialize the soil water balance. From: WOFOST default value.
        'INITIAL_ROOTING_DEPTH_CM': 10.0,
        # What: The density of mineral soil particles (g/cm3). Why: Used to calculate soil porosity from bulk density. From: Standard soil science value.
        'SOIL_PARTICLE_DENSITY': 2.65,
    },

    # --- Data Mapping Configuration ---
    'SOIL_COLUMN_MAPPING': {
        # What: Maps PTF variable names to columns in the STATIC_FEATURES CSV. Why: Makes the code robust if input column names change. From: User-defined based on CSV format.
        'sand': 'avg_sand_0_100cm',
        'clay': 'avg_clay_0_100cm',
        'som': 'avg_som_0_100cm',
        'bdod': 'avg_bdod_0_100cm',
    },

    # --- Model Assumptions & Fallbacks ---
    'SOIL_DEFAULTS_AND_CONSTANTS': {
        # What: Maximum rooting depth (cm) if not provided by data. Why: A critical parameter determining water access. From: Agronomic assumption (major source of uncertainty).
        'RDMSOL': 150.0,
        # What: Default percolation rate of subsoil (cm/day). Why: A simplified parameter for water movement. From: WOFOST default/agronomic assumption.
        'KSUB': 10.0,
        # What: Default percolation rate of the root zone (cm/day). Why: A simplified parameter for water movement. From: WOFOST default/agronomic assumption.
        'SOPE': 10.0,
    },
    'SPIN_UP_PERIOD': {
        # What: Defines the winter months for the soil moisture spin-up. Why: Sets the period for calculating initial conditions. From: User-defined.
        'START_MONTH': 10,
        'END_MONTH': 2,
    },
    'GENERIC_SITE': {
        # What: A fallback location if a district is missing from the GeoJSON. Why: Prevents the script from crashing on missing data. From: User-defined (central Germany).
        'LATITUDE': 52.0, 'LONGITUDE': 10.0, 'ELEVATION': 50.0,
    }
}
# ==============================================================================
# === S C R I P T   S T A R T S   H E R E ===
# ==============================================================================

logging.getLogger().handlers = []
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

class ParameterDict(dict):
    def add_variable(self, name, value, description=""): self[name] = value
    def __getattr__(self, name):
        try: return self[name]
        except KeyError: raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    def __setattr__(self, name, value): self[name] = value
    def copy(self): return ParameterDict(self)

class SimpleWeatherDataProvider(WeatherDataProvider):
    def __init__(self, weather_df, site_data):
        super().__init__()
        self.latitude = site_data['LAT']; self.longitude = site_data['LON']; self.elevation = site_data['ELEV']
        self.angstA = 0.25; self.angstB = 0.5
        weather_df = weather_df.copy(); weather_df['date'] = pd.to_datetime(weather_df['date'])
        self.store = {}
        for _, row in weather_df.iterrows():
            try:
                day = row['date']; tmin = float(row['tmin']); tmax = float(row['tmax'])
                wind = float(row['wind'])
                vap_kpa = float(row['vap'])
                irrad_j_m2_day = float(row['srad']) * 10.0
                irrad_kj_m2_day = irrad_j_m2_day / 1000.0
                precip_mm = float(row.get('precip', 0)) / 100000.0 # Use .get() for safety
                precip_cm = precip_mm / 10.0
                vap_hpa = vap_kpa * 10.0
                et0_mm = penman_monteith(day, self.latitude, self.elevation, tmin, tmax, irrad_kj_m2_day, vap_hpa, wind)
                et0_cm = et0_mm / 10.0
                self.store[(day, 0)] = ParameterDict({'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_cm,
                                                      'IRRAD': irrad_j_m2_day, 'VAP': vap_hpa, 'WIND': wind, 'E0': et0_cm,
                                                      'ES0': et0_cm, 'ET0': et0_cm, 'SNOWDEPTH': 0.0})
            except Exception as e:
                logging.error(f"CRITICAL: Failed processing weather row for date: {row.get('date')}. Error: {e}"); raise e

class WeatherGenerator:
    def __init__(self):
        self.stats = defaultdict(dict)
        self.PRECIP_THRESHOLD_MM = CONFIG['WEATHER_GENERATOR']['PRECIP_THRESHOLD_MM']
        self.MIN_SRAD = CONFIG['WEATHER_GENERATOR']['MIN_SRAD']
    def fit(self, daily_df: pd.DataFrame):
        logging.info("[WEATHER_GEN] Fitting Weather Generator...")
        daily_df = daily_df.copy(); daily_df['month'] = daily_df['date'].dt.month
        daily_df['is_wet'] = (daily_df['precip'] > self.PRECIP_THRESHOLD_MM).astype(int)
        for (district_no, month), group in tqdm(daily_df.groupby(['district_no', 'month']), desc="Learning Weather Patterns"):
            p01 = ((group['is_wet'].shift(-1) == 1) & (group['is_wet'] == 0)).sum(); p00 = ((group['is_wet'].shift(-1) == 0) & (group['is_wet'] == 0)).sum()
            p11 = ((group['is_wet'].shift(-1) == 1) & (group['is_wet'] == 1)).sum(); p10 = ((group['is_wet'].shift(-1) == 0) & (group['is_wet'] == 1)).sum()
            prob_wet_given_dry = p10 / (p10 + p00) if (p10 + p00) > 0 else 0.1
            prob_wet_given_wet = p11 / (p11 + p01) if (p11 + p01) > 0 else 0.5
            wet_day_precip = group[group['is_wet'] == 1]['precip']
            self.stats[(district_no, month)] = {'p_wet_given_dry': prob_wet_given_dry, 'p_wet_given_wet': prob_wet_given_wet,
                                                'precip_wet_day_mean': wet_day_precip.mean() if len(wet_day_precip) > 0 else 1.0,
                                                'precip_wet_day_std': wet_day_precip.std() if len(wet_day_precip) > 1 else 0.5,
                                                'tmin_mean': group['tmin'].mean(), 'tmin_std': max(group['tmin'].std(), 0.5),
                                                'tmax_mean': group['tmax'].mean(), 'tmax_std': max(group['tmax'].std(), 0.5),
                                                'srad_mean': group['srad'].mean(), 'srad_std': max(group['srad'].std(), 0.5)}

    def generate(self, district_no: str, start_date_str: str, end_date_str: str, monthly_anomalies: dict):
        dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D'); generated_data = []
        yesterday_was_wet = np.random.rand() < 0.5
        for date in dates:
            month, key = date.month, (str(district_no).zfill(5), date.month)
            if key not in self.stats: continue
            month_stats = self.stats[key]
            transition_prob = month_stats['p_wet_given_wet'] if yesterday_was_wet else month_stats['p_wet_given_dry']
            today_is_wet = np.random.rand() < transition_prob
            precip = max(0.0, np.random.normal(month_stats['precip_wet_day_mean'], month_stats['precip_wet_day_std'])) if today_is_wet else 0.0
            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std']); tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin: tmax = tmin + abs(np.random.normal(0, 1.0))
            srad = max(self.MIN_SRAD, np.random.normal(month_stats['srad_mean'], month_stats['srad_std']))
            generated_data.append({'date': date, 'tmin': tmin, 'tmax': tmax, 'precip': precip, 'srad': srad}); yesterday_was_wet = today_is_wet
        if not generated_data: return pd.DataFrame()
        synthetic_df = pd.DataFrame(generated_data); synthetic_df['month'] = synthetic_df['date'].dt.month
        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month; key = (str(district_no).zfill(5), month)
            if key not in self.stats: continue
            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0)
            hist_tmean = (self.stats[key]['tmin_mean'] + self.stats[key]['tmax_mean']) / 2; synth_tmean = (synthetic_df.loc[month_mask, 'tmin'].mean() + synthetic_df.loc[month_mask, 'tmax'].mean()) / 2
            temp_correction = (hist_tmean + temp_anomaly) - synth_tmean
            synthetic_df.loc[month_mask, ['tmin', 'tmax']] += temp_correction
            precip_anomaly_factor = 1.0 + monthly_anomalies.get(f'precip_anomaly_{month}', 0)
            target_precip = self.stats[key].get('precip_mean', 0) * month_mask.sum() * precip_anomaly_factor
            synth_precip = synthetic_df.loc[month_mask, 'precip'].sum()
            if synth_precip > 0: synthetic_df.loc[month_mask, 'precip'] *= (target_precip / synth_precip)
            srad_anomaly = monthly_anomalies.get(f'solar_rad_anomaly_{month}', 0)
            if srad_anomaly != 0:
                target_mean_srad = self.stats[key]['srad_mean'] + srad_anomaly
                srad_correction = target_mean_srad - synthetic_df.loc[month_mask, 'srad'].mean()
                synthetic_df.loc[month_mask, 'srad'] = (synthetic_df.loc[month_mask, 'srad'] + srad_correction).clip(lower=self.MIN_SRAD)
        return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad']]

def calculate_initial_soil_moisture(winter_weather_df: pd.DataFrame, soildata: ParameterDict, site_data: ParameterDict) -> float:
    wav = (soildata.SMFCF - soildata.SMW) * soildata.RDMSOL; water_amount_cm = wav
    for _, row in winter_weather_df.iterrows():
        day = row['date'].date()
        tmin = float(row['tmin'])
        tmax = float(row['tmax'])
        irrad_kj_m2_day = float(row['srad']) * 10.0 / 1000.0; precip_mm = float(row.get('precip', 0)) / 100000.0
        wind = float(row['wind'])
        vap_kpa = float(row['vap'])
        et0_mm = penman_monteith(day, site_data.LAT, site_data.ELEV, tmin, tmax, irrad_kj_m2_day, vap_kpa * 10, wind)
        water_amount_cm += (precip_mm / 10.0)
        actual_evaporation_cm = (et0_mm / 10.0) * (water_amount_cm / wav if wav > 0 else 0)
        water_amount_cm -= actual_evaporation_cm
        water_amount_cm = np.clip(water_amount_cm, 0, wav)
    return water_amount_cm

def find_sowing_date(weather_df_spring: pd.DataFrame, year: int, window_start_day: int = 75, window_end_day: int = 120, temp_threshold_c: float = 8.0, temp_period_days: int = 7, precip_threshold_mm: float = 10.0, precip_period_days: int = 3, default_day: int = 105) -> datetime.date:
    df = weather_df_spring[(weather_df_spring['date'].dt.dayofyear >= window_start_day) & (weather_df_spring['date'].dt.dayofyear <= window_end_day)].copy()
    if df.empty: return datetime.date(year, 1, 1) + datetime.timedelta(days=default_day - 1)
    df['temp_rolling_avg'] = df['tmin'].rolling(window=temp_period_days).mean()
    df['precip_rolling_sum'] = df['precip'].rolling(window=precip_period_days).sum()
    suitable_days = df[(df['temp_rolling_avg'] > temp_threshold_c) & (df['precip_rolling_sum'] < precip_threshold_mm)]
    if not suitable_days.empty: return suitable_days['date'].iloc[0].date()
    else: return datetime.date(year, 1, 1) + datetime.timedelta(days=default_day - 1)

def _calculate_soil_hydraulic_properties(sand_frac: float, clay_frac: float, som_frac: float, bdod: float) -> dict:
    c_wp_1 = -0.024*sand_frac + 0.487*clay_frac + 0.006*som_frac + 0.005*(sand_frac*som_frac) - 0.013*(clay_frac*som_frac) + 0.068*(sand_frac*clay_frac) + 0.031
    pwp1500 = c_wp_1 + (0.14*c_wp_1 - 0.02)
    c_fc_1 = -0.251*sand_frac + 0.195*clay_frac + 0.011*som_frac + 0.006*(sand_frac*som_frac) - 0.027*(clay_frac*som_frac) + 0.452*(sand_frac*clay_frac) + 0.299
    fc33 = c_fc_1 + (1.283*c_fc_1**2 - 0.374*c_fc_1 - 0.015)
    porosity = 1 - (bdod / CONFIG['CONSTANTS']['SOIL_PARTICLE_DENSITY'])
    fc33_porosity_adj = fc33 + 0.009*(sand_frac*100)*(porosity - ((fc33 - 0.1)*1.55 + 0.1))
    sm0 = porosity; smw = max(0.01, pwp1500)
    smfcf = min(max(smw + 0.01, fc33_porosity_adj), sm0 - 0.01)
    crairc = max(0.01, sm0 - smfcf)
    return {'SMW': smw, 'SMFCF': smfcf, 'SM0': sm0, 'CRAIRC': crairc}

def _create_district_specific_parameters(static_row, cropdata, verbose=False):
    sitedata = ParameterDict()
    latitude = static_row.get('latitude', CONFIG['GENERIC_SITE']['LATITUDE']); longitude = static_row.get('longitude', CONFIG['GENERIC_SITE']['LONGITUDE'])
    elevation = static_row.get('avg_elevation', CONFIG['GENERIC_SITE']['ELEVATION'])
    sitedata.add_variable('LAT', latitude); sitedata.add_variable('LON', longitude); sitedata.add_variable('ELEV', elevation)
    soildata = ParameterDict()
    try:
        sand = static_row[CONFIG['SOIL_COLUMN_MAPPING']['sand']] / 100.0; clay = static_row[CONFIG['SOIL_COLUMN_MAPPING']['clay']] / 100.0
        som = static_row[CONFIG['SOIL_COLUMN_MAPPING']['som']] / 100.0; bdod = static_row[CONFIG['SOIL_COLUMN_MAPPING']['bdod']]
        calculated_soil_params = _calculate_soil_hydraulic_properties(sand, clay, som, bdod)
        if verbose:
            logging.info(f"--- SOIL DIAGNOSTICS for District: {static_row.get('district_no', 'N/A')} ---")
            logging.info(f"  Inputs: Sand={sand*100:.2f}%, Clay={clay*100:.2f}%, SOM={som*100:.2f}%, BDOD={bdod:.2f} g/cm3")
            logging.info(f"  Outputs: SMW={calculated_soil_params['SMW']:.3f}, SMFCF={calculated_soil_params['SMFCF']:.3f}, SM0={calculated_soil_params['SM0']:.3f}")
        for key, value in calculated_soil_params.items(): soildata.add_variable(key, value)
    except (KeyError, TypeError) as e:
        logging.error(f"FATAL: Missing soil data for district {static_row.get('district_no', 'N/A')}. Error: {e}"); raise e
    for key, value in CONFIG['SOIL_DEFAULTS_AND_CONSTANTS'].items(): soildata.add_variable(key, value)
    smfc = soildata['SMFCF']; smw = soildata['SMW']; rdi = CONFIG['CONSTANTS']['INITIAL_ROOTING_DEPTH_CM']
    rdmsol = soildata['RDMSOL']; smlim = smfc * rdi; wav = (smfc - smw) * rdmsol
    sitedata.add_variable('SMLIM', smlim); sitedata.add_variable('WAV', wav); sitedata.add_variable('IFUNRN', 0.0)
    sitedata.add_variable('NOTINF', 0.0); sitedata.add_variable('SSI', 0.0); sitedata.add_variable('SSMAX', 0.0)
    return ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata), sitedata

def load_data_and_setup_model(year, cfg):
    logging.info("="*70 + "\n[SETUP] Loading input data...\n" + "="*70)
    try:
        df_static = pd.read_csv(cfg['FILE_PATHS']['STATIC_FEATURES']); df_daily_hist = pd.read_csv(cfg['FILE_PATHS']['HISTORICAL_DAILY_WEATHER'], parse_dates=['date'])
        df_seas5 = pd.read_csv(cfg['FILE_PATHS']['SEAS5_MEMBER_FEATURES']); gdf_districts = gpd.read_file(cfg['FILE_PATHS']['DISTRICTS_GEOJSON'])
        logging.info(f"[DATA] Loaded {len(gdf_districts)} district geometries.")
    except FileNotFoundError as e:
        logging.error(f"[DATA] FATAL: Input file not found. {e}"); return None, None, None, None
    gdf_districts_proj = gdf_districts.to_crs(epsg=3035); centroids_wgs84 = gdf_districts_proj.geometry.centroid.to_crs(epsg=4326)
    gdf_districts['latitude'] = centroids_wgs84.y; gdf_districts['longitude'] = centroids_wgs84.x
    df_geo = gdf_districts[['id', 'latitude', 'longitude']].rename(columns={'id': 'district_no'})
    df_geo['district_no'] = pd.to_numeric(df_geo['district_no'], errors='coerce').astype('Int64').astype(str).str.zfill(5)
    for df in [df_static, df_daily_hist, df_seas5]:
        df['district_no'] = pd.to_numeric(df['district_no'], errors='coerce').astype('Int64').astype(str).str.zfill(5)
    df_static = pd.merge(df_static, df_geo, on='district_no', how='left')
    if df_static['latitude'].isna().sum() > 0: logging.warning(f"[DATA] {df_static['latitude'].isna().sum()} districts had no matching geometry.")
    else: logging.info("[DATA] ✓ Successfully merged all district geometries.")
    df_static = df_static[df_static['year'] == year].copy(); df_seas5 = df_seas5[df_seas5['year'] == year].copy()
    if cfg['DISTRICT_LIMIT'] is not None:
        districts_to_run = df_static['district_no'].unique()[:cfg['DISTRICT_LIMIT']]
        df_static = df_static[df_static['district_no'].isin(districts_to_run)].copy(); df_seas5 = df_seas5[df_seas5['district_no'].isin(districts_to_run)].copy()
        logging.info(f"[QUICK TEST] Now running with districts: {list(districts_to_run)}\n")
    try:
        with open(cfg['FILE_PATHS']['CROP_YAML'], 'r') as f: crop_params = yaml.safe_load(f)['CropParameters']
        variety_name = [k for k in crop_params.keys() if not k.startswith('Generic')][0]
        logging.info(f"[PARAMS] Using crop variety: {variety_name}")
        cropdata = ParameterDict()
        def add_params(param_dict, data):
            for key, val in data.items():
                if isinstance(val, dict): add_params(param_dict, val)
                elif isinstance(val, list) and len(val) > 0: param_dict.add_variable(key, val[0])
        add_params(cropdata, {**crop_params.get('GenericC3', {}), **crop_params.get('Generic', {})}); add_params(cropdata, crop_params[variety_name])
        logging.info(f"[PARAMS] Loaded {len(cropdata)} crop parameters.")
    except Exception as e:
        logging.error(f"[PARAMS] FATAL: Could not load crop parameters: {e}", exc_info=True); return None, None, None, None
    return df_static, df_daily_hist, df_seas5, cropdata


def run_historical_simulation(df_static, df_daily_hist, cropdata, year, cfg):
    logging.info("=" * 70 + f"\n[HISTORICAL] Running Historical Simulation for {year}\n" + "=" * 70)
    results = []
    spin_up_cfg = cfg['SPIN_UP_PERIOD']
    for _, row in tqdm(df_static.iterrows(), total=len(df_static), desc="Historical Sim"):
        district_no = row['district_no']
        # Get all historical weather for the district across all years available
        weather_df = df_daily_hist[df_daily_hist['district_no'] == district_no].copy()
        if weather_df.empty:
            logging.warning(f"No historical weather data at all for district {district_no}, skipping.")
            continue

        try:
            # --- DYNAMIC INITIALIZATION LOGIC ---
            parameters, site_data = _create_district_specific_parameters(row, cropdata, verbose=False)

            # 1. Spin-up for initial soil moisture using the actual preceding winter
            winter_start = pd.to_datetime(f"{year - 1}-{spin_up_cfg['START_MONTH']}-01")
            winter_end = pd.to_datetime(f"{year}-{spin_up_cfg['END_MONTH']}-28")  # Safe for leap years
            actual_winter_df = weather_df[(weather_df['date'] >= winter_start) & (weather_df['date'] <= winter_end)]
            initial_water_amount_cm = calculate_initial_soil_moisture(actual_winter_df, parameters.soildata,
                                                                      site_data) if not actual_winter_df.empty else None

            # 2. Dynamic Sowing Date with a buffer for rolling calculations
            # We need weather from Dec of the previous year to properly calculate rolling averages in March
            sowing_logic_start_date = pd.to_datetime(f"{year - 1}-12-01")
            sowing_logic_end_date = pd.to_datetime(f"{year}-05-31")
            weather_for_sowing_logic = weather_df[
                (weather_df['date'] >= sowing_logic_start_date) & (weather_df['date'] <= sowing_logic_end_date)].copy()

            crop_start = None
            if not weather_for_sowing_logic.empty:
                # The sowing function expects units in mm, but our provider expects the raw data.
                # So we create a temporary copy with corrected units just for this function.
                weather_for_sowing_logic['precip'] = weather_for_sowing_logic['precip'] / 100000.0
                crop_start = find_sowing_date(weather_for_sowing_logic, year)
            else:
                # If there's no data for the sowing window (e.g., first year of dataset), fallback.
                crop_start = datetime.date(year, 4, 15)

            # --- MODEL EXECUTION ---
            # The main weather provider ONLY gets data for the simulation year
            weather_df_year = weather_df[weather_df['date'].dt.year == year].copy()
            if weather_df_year.empty:
                logging.error(f"[HISTORICAL] No weather data found for district {district_no} in year {year}.")
                simulated_yield = np.nan
            else:
                weather_provider = SimpleWeatherDataProvider(weather_df_year, site_data)
                crop_end = cfg['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)
                agromanagement = [{crop_start: ParameterDict({'CropCalendar': ParameterDict(
                    {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
                     'crop_end_type': 'harvest', 'max_duration': cfg['AGROMANAGEMENT']['MAX_DURATION']}),
                    'TimedEvents': None, 'StateEvents': None})}]

                model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)

                # Override the initial soil water state
                if initial_water_amount_cm is not None:
                    model.__waterbalance__.W = initial_water_amount_cm

                model.run_till_terminate()
                output = model.get_output()
                simulated_yield = output[-1]['TWSO'] if output else np.nan

        except Exception as e:
            logging.error(f"[HISTORICAL] ERROR for district {district_no}: {e}", exc_info=True)
            simulated_yield = np.nan

        results.append({'year': year, 'district_no': district_no, 'actual_yield': row['kreisYield'],
                        'lintul_yield_perfect_weather': simulated_yield})

    logging.info(f"[HISTORICAL] Completed {len(results)} simulations.\n")
    return pd.DataFrame(results)


def _run_single_forecast_member(member_row, district_no, year, wg, parameters, site_data, cfg, avg_winter_df):
    """
    Runs a complete forecast simulation for a single ensemble member, including:
    1. Dynamic sowing date simulation based on a synthetic spring.
    2. Dynamic initial soil moisture based on a climatological winter spin-up.
    """
    try:
        monthly_anomalies = {}
        for month in range(3, 11):
            monthly_anomalies[f'temp_anomaly_{month}'] = member_row.get(f'temp_anomaly_forecast_{month}', 0)
            monthly_anomalies[f'precip_anomaly_{month}'] = member_row.get(f'precip_anomaly_forecast_{month}', 0)
            monthly_anomalies[f'solar_rad_anomaly_{month}'] = member_row.get(f'solar_rad_anomaly_forecast_{month}', 0)

        # --- DYNAMIC SOWING DATE LOGIC ---
        spring_start, spring_end = f'{year}-03-01', f'{year}-05-31'
        synth_spring_weather = wg.generate(district_no, spring_start, spring_end, monthly_anomalies)
        if synth_spring_weather.empty:
            return np.nan
        crop_start = find_sowing_date(synth_spring_weather, year)

        season_start, season_end = f'{year}-03-01', f'{year}-10-31'
        synth_full_season_weather = wg.generate(district_no, season_start, season_end, monthly_anomalies)
        if synth_full_season_weather.empty:
            return np.nan

        # --- DYNAMIC INITIAL SOIL MOISTURE LOGIC ---
        initial_water_amount_cm = None
        if avg_winter_df is not None and not avg_winter_df.empty:
            dates = []
            start_year_winter = year - 1
            for _, row in avg_winter_df.iterrows():
                day_of_year = int(row['day_of_year'])
                current_year_winter = start_year_winter if day_of_year > 180 else year
                try:
                    base_date = datetime.date(current_year_winter, 1, 1) + datetime.timedelta(days=day_of_year - 1)
                    date = pd.Timestamp(base_date)
                    dates.append(date)
                except (ValueError, OverflowError):
                    continue

            # Ensure we don't have a mismatch if a date failed (e.g., leap day)
            avg_winter_df = avg_winter_df.iloc[:len(dates)].copy()
            avg_winter_df['date'] = dates

            initial_water_amount_cm = calculate_initial_soil_moisture(avg_winter_df, parameters.soildata, site_data)

            wav = (parameters.soildata.SMFCF - parameters.soildata.SMW) * parameters.soildata.RDMSOL
            initial_fraction = initial_water_amount_cm / wav if wav > 0 else 0
            logging.debug(f"Dist {district_no} initial soil water fraction: {initial_fraction:.2f}")

        # --- MODEL EXECUTION ---
        weather_provider = SimpleWeatherDataProvider(synth_full_season_weather, site_data)

        crop_end = cfg['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)
        agromanagement = [{crop_start: ParameterDict({'CropCalendar': ParameterDict(
            {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
             'crop_end_type': 'harvest', 'max_duration': cfg['AGROMANAGEMENT']['MAX_DURATION']}),
            'TimedEvents': None, 'StateEvents': None})}]

        model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)

        if initial_water_amount_cm is not None:
            model.__waterbalance__.W = initial_water_amount_cm

        model.run_till_terminate()
        output = model.get_output()
        return output[-1]['TWSO'] if output else np.nan

    except Exception as e:
        member_id = member_row.get('member', 'N/A')
        # This error is noisy during parallel execution; we'll log it as debug.
        logging.debug(f"[FORECAST_WORKER] Error for dist {district_no}, member {member_id}: {e}", exc_info=True)
        return np.nan

def run_forecast_simulation(df_static, df_daily_hist, df_seas5, cropdata, year, cfg):
    logging.info("=" * 70 + f"\n[FORECAST] Running Parallel Forecast Simulation for {year}\n" + "=" * 70)
    wg = WeatherGenerator(); wg.fit(df_daily_hist)
    logging.info("Pre-calculating climatological winter weather for spin-up...")
    climatological_winters = {}
    spin_up_cfg = cfg['SPIN_UP_PERIOD']
    winter_mask = (df_daily_hist['date'].dt.month >= spin_up_cfg['START_MONTH']) | (df_daily_hist['date'].dt.month <= spin_up_cfg['END_MONTH'])
    df_winter_all_years = df_daily_hist[winter_mask].copy()
    for district_no, group in tqdm(df_winter_all_years.groupby('district_no'), desc="Averaging Winters"):
        group['day_of_year'] = group['date'].dt.dayofyear
        avg_winter = group.groupby('day_of_year')[['tmin', 'tmax', 'srad', 'precip', 'wind', 'vap']].mean().reset_index()
        climatological_winters[district_no] = avg_winter
    district_params_cache = {row['district_no']: _create_district_specific_parameters(row, cropdata, verbose=False) for _, row in df_static.iterrows()}
    forecast_results = []
    for district_no, group in tqdm(df_seas5.groupby('district_no'), desc="Forecast Sim (Districts)"):
        if district_no not in district_params_cache: continue
        parameters, site_data = district_params_cache[district_no]
        avg_winter_df = climatological_winters.get(district_no)
        tasks = [delayed(_run_single_forecast_member)(member_row, district_no, year, wg, parameters, site_data, cfg, avg_winter_df) for _, member_row in group.iterrows()]
        ensemble_yields = Parallel(n_jobs=-1, backend='loky')(tasks)
        valid_yields = [y for y in ensemble_yields if not np.isnan(y)]
        forecast_results.append({'year': year, 'district_no': district_no, 'lintul_yield_forecast_weather': np.mean(valid_yields) if valid_yields else np.nan,
                                 'forecast_uncertainty_std': np.std(valid_yields) if valid_yields else np.nan})
    logging.info(f"[FORECAST] Completed {len(forecast_results)} district ensembles.\n")
    return pd.DataFrame(forecast_results)

def analyze_and_plot_results(df_hist, df_fcst, year, cfg):
    logging.info("=" * 70 + "\n[ANALYSIS] Analyzing Results\n" + "=" * 70)
    df_final = pd.merge(df_hist, df_fcst, on=['year', 'district_no']).dropna()
    if df_final.empty: logging.error("[ANALYSIS] No valid results to analyze!"); return
    dmc = cfg['CONSTANTS']['DMC_SUGARBEET']
    df_final['actual_yield_dry_kgha'] = df_final['actual_yield'] * 100.0 * dmc
    df_final = df_final.rename(columns={'lintul_yield_perfect_weather': 'perfect_yield_dry_kgha', 'lintul_yield_forecast_weather': 'forecast_yield_dry_kgha'})
    for col in ['actual_yield_dry_kgha', 'perfect_yield_dry_kgha', 'forecast_yield_dry_kgha']:
        df_final[col.replace('_dry_kgha', '_dt')] = df_final[col] / 100.0
    output_path = os.path.join(cfg['FILE_PATHS']['OUTPUT_DIR'], f'final_comparison_{year}_TEST.csv')
    df_final.to_csv(output_path, index=False); logging.info(f"[ANALYSIS] ✓ Results saved to {output_path}")
    print("\n--- Performance Metrics (Dry Weight dt/ha) ---")
    mae_p = mean_absolute_error(df_final['actual_yield_dt'], df_final['perfect_yield_dt']); r2_p = r2_score(df_final['actual_yield_dt'], df_final['perfect_yield_dt'])
    mae_f = mean_absolute_error(df_final['actual_yield_dt'], df_final['forecast_yield_dt']); r2_f = r2_score(df_final['actual_yield_dt'], df_final['forecast_yield_dt'])
    print(f"  Perfect Weather:  MAE = {mae_p:.2f}, R² = {r2_p:.3f}"); print(f"  Forecast Weather: MAE = {mae_f:.2f}, R² = {r2_f:.3f}\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)); fig.suptitle(f'WOFOST Performance Comparison - {year} (Dry Weight)', fontsize=14, fontweight='bold')
    min_val = df_final[['actual_yield_dt', 'perfect_yield_dt', 'forecast_yield_dt']].min().min() * 0.9
    max_val = df_final[['actual_yield_dt', 'perfect_yield_dt', 'forecast_yield_dt']].max().max() * 1.1
    axes[0].scatter(df_final['actual_yield_dt'], df_final['perfect_yield_dt'], alpha=0.6); axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    axes[0].set_title(f'Perfect Weather\nMAE={mae_p:.2f}, R²={r2_p:.3f}'); axes[0].set_xlabel('Actual Yield (dt/ha)'); axes[0].set_ylabel('Simulated Yield (dt/ha)')
    axes[1].scatter(df_final['actual_yield_dt'], df_final['forecast_yield_dt'], alpha=0.6, color='orange'); axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    axes[1].set_title(f'Forecast Weather\nMAE={mae_f:.2f}, R²={r2_f:.3f}'); axes[1].set_xlabel('Actual Yield (dt/ha)'); axes[1].set_ylabel('Simulated Yield (dt/ha)')
    for ax in axes: ax.set_xlim(min_val, max_val); ax.set_ylim(min_val, max_val); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plot_path = os.path.join(cfg['FILE_PATHS']['OUTPUT_DIR'], f'results_{year}.png'); plt.savefig(plot_path, dpi=300)
    logging.info(f"[ANALYSIS] ✓ Plot saved to {plot_path}"); plt.show()


if __name__ == "__main__":
    test_year = CONFIG['TEST_YEAR']
    os.makedirs(CONFIG['FILE_PATHS']['OUTPUT_DIR'], exist_ok=True)
    logging.info("=" * 70 + f"\nWOFOST ONE-YEAR TEST PIPELINE - {test_year}\n" + "=" * 70)
    df_static, df_daily_hist, df_seas5, cropdata = load_data_and_setup_model(test_year, CONFIG)
    if cropdata is not None:
        logging.info("=" * 70 + "\n[DIAGNOSTICS] Forcing parameter creation for initial districts to show logs\n" + "=" * 70)
        for _, row in df_static.iterrows():
            try: _create_district_specific_parameters(row, cropdata, verbose=True)
            except Exception as e: logging.error(f"Failed during pre-computation logging for district {row.get('district_no')}: {e}")
        logging.info("=" * 70 + "\n[DIAGNOSTICS] Pre-computation logging complete. Starting simulations.\n" + "=" * 70)
        df_hist = run_historical_simulation(df_static, df_daily_hist, cropdata, test_year, CONFIG)
        df_fcst = run_forecast_simulation(df_static, df_daily_hist, df_seas5, cropdata, test_year, CONFIG)
        if not df_hist.empty and not df_fcst.empty:
            analyze_and_plot_results(df_hist, df_fcst, test_year, CONFIG)
            logging.info("\n" + "=" * 70 + "\n✓ PIPELINE COMPLETED SUCCESSFULLY!\n" + "=" * 70)
        else: logging.error("Simulations failed - no results to analyze")
    else: logging.error("Pipeline aborted - data loading or setup failed")
    logging.shutdown()
