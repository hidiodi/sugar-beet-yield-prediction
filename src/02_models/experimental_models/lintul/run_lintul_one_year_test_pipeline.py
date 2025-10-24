# File: run_lintul_one_year_test_pipeline.py
# Description: A consolidated script to run the full one-year test pipeline.
#
# REFACTORED & PARALLELIZED VERSION v3: Now integrates district-specific
# latitude and longitude by reading a GeoJSON file and calculating the centroid
# of each district's polygon. The generic site data is now only a fallback.

import datetime
import yaml
import pandas as pd
import numpy as np
import os
import logging

# --- NEW LIBRARY IMPORT ---
# Geopandas is used to read the GeoJSON file and work with geometries.
# You will need to install it: pip install geopandas
import geopandas as gpd

from pcse.util import penman_monteith

# --- PCSE Imports ---
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider, WeatherDataProvider

# --- Matplotlib, Sklearn, and Joblib Imports ---
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import Parallel, delayed

# ==============================================================================
# === G L O B A L   C O N F I G U R A T I O N ===
# ==============================================================================
CONFIG = {
    'TEST_YEAR': 2018,
    'DISTRICT_LIMIT': 2,
    'FILE_PATHS': {
        'HISTORICAL_DAILY_WEATHER': 'data/02_intermediate/historical_daily_weather_era5_2018_TEST.csv',
        'STATIC_FEATURES': 'data/05_model_input/stage1_preseason_features.csv',
        'SEAS5_MEMBER_FEATURES': 'data/02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv',
        'CROP_YAML': 'data/01_raw/sugarbeet.yaml',
        'OUTPUT_DIR': 'data/06_model_output/one_year_test',
        'DISTRICTS_GEOJSON': 'data/01_raw/districts_official.geojson',
    },
    # --- Weather Provider Placeholders ---
    # STATUS: These are major placeholders and should be replaced by real data.
    # TODO: Your ERA5 and SEAS5 datasets should be updated to include wind and dewpoint/vapor pressure.
    'WEATHER_DEFAULTS': {
        'WIND_SPEED': 2.0,  # Placeholder wind speed in m/s.
        'VAPOR_PRESSURE': 1.0,  # Placeholder vapor pressure in kPa.
    },
    # --- Weather Generator Parameters ---
    'WEATHER_GENERATOR': {
        'PRECIP_THRESHOLD_MM': 0.3, # Defines a "wet day" for the Markov chain. STATUS: Justified assumption.
        'MIN_SRAD': 100.0, # Minimum generated solar radiation (J/m2/day). STATUS: Justified assumption to prevent errors.
    },
    'AGROMANAGEMENT': {
        'CROP_START_DATE': datetime.date(2018, 3, 15), # Fixed emergence date.
        'CROP_END_DATE': datetime.date(2018, 10, 20),   # Fixed harvest date.
        'MAX_DURATION': 250, # Max crop duration in days.
    },

    # --- Biophysical Constants ---
    'CONSTANTS': {
        'DMC_SUGARBEET': 0.25, # Dry Matter Content of sugar beet. SOURCE: Standard literature value. STATUS: Justified assumption.
        'INITIAL_ROOTING_DEPTH_CM': 10.0, # Assumed initial rooting depth (RDI). SOURCE: WOFOST default. STATUS: Justified assumption.
    },

    # --- Generic Soil Parameters ---
    # STATUS: CRITICAL PLACEHOLDER. These values represent a single, generic "loam" soil
    # for all of Germany. This is the most important section to replace with dynamic,
    # data-driven values using your GEE soil data and a Pedotransfer Function (PTF).
    # The function `_create_district_specific_parameters` is designed for this.
    'GENERIC_SOIL': {
        'SMW': 0.10,     # Soil moisture at Wilting Point (fraction).
        'SMFCF': 0.30,   # Soil moisture at Field Capacity (fraction).
        'SM0': 0.40,     # Soil moisture at Saturation (fraction).
        'CRAIRC': 0.06,  # Critical air content (fraction).
        'RDMSOL': 150.0, # Maximum Soil Rooting Depth (cm).
        'KSUB': 10.0,    # Max percolation rate of subsoil (cm/day).
        'SOPE': 10.0,    # Max percolation rate of root zone (cm/day).
    },
    # --- UPDATED ROLE ---
    # This is now a FALLBACK for districts not found in the GeoJSON.
    'GENERIC_SITE': {
        'LATITUDE': 52.0, 'LONGITUDE': 10.0, 'ELEVATION': 50.0,
    }
}

# ==============================================================================
# === S C R I P T   S T A R T S   H E R E ===
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')


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
        self.latitude = site_data['LAT'];
        self.longitude = site_data['LON'];
        self.elevation = site_data['ELEV']
        self.angstA = 0.25;
        self.angstB = 0.5
        weather_df = weather_df.copy();
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        self.store = {}
        placeholder_wind = CONFIG['WEATHER_DEFAULTS']['WIND_SPEED'];
        placeholder_vap_kpa = CONFIG['WEATHER_DEFAULTS']['VAPOR_PRESSURE']
        for _, row in weather_df.iterrows():
            try:
                day = row['date'].date();
                tmin = float(row['tmin']);
                tmax = float(row['tmax'])
                wind = float(row.get('wind', placeholder_wind));
                vap_kpa = float(row.get('vap', placeholder_vap_kpa))
                irrad_j_m2_day = float(row['srad']) * 10.0;
                irrad_kj_m2_day = irrad_j_m2_day / 1000.0
                precip_mm = float(row['precip']) / 100000.0;
                precip_cm = precip_mm / 10.0
                vap_hpa = vap_kpa * 10.0
                et0_mm = penman_monteith(day, self.latitude, self.elevation, tmin, tmax, irrad_kj_m2_day, vap_hpa, wind)
                et0_cm = et0_mm / 10.0
                data_dict = ParameterDict(
                    {'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_cm,
                     'IRRAD': irrad_j_m2_day, 'VAP': vap_hpa, 'WIND': wind, 'SNOWDEPTH': 0.0, 'E0': et0_cm,
                     'ES0': et0_cm, 'ET0': et0_cm})
                self.store[(day, 0)] = data_dict
            except Exception as e:
                logging.error(f"CRITICAL: Failed processing weather row for date: {row.get('date')}. Error: {e}");
                raise e
        logging.debug(f"Successfully loaded {len(self.store)} days into weather provider")


class WeatherGenerator:
    def __init__(self):
        self.stats = defaultdict(dict)
        self.PRECIP_THRESHOLD_MM = CONFIG['WEATHER_GENERATOR']['PRECIP_THRESHOLD_MM']
        self.MIN_SRAD = CONFIG['WEATHER_GENERATOR']['MIN_SRAD']

    def fit(self, daily_df: pd.DataFrame):
        logging.info("[WEATHER_GEN] Fitting Weather Generator...")
        daily_df = daily_df.copy();
        daily_df['month'] = daily_df['date'].dt.month
        daily_df['is_wet'] = (daily_df['precip'] > self.PRECIP_THRESHOLD_MM).astype(int)
        for (district_no, month), group in tqdm(daily_df.groupby(['district_no', 'month']),
                                                desc="Learning Weather Patterns"):
            p01 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 1)).sum();
            p00 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 0)).sum()
            p11 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 1)).sum();
            p10 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 0)).sum()
            prob_wet_given_dry = p01 / (p01 + p00) if (p01 + p00) > 0 else 0.1
            prob_wet_given_wet = p11 / (p11 + p10) if (p11 + p10) > 0 else 0.5
            wet_day_precip = group[group['is_wet'] == 1]['precip']
            self.stats[(district_no, month)] = {'p_wet_given_dry': prob_wet_given_dry,
                                                'p_wet_given_wet': prob_wet_given_wet,
                                                'precip_wet_day_mean': wet_day_precip.mean() if len(
                                                    wet_day_precip) > 0 else 1.0,
                                                'precip_wet_day_std': wet_day_precip.std() if len(
                                                    wet_day_precip) > 1 else 0.5, 'precip_mean': group['precip'].mean(),
                                                'tmin_mean': group['tmin'].mean(),
                                                'tmin_std': max(group['tmin'].std(), 0.5),
                                                'tmax_mean': group['tmax'].mean(),
                                                'tmax_std': max(group['tmax'].std(), 0.5),
                                                'srad_mean': group['srad'].mean(),
                                                'srad_std': max(group['srad'].std(), 0.5)}
        logging.info(f"[WEATHER_GEN] Learned statistics for {len(self.stats)} district-month pairs")

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
            if today_is_wet: precip = max(0, np.random.normal(month_stats['precip_wet_day_mean'],
                                                              month_stats['precip_wet_day_std']))
            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std']);
            tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin: tmax = tmin + abs(np.random.normal(0, 1.0))
            srad = max(self.MIN_SRAD, np.random.normal(month_stats['srad_mean'], month_stats['srad_std']))
            generated_data.append({'date': date, 'tmin': tmin, 'tmax': tmax, 'precip': precip, 'srad': srad});
            yesterday_was_wet = today_is_wet
        if not generated_data: return pd.DataFrame()
        synthetic_df = pd.DataFrame(generated_data);
        synthetic_df['month'] = synthetic_df['date'].dt.month
        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month;
            key = (str(district_no).zfill(5), month)
            if key not in self.stats: continue
            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0);
            synth_tmean = (synthetic_df.loc[month_mask, 'tmin'].mean() + synthetic_df.loc[
                month_mask, 'tmax'].mean()) / 2
            hist_tmean = (self.stats[key]['tmin_mean'] + self.stats[key]['tmax_mean']) / 2;
            temp_correction = (hist_tmean + temp_anomaly) - synth_tmean
            synthetic_df.loc[month_mask, 'tmin'] += temp_correction;
            synthetic_df.loc[month_mask, 'tmax'] += temp_correction
            precip_anomaly_factor = 1.0 + monthly_anomalies.get(f'precip_anomaly_{month}', 0)
            synth_precip, hist_precip = synthetic_df.loc[month_mask, 'precip'].sum(), self.stats[key][
                                                                                          'precip_mean'] * month_mask.sum()
            target_precip = hist_precip * precip_anomaly_factor
            if synth_precip > 0: synthetic_df.loc[month_mask, 'precip'] *= (target_precip / synth_precip)
        return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad']]


def _create_district_specific_parameters(static_row, cropdata):
    """Creates a complete PCSE ParameterProvider for a single district."""
    sitedata = ParameterDict()

    # --- SITE DATA: Use specific data from row, with generic as a fallback ---
    latitude = static_row.get('latitude', CONFIG['GENERIC_SITE']['LATITUDE'])
    longitude = static_row.get('longitude', CONFIG['GENERIC_SITE']['LONGITUDE'])
    elevation = static_row.get('avg_elevation', CONFIG['GENERIC_SITE']['ELEVATION'])
    sitedata.add_variable('LAT', latitude)
    sitedata.add_variable('LON', longitude)
    sitedata.add_variable('ELEV', elevation)

    # --- SOIL DATA: Currently generic, prepared for PTF integration ---
    soildata = ParameterDict()
    for key, value in CONFIG['GENERIC_SOIL'].items():
        soildata.add_variable(key, value)

    # --- Initial Water Availability (derived from soil) ---
    smfc = soildata['SMFCF'];
    smw = soildata['SMW'];
    rdi = CONFIG['CONSTANTS']['INITIAL_ROOTING_DEPTH_CM'];
    rdmsol = soildata['RDMSOL']
    smlim = smfc * rdi;
    wav = (smfc - smw) * rdmsol
    sitedata.add_variable('SMLIM', smlim);
    sitedata.add_variable('WAV', wav)
    sitedata.add_variable('IFUNRN', 0.0);
    sitedata.add_variable('NOTINF', 0.0);
    sitedata.add_variable('SSI', 0.0);
    sitedata.add_variable('SSMAX', 0.0)

    return ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata), sitedata


def load_data_and_setup_model(year, cfg):
    logging.info("=" * 70 + "\n[SETUP] Loading input data and setting up model\n" + "=" * 70)
    try:
        df_static = pd.read_csv(cfg['FILE_PATHS']['STATIC_FEATURES'])
        df_daily_hist = pd.read_csv(cfg['FILE_PATHS']['HISTORICAL_DAILY_WEATHER'], parse_dates=['date'])
        df_seas5 = pd.read_csv(cfg['FILE_PATHS']['SEAS5_MEMBER_FEATURES'])
        # --- NEW: Load GeoJSON file ---
        gdf_districts = gpd.read_file(cfg['FILE_PATHS']['DISTRICTS_GEOJSON'])
        logging.info(f"[DATA] Loaded {len(gdf_districts)} district geometries from GeoJSON.")
    except FileNotFoundError as e:
        logging.error(f"[DATA] FATAL: Input file not found. {e}");
        return None, None, None, None

    # --- NEW: Calculate centroids and merge into static features ---
    gdf_districts['latitude'] = gdf_districts.geometry.centroid.y
    gdf_districts['longitude'] = gdf_districts.geometry.centroid.x
    df_geo = gdf_districts[['id', 'latitude', 'longitude']].rename(columns={'id': 'district_no'})
    df_geo['district_no'] = pd.to_numeric(df_geo['district_no'], errors='coerce').astype('Int64').astype(str).str.zfill(
        5)

    # Standardize district_no in main dataframes before merging
    for df in [df_static, df_daily_hist, df_seas5]:
        df['district_no'] = pd.to_numeric(df['district_no'], errors='coerce').astype('Int64').astype(str).str.zfill(5)

    # Merge the geographic data into the static features table
    df_static = pd.merge(df_static, df_geo, on='district_no', how='left')
    missing_geo = df_static['latitude'].isna().sum()
    if missing_geo > 0:
        logging.warning(
            f"[DATA] {missing_geo} districts in static features had no matching geometry. They will use fallback site data.")
    else:
        logging.info("[DATA] ✓ Successfully merged all district geometries.")

    # Filter for the test year
    df_static = df_static[df_static['year'] == year].copy()
    df_seas5 = df_seas5[df_seas5['year'] == year].copy()

    if cfg['DISTRICT_LIMIT'] is not None:
        logging.warning(f"[QUICK TEST] Limiting run to the first {cfg['DISTRICT_LIMIT']} available districts.")
        districts_to_run = df_static['district_no'].unique()[:cfg['DISTRICT_LIMIT']]
        df_static = df_static[df_static['district_no'].isin(districts_to_run)].copy()
        df_seas5 = df_seas5[df_seas5['district_no'].isin(districts_to_run)].copy()
        logging.info(f"[QUICK TEST] Now running with districts: {list(districts_to_run)}\n")

    try:
        with open(cfg['FILE_PATHS']['CROP_YAML'], 'r') as f:
            crop_params = yaml.safe_load(f)['CropParameters']
        variety_name = [k for k in crop_params.keys() if not k.startswith('Generic')][0]
        logging.info(f"[PARAMS] Using crop variety: {variety_name}")
        cropdata = ParameterDict()

        def add_params(param_dict, data):
            for key, val in data.items():
                if isinstance(val, dict):
                    add_params(param_dict, val)
                elif isinstance(val, list) and len(val) > 0:
                    param_dict.add_variable(key, val[0])

        generic_data = {**crop_params.get('GenericC3', {}), **crop_params.get('Generic', {})}
        add_params(cropdata, generic_data);
        add_params(cropdata, crop_params[variety_name])
        logging.info(f"[PARAMS] Loaded {len(cropdata)} crop parameters (flattened)")
    except Exception as e:
        logging.error(f"[PARAMS] FATAL: Could not load crop parameters: {e}", exc_info=True);
        return None, None, None, None

    return df_static, df_daily_hist, df_seas5, cropdata


def run_historical_simulation(df_static, df_daily_hist, cropdata, year, cfg):
    logging.info("=" * 70 + f"\n[HISTORICAL] Running Historical Simulation for {year}\n" + "=" * 70)
    df_daily_hist_year = df_daily_hist[df_daily_hist['date'].dt.year == year].copy()
    if df_daily_hist_year.empty: logging.error(
        f"[HISTORICAL] FATAL: No weather data for {year}!"); return pd.DataFrame()
    results = []
    for _, row in tqdm(df_static.iterrows(), total=len(df_static), desc="Historical Sim"):
        district_no = row['district_no'];
        weather_df = df_daily_hist_year[df_daily_hist_year['district_no'] == district_no].copy()
        if weather_df.empty: logging.warning(
            f"[HISTORICAL] No weather for district {district_no} in year {year}"); continue
        try:
            parameters, site_data = _create_district_specific_parameters(row, cropdata);
            weather_provider = SimpleWeatherDataProvider(weather_df, site_data)
            crop_start = cfg['AGROMANAGEMENT']['CROP_START_DATE'].replace(year=year);
            crop_end = cfg['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)
            agromanagement = [{crop_start: ParameterDict({'CropCalendar': ParameterDict(
                {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
                 'crop_end_type': 'harvest', 'max_duration': cfg['AGROMANAGEMENT']['MAX_DURATION']}),
                                                          'TimedEvents': None, 'StateEvents': None})}]
            model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement);
            model.run_till_terminate();
            output = model.get_output()
            simulated_yield = output[-1]['TWSO'] if output else np.nan
        except Exception as e:
            logging.error(f"[HISTORICAL] ERROR for district {district_no}: {e}");
            simulated_yield = np.nan
        results.append({'year': year, 'district_no': district_no, 'actual_yield': row['kreisYield'],
                        'lintul_yield_perfect_weather': simulated_yield})
    logging.info(f"[HISTORICAL] Completed {len(results)} simulations\n")
    return pd.DataFrame(results)


def _run_single_forecast_member(member_row, district_no, year, wg, parameters, site_data, cfg):
    try:
        monthly_anomalies = {}
        for month in range(3, 11):
            season = 'spring' if month <= 6 else 'summer'
            monthly_anomalies[f'temp_anomaly_{month}'] = member_row.get(f'{season}_temp_anomaly_forecast', 0)
            monthly_anomalies[f'precip_anomaly_{month}'] = member_row.get(f'{season}_precip_anomaly_forecast', 0)
        start_date, end_date = f'{year}-03-01', f'{year}-10-31';
        synth_weather = wg.generate(district_no, start_date, end_date, monthly_anomalies)
        if synth_weather.empty: return np.nan
        weather_provider = SimpleWeatherDataProvider(synth_weather, site_data)
        crop_start = cfg['AGROMANAGEMENT']['CROP_START_DATE'].replace(year=year);
        crop_end = cfg['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)
        agromanagement = [{crop_start: ParameterDict({'CropCalendar': ParameterDict(
            {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
             'crop_end_type': 'harvest', 'max_duration': cfg['AGROMANAGEMENT']['MAX_DURATION']}), 'TimedEvents': None,
                                                      'StateEvents': None})}]
        model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement);
        model.run_till_terminate();
        output = model.get_output()
        return output[-1]['TWSO'] if output else np.nan
    except Exception as e:
        member_id = member_row.get('member', 'N/A');
        logging.error(f"[FORECAST_WORKER] Error for dist {district_no}, member {member_id}: {e}")
        return np.nan


def run_forecast_simulation(df_static, df_daily_hist, df_seas5, cropdata, year, cfg):
    logging.info("=" * 70 + f"\n[FORECAST] Running Parallel Forecast Simulation for {year}\n" + "=" * 70)
    wg = WeatherGenerator();
    wg.fit(df_daily_hist)
    forecast_results = []
    district_params = {row['district_no']: _create_district_specific_parameters(row, cropdata) for _, row in
                       df_static.iterrows()}
    for district_no, group in tqdm(df_seas5.groupby('district_no'), desc="Forecast Sim (Districts)"):
        if district_no not in district_params: continue
        parameters, site_data = district_params[district_no]
        tasks = [delayed(_run_single_forecast_member)(member_row, district_no, year, wg, parameters, site_data, cfg) for
                 _, member_row in group.iterrows()]
        ensemble_yields = Parallel(n_jobs=-1, backend='loky')(tasks)
        valid_yields = [y for y in ensemble_yields if not np.isnan(y)]
        forecast_mean = np.mean(valid_yields) if valid_yields else np.nan;
        forecast_std = np.std(valid_yields) if valid_yields else np.nan
        forecast_results.append(
            {'year': year, 'district_no': district_no, 'lintul_yield_forecast_weather': forecast_mean,
             'forecast_uncertainty_std': forecast_std})
    logging.info(f"[FORECAST] Completed {len(forecast_results)} district ensembles\n")
    return pd.DataFrame(forecast_results)


def analyze_and_plot_results(df_hist, df_fcst, year, cfg):
    logging.info("=" * 70 + "\n[ANALYSIS] Analyzing Results\n" + "=" * 70)
    df_final = pd.merge(df_hist, df_fcst, on=['year', 'district_no']).dropna()
    if df_final.empty: logging.error("[ANALYSIS] No valid results to analyze!"); return
    dmc = cfg['CONSTANTS']['DMC_SUGARBEET']
    df_final['actual_yield_dry_kgha'] = df_final['actual_yield'] * 100.0 * dmc
    df_final = df_final.rename(columns={'lintul_yield_perfect_weather': 'perfect_yield_dry_kgha',
                                        'lintul_yield_forecast_weather': 'forecast_yield_dry_kgha'})
    for col in ['actual_yield_dry_kgha', 'perfect_yield_dry_kgha', 'forecast_yield_dry_kgha']: df_final[
        col.replace('_dry_kgha', '_dt')] = df_final[col] / 100.0
    output_path = os.path.join(cfg['FILE_PATHS']['OUTPUT_DIR'], f'final_comparison_{year}_TEST.csv')
    df_final.to_csv(output_path, index=False);
    logging.info(f"[ANALYSIS] ✓ Results saved to {output_path}")
    print("\n--- Performance Metrics (Dry Weight dt/ha) ---")
    mae_p = mean_absolute_error(df_final['actual_yield_dt'], df_final['perfect_yield_dt']);
    r2_p = r2_score(df_final['actual_yield_dt'], df_final['perfect_yield_dt'])
    mae_f = mean_absolute_error(df_final['actual_yield_dt'], df_final['forecast_yield_dt']);
    r2_f = r2_score(df_final['actual_yield_dt'], df_final['forecast_yield_dt'])
    print(f"  Perfect Weather:  MAE = {mae_p:.2f}, R² = {r2_p:.3f}");
    print(f"  Forecast Weather: MAE = {mae_f:.2f}, R² = {r2_f:.3f}\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6));
    fig.suptitle(f'WOFOST Performance Comparison - {year} (Dry Weight)', fontsize=14, fontweight='bold')
    min_val = df_final[['actual_yield_dt', 'perfect_yield_dt', 'forecast_yield_dt']].min().min() * 0.9
    max_val = df_final[['actual_yield_dt', 'perfect_yield_dt', 'forecast_yield_dt']].max().max() * 1.1
    axes[0].scatter(df_final['actual_yield_dt'], df_final['perfect_yield_dt'], alpha=0.6);
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    axes[0].set_title(f'Perfect Weather\nMAE={mae_p:.2f}, R²={r2_p:.3f}');
    axes[0].set_xlabel('Actual Yield (dt/ha)');
    axes[0].set_ylabel('Simulated Yield (dt/ha)')
    axes[1].scatter(df_final['actual_yield_dt'], df_final['forecast_yield_dt'], alpha=0.6, color='orange');
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    axes[1].set_title(f'Forecast Weather\nMAE={mae_f:.2f}, R²={r2_f:.3f}');
    axes[1].set_xlabel('Actual Yield (dt/ha)');
    axes[1].set_ylabel('Simulated Yield (dt/ha)')
    for ax in axes: ax.set_xlim(min_val, max_val); ax.set_ylim(min_val, max_val); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout();
    plot_path = os.path.join(cfg['FILE_PATHS']['OUTPUT_DIR'], f'results_{year}.png');
    plt.savefig(plot_path, dpi=300);
    logging.info(f"[ANALYSIS] ✓ Plot saved to {plot_path}");
    plt.show()


if __name__ == "__main__":
    test_year = CONFIG['TEST_YEAR']
    os.makedirs(CONFIG['FILE_PATHS']['OUTPUT_DIR'], exist_ok=True)
    logging.info("=" * 70 + f"\nWOFOST ONE-YEAR TEST PIPELINE - {test_year}\n" + "=" * 70)
    df_static, df_daily_hist, df_seas5, cropdata = load_data_and_setup_model(test_year, CONFIG)
    if cropdata is not None:
        df_hist = run_historical_simulation(df_static, df_daily_hist, cropdata, test_year, CONFIG)
        df_fcst = run_forecast_simulation(df_static, df_daily_hist, df_seas5, cropdata, test_year, CONFIG)
        if not df_hist.empty and not df_fcst.empty:
            analyze_and_plot_results(df_hist, df_fcst, test_year, CONFIG)
            logging.info("\n" + "=" * 70 + "\n✓ PIPELINE COMPLETED SUCCESSFULLY!\n" + "=" * 70)
        else:
            logging.error("Simulations failed - no results to analyze")
    else:
        logging.error("Pipeline aborted - data loading or setup failed")