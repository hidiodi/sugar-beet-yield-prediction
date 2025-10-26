# File: run_wofost_pipeline.py
# Description: A consolidated script to run the full multi-year pipeline.
#
# REFACTORED & PARALLELIZED VERSION v17: Corrected multi-year implementation.
# This version now correctly loads one historical weather file per year inside a
# loop, as per the specified data structure.

import datetime
import yaml
import pandas as pd
import numpy as np
import os
import logging
import sys

import geopandas as gpd
from scipy.stats import gamma

from pcse.util import penman_monteith
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider, WeatherDataProvider

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
    # --- MODIFIED: Define the multi-year (max 1981-2024) range for evaluation ---
    'START_YEAR': 1981,
    'END_YEAR': 2024,
    'DISTRICT_LIMIT': None,  # Set to None for a full run across all districts

    'FILE_PATHS': {
        # --- MODIFIED: Path to the DIRECTORY containing yearly weather files ---
        'HISTORICAL_DAILY_WEATHER_DIR': 'data/02_intermediate/daily_weather',
        'STATIC_FEATURES': 'data/05_model_input/stage1_preseason_features.csv',
        'SEAS5_MEMBER_FEATURES': 'data/02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv',
        'CROP_YAML': 'data/01_raw/sugarbeet.yaml',
        'OUTPUT_DIR': 'data/06_model_output/multi_year_final',
        'DISTRICTS_GEOJSON': 'data/01_raw/districts_official.geojson',
    },
    'WEATHER_DEFAULTS': {'WIND_SPEED': 2.0, 'VAPOR_PRESSURE': 1.0},
    'WEATHER_GENERATOR': {'PRECIP_THRESHOLD_MM': 0.3, 'MIN_SRAD': 100.0},
    'AGROMANAGEMENT': {
        'CROP_START_DATE': datetime.date(2018, 3, 15),
        'CROP_END_DATE': datetime.date(2018, 10, 20),
        'MAX_DURATION': 250,
    },
    'CONSTANTS': {
        'DMC_SUGARBEET': 0.25, 'INITIAL_ROOTING_DEPTH_CM': 10.0,
        'SOIL_PARTICLE_DENSITY': 2.65,
    },
    'SOIL_COLUMN_MAPPING': {
        'sand': 'avg_sand_0_100cm', 'clay': 'avg_clay_0_100cm',
        'som': 'avg_som_0_100cm', 'bdod': 'avg_bdod_0_100cm',
    },
    'SOIL_DEFAULTS_AND_CONSTANTS': {'RDMSOL': 150.0, 'KSUB': 10.0, 'SOPE': 10.0},
    'GENERIC_SITE': {'LATITUDE': 52.0, 'LONGITUDE': 10.0, 'ELEVATION': 50.0}
}

# ==============================================================================
# === S C R I P T   S T A R T S   H E R E ===
# ==============================================================================

# --- UNCHANGED SECTION: Core Classes and Functions ---
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
        self.latitude = site_data['LAT'];
        self.longitude = site_data['LON'];
        self.elevation = site_data['ELEV']
        self.angstA = 0.25;
        self.angstB = 0.5
        weather_df = weather_df.copy();
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        self.store = {}
        fallback_wind = CONFIG['WEATHER_DEFAULTS']['WIND_SPEED']
        fallback_vap_kpa = CONFIG['WEATHER_DEFAULTS']['VAPOR_PRESSURE']
        for _, row in weather_df.iterrows():
            try:
                day = row['date'].date();
                tmin = float(row['tmin']);
                tmax = float(row['tmax'])
                wind = float(row.get('wind', fallback_wind));
                vap_kpa = float(row.get('vap', fallback_vap_kpa))
                irrad_j_m2_day = float(row['srad']) * 10.0
                irrad_kj_m2_day = irrad_j_m2_day / 1000.0
                precip_mm = float(row['precip']) / 100000.0
                precip_cm = precip_mm / 10.0
                vap_hpa = vap_kpa * 10.0
                et0_mm = penman_monteith(day, self.latitude, self.elevation, tmin, tmax, irrad_kj_m2_day, vap_hpa, wind)
                et0_cm = et0_mm / 10.0
                self.store[(day, 0)] = ParameterDict(
                    {'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_cm,
                     'IRRAD': irrad_j_m2_day, 'VAP': vap_hpa, 'WIND': wind, 'E0': et0_cm,
                     'ES0': et0_cm, 'ET0': et0_cm, 'SNOWDEPTH': 0.0})
            except Exception as e:
                logging.error(f"CRITICAL: Failed processing weather row for date: {row.get('date')}. Error: {e}");
                raise e


class WeatherGenerator:
    def __init__(self):
        self.stats = defaultdict(dict)
        self.PRECIP_THRESHOLD_MM = CONFIG['WEATHER_GENERATOR']['PRECIP_THRESHOLD_MM']
        self.MIN_SRAD = CONFIG['WEATHER_GENERATOR']['MIN_SRAD']

    def fit(self, daily_df: pd.DataFrame):
        logging.info("[WEATHER_GEN] Fitting Weather Generator...")
        daily_df = daily_df.copy()
        daily_df['district_no'] = daily_df['district_no'].astype(str).str.zfill(5)
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
                'srad_mean': group['srad'].mean(), 'srad_std': max(group['srad'].std(), 0.5)
            }

    def generate(self, district_no: str, start_date_str: str, end_date_str: str, monthly_anomalies: dict):
        dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D');
        generated_data = []
        yesterday_was_wet = np.random.rand() < 0.5
        for date in dates:
            month, key = date.month, (str(district_no).zfill(5), date.month)
            if key not in self.stats: continue
            month_stats = self.stats[key]
            transition_prob = month_stats['p_wet_given_wet'] if yesterday_was_wet else month_stats['p_wet_given_dry']
            today_is_wet = np.random.rand() < transition_prob
            precip = 0.0
            if today_is_wet:
                alpha = month_stats['precip_gamma_shape'];
                beta = month_stats['precip_gamma_scale']
                precip = max(0, gamma.rvs(a=alpha, scale=beta, size=1)[0])
            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std']);
            tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin: tmax = tmin + abs(np.random.normal(0, 1.0))
            srad = max(self.MIN_SRAD, np.random.normal(month_stats['srad_mean'], month_stats['srad_std']))
            generated_data.append({'date': date, 'tmin': tmin, 'tmax': tmax, 'precip': precip, 'srad': srad})
            yesterday_was_wet = today_is_wet
        if not generated_data: return pd.DataFrame()
        synthetic_df = pd.DataFrame(generated_data);
        synthetic_df['month'] = synthetic_df['date'].dt.month
        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month;
            key = (str(district_no).zfill(5), month)
            if key not in self.stats: continue
            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0)
            hist_tmean = (self.stats[key]['tmin_mean'] + self.stats[key]['tmax_mean']) / 2
            synth_tmean = (synthetic_df.loc[month_mask, 'tmin'].mean() + synthetic_df.loc[
                month_mask, 'tmax'].mean()) / 2
            temp_correction = (hist_tmean + temp_anomaly) - synth_tmean
            synthetic_df.loc[month_mask, ['tmin', 'tmax']] += temp_correction
            precip_anomaly_factor = 1.0 + monthly_anomalies.get(f'precip_anomaly_{month}', 0)
            target_precip = self.stats[key].get('precip_mean', 0) * month_mask.sum() * precip_anomaly_factor
            synth_precip = synthetic_df.loc[month_mask, 'precip'].sum()
            if synth_precip > 0: synthetic_df.loc[month_mask, 'precip'] *= (target_precip / synth_precip)
        return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad']]


def _calculate_soil_hydraulic_properties(sand_frac, clay_frac, som_frac, bdod):
    c_wp_1 = -0.024 * sand_frac + 0.487 * clay_frac + 0.006 * som_frac + 0.005 * (sand_frac * som_frac) - 0.013 * (
                clay_frac * som_frac) + 0.068 * (sand_frac * clay_frac) + 0.031
    pwp1500 = c_wp_1 + (0.14 * c_wp_1 - 0.02)
    c_fc_1 = -0.251 * sand_frac + 0.195 * clay_frac + 0.011 * som_frac + 0.006 * (sand_frac * som_frac) - 0.027 * (
                clay_frac * som_frac) + 0.452 * (sand_frac * clay_frac) + 0.299
    fc33 = c_fc_1 + (1.283 * c_fc_1 ** 2 - 0.374 * c_fc_1 - 0.015)
    porosity = 1 - (bdod / CONFIG['CONSTANTS']['SOIL_PARTICLE_DENSITY'])
    c_sat_1 = (fc33 - 0.1) * 1.55 + 0.1
    fc33_porosity_adj = fc33 + 0.009 * (sand_frac * 100) * (porosity - c_sat_1)
    sm0 = porosity;
    smw = max(0.01, pwp1500)
    smfcf = min(max(smw + 0.01, fc33_porosity_adj), sm0 - 0.01)
    crairc = max(0.01, sm0 - smfcf)
    return {'SMW': smw, 'SMFCF': smfcf, 'SM0': sm0, 'CRAIRC': crairc}


def _create_district_specific_parameters(static_row, cropdata):
    sitedata = ParameterDict()
    latitude = static_row.get('latitude', CONFIG['GENERIC_SITE']['LATITUDE']);
    longitude = static_row.get('longitude', CONFIG['GENERIC_SITE']['LONGITUDE'])
    elevation = static_row.get('avg_elevation', CONFIG['GENERIC_SITE']['ELEVATION'])
    sitedata.add_variable('LAT', latitude);
    sitedata.add_variable('LON', longitude);
    sitedata.add_variable('ELEV', elevation)
    soildata = ParameterDict()
    soil_map = CONFIG['SOIL_COLUMN_MAPPING']
    try:
        sand = static_row[soil_map['sand']] / 100.0;
        clay = static_row[soil_map['clay']] / 100.0
        som = static_row[soil_map['som']] / 100.0;
        bdod = static_row[soil_map['bdod']]
        calculated_soil_params = _calculate_soil_hydraulic_properties(sand, clay, som, bdod)
        for key, value in calculated_soil_params.items():
            soildata.add_variable(key, value)
    except (KeyError, TypeError) as e:
        logging.error(f"FATAL: Missing soil data for district {static_row.get('district_no', 'N/A')}. Error: {e}");
        raise e
    for key, value in CONFIG['SOIL_DEFAULTS_AND_CONSTANTS'].items():
        soildata.add_variable(key, value)
    smfc = soildata['SMFCF'];
    smw = soildata['SMW'];
    rdi = CONFIG['CONSTANTS']['INITIAL_ROOTING_DEPTH_CM']
    rdmsol = soildata['RDMSOL'];
    smlim = smfc;
    wav = (smfc - smw) * rdmsol
    sitedata.add_variable('SMLIM', smlim);
    sitedata.add_variable('WAV', wav)
    sitedata.add_variable('IFUNRN', 0.0);
    sitedata.add_variable('NOTINF', 0.0);
    sitedata.add_variable('SSI', 0.0);
    sitedata.add_variable('SSMAX', 0.0)
    return ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata), sitedata


def run_historical_simulation(df_static_year, df_daily_hist_year, cropdata, year, cfg):
    logging.info(f"--- Running Historical Simulation for {year} ---")
    results = []
    for _, row in tqdm(df_static_year.iterrows(), total=len(df_static_year), desc=f"Historical Sim {year}"):
        district_no = row['district_no']
        weather_df = df_daily_hist_year[df_daily_hist_year['district_no'] == district_no].copy()
        if weather_df.empty: continue
        try:
            parameters, site_data = _create_district_specific_parameters(row, cropdata)
            weather_provider = SimpleWeatherDataProvider(weather_df, site_data)
            crop_start = cfg['AGROMANAGEMENT']['CROP_START_DATE'].replace(year=year);
            crop_end = cfg['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)
            agromanagement = [{crop_start: ParameterDict({'CropCalendar': ParameterDict(
                {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
                 'crop_end_type': 'harvest', 'max_duration': cfg['AGROMANAGEMENT']['MAX_DURATION']}),
                'TimedEvents': None, 'StateEvents': None})}]
            model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement);
            model.run_till_terminate()
            output = model.get_output();
            simulated_yield = output[-1]['TWSO'] if output else np.nan
        except Exception as e:
            logging.error(f"[HISTORICAL] ERROR for district {district_no} in {year}: {e}", exc_info=True);
            simulated_yield = np.nan
        results.append({'year': year, 'district_no': district_no, 'actual_yield': row['kreisYield'],
                        'lintul_yield_perfect_weather': simulated_yield})
    return pd.DataFrame(results)


def _run_single_forecast_member(member_row, district_no, year, wg, parameters, site_data, cfg):
    try:
        monthly_anomalies = {}
        for month in range(3, 11):
            temp_col = f'temp_anomaly_forecast_{month}';
            precip_col = f'precip_anomaly_forecast_{month}'
            monthly_anomalies[f'temp_anomaly_{month}'] = member_row.get(temp_col, 0)
            monthly_anomalies[f'precip_anomaly_{month}'] = member_row.get(precip_col, 0)
        start_date, end_date = f'{year}-03-01', f'{year}-10-31'
        synth_weather = wg.generate(district_no, start_date, end_date, monthly_anomalies)
        if synth_weather.empty: return np.nan
        weather_provider = SimpleWeatherDataProvider(synth_weather, site_data)
        crop_start = cfg['AGROMANAGEMENT']['CROP_START_DATE'].replace(year=year);
        crop_end = cfg['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)
        agromanagement = [{crop_start: ParameterDict({'CropCalendar': ParameterDict(
            {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
             'crop_end_type': 'harvest', 'max_duration': cfg['AGROMANAGEMENT']['MAX_DURATION']}),
            'TimedEvents': None, 'StateEvents': None})}]
        model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement);
        model.run_till_terminate()
        return model.get_output()[-1]['TWSO'] if model.get_output() else np.nan
    except Exception as e:
        logging.error(f"[FORECAST_WORKER] Error for dist {district_no}, member {member_row.get('member', 'N/A')}: {e}");
        return np.nan


def run_forecast_simulation(df_static_year, df_seas5_year, wg, cropdata, year, cfg):
    logging.info(f"--- Running Parallel Forecast Simulation for {year} ---")
    forecast_results = []
    # Disable logging during parameter creation to avoid console spam
    logging.disable(logging.INFO)
    district_params = {row['district_no']: _create_district_specific_parameters(row, cropdata) for _, row in
                       df_static_year.iterrows()}
    logging.disable(logging.NOTSET)
    for district_no, group in tqdm(df_seas5_year.groupby('district_no'), desc=f"Forecast Sim {year}"):
        if district_no not in district_params: continue
        parameters, site_data = district_params[district_no]
        tasks = [delayed(_run_single_forecast_member)(member_row, district_no, year, wg, parameters, site_data, cfg) for
                 _, member_row in group.iterrows()]
        ensemble_yields = Parallel(n_jobs=-1, backend='loky')(tasks)
        valid_yields = [y for y in ensemble_yields if not np.isnan(y)]
        forecast_results.append({'year': year, 'district_no': district_no,
                                 'lintul_yield_forecast_weather': np.mean(valid_yields) if valid_yields else np.nan,
                                 'forecast_uncertainty_std': np.std(valid_yields) if valid_yields else np.nan})
    return pd.DataFrame(forecast_results)


def analyze_and_plot_results(df_hist, df_fcst, output_dir, start_year, end_year):
    logging.info("=" * 70 + "\n[ANALYSIS] Analyzing Final Multi-Year Results\n" + "=" * 70)
    df_final = pd.merge(df_hist, df_fcst, on=['year', 'district_no']).dropna()
    if df_final.empty: logging.error("[ANALYSIS] No valid merged results to analyze!"); return
    dmc = CONFIG['CONSTANTS']['DMC_SUGARBEET']
    df_final['actual_yield_dry_kgha'] = df_final['actual_yield'] * 100.0 * dmc
    df_final = df_final.rename(columns={'lintul_yield_perfect_weather': 'perfect_yield_dry_kgha',
                                        'lintul_yield_forecast_weather': 'forecast_yield_dry_kgha'})
    for col in ['actual_yield_dry_kgha', 'perfect_yield_dry_kgha', 'forecast_yield_dry_kgha']:
        df_final[col.replace('_dry_kgha', '_dt')] = df_final[col] / 100.0
    output_path = os.path.join(output_dir, f'final_comparison_{start_year}-{end_year}.csv')
    df_final.to_csv(output_path, index=False);
    logging.info(f"[ANALYSIS] ✓ Multi-year results saved to {output_path}")
    print("\n--- Overall Performance Metrics (Dry Weight dt/ha) ---")
    mae_p = mean_absolute_error(df_final['actual_yield_dt'], df_final['perfect_yield_dt']);
    r2_p = r2_score(df_final['actual_yield_dt'], df_final['perfect_yield_dt'])
    mae_f = mean_absolute_error(df_final['actual_yield_dt'], df_final['forecast_yield_dt']);
    r2_f = r2_score(df_final['actual_yield_dt'], df_final['forecast_yield_dt'])
    print(f"  Perfect Weather:  MAE = {mae_p:.2f}, R² = {r2_p:.3f}");
    print(f"  Forecast Weather: MAE = {mae_f:.2f}, R² = {r2_f:.3f}\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6));
    fig.suptitle(f'WOFOST Performance ({start_year}-{end_year})', fontsize=16)
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
    plot_path = os.path.join(output_dir, f'results_scatter_{start_year}-{end_year}.png');
    plt.savefig(plot_path, dpi=300)
    logging.info(f"[ANALYSIS] ✓ Plot saved to {plot_path}");
    plt.show()


if __name__ == "__main__":
    os.makedirs(CONFIG['FILE_PATHS']['OUTPUT_DIR'], exist_ok=True)

    # 1. Load and Synchronize Multi-Year Data Sources
    logging.info("=" * 70 + "\nLoading and synchronizing multi-year data sources...\n" + "=" * 70)
    try:
        df_static_all = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_FEATURES'])
        df_seas5_all = pd.read_csv(CONFIG['FILE_PATHS']['SEAS5_MEMBER_FEATURES'])

        # Convert district_no to a consistent string format in both dataframes first
        for df in [df_static_all, df_seas5_all]:
            df['district_no'] = pd.to_numeric(df['district_no'], errors='coerce').astype('Int64').astype(str).str.zfill(
                5)

        # --- CRITICAL FILTERING STEP ---
        initial_rows = len(df_static_all)
        # Drop rows from the static data where 'kreisYield' is missing (NaN) or zero.
        df_static_all.dropna(subset=['kreisYield'], inplace=True)
        df_static_all = df_static_all[df_static_all['kreisYield'] > 0]
        rows_removed = initial_rows - len(df_static_all)
        logging.info(f"Filtered static data: Removed {rows_removed} rows with no valid historical yield.")

        # --- NEW SYNCHRONIZATION LOGIC ---
        # Create a "key" dataframe of the valid year/district combinations from the filtered static data
        valid_combinations = df_static_all[['year', 'district_no']].drop_duplicates()

        # Use an 'inner' merge to filter df_seas5_all, keeping only the rows that match
        # the valid year/district combinations. This synchronizes the two dataframes.
        initial_seas5_rows = len(df_seas5_all)
        df_seas5_all = pd.merge(df_seas5_all, valid_combinations, on=['year', 'district_no'], how='inner')
        seas5_rows_removed = initial_seas5_rows - len(df_seas5_all)

        logging.info(f"Synchronized forecast data: Removed {seas5_rows_removed} rows to match valid districts.")
        logging.info(f"Proceeding with {len(df_static_all)} valid district-year records across all data sources.")
        # --- END OF NEW LOGIC ---

    except FileNotFoundError as e:
        logging.error(f"FATAL: A required multi-year data file was not found. Error: {e}");
        sys.exit()

    # 2. Fit the WeatherGenerator ONCE using all available historical data for maximum learning
    wg = WeatherGenerator()
    all_hist_dfs = []
    logging.info("Loading all available yearly weather files to fit the Weather Generator...")
    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        hist_weather_path = os.path.join(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'],
                                         f"historical_daily_weather_era5_{year}.csv")
        if os.path.exists(hist_weather_path):
            all_hist_dfs.append(pd.read_csv(hist_weather_path, parse_dates=['date']))
        else:
            logging.warning(f"Could not find historical weather file for fitting: {hist_weather_path}")

    if not all_hist_dfs:
        logging.error("FATAL: No historical weather files found to fit the WeatherGenerator. Aborting.");
        sys.exit()

    full_hist_df_for_fitting = pd.concat(all_hist_dfs, ignore_index=True)
    wg.fit(full_hist_df_for_fitting)

    # 3. Load crop parameters ONCE
    with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
        crop_params = yaml.safe_load(f)['CropParameters']
    variety_name = [k for k in crop_params.keys() if not k.startswith('Generic')][0]
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

    # 4. Main loop through years to run simulations
    all_hist_results = []
    all_fcst_results = []
    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        logging.info("=" * 70 + f"\nPROCESSING YEAR: {year}\n" + "=" * 70)

        # Define the path for this year's specific weather file
        hist_weather_path = os.path.join(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'],
                                         f"historical_daily_weather_era5_{year}.csv")
        try:
            df_daily_hist_year = pd.read_csv(hist_weather_path, parse_dates=['date'])
            df_daily_hist_year['district_no'] = df_daily_hist_year['district_no'].astype(str).str.zfill(5)
        except FileNotFoundError:
            logging.error(
                f"FATAL: The required weather file for {year} was not found at {hist_weather_path}. Skipping year.");
            continue

        # Filter the master static/forecast dataframes for the current year
        df_static_year = df_static_all[df_static_all['year'] == year].copy()
        df_seas5_year = df_seas5_all[df_seas5_all['year'] == year].copy()

        if CONFIG['DISTRICT_LIMIT'] is not None:
            # Get a unique list of the first N districts
            limited_districts = df_static_year['district_no'].unique()[:CONFIG['DISTRICT_LIMIT']]

            # Filter both dataframes to only include those districts
            df_static_year = df_static_year[df_static_year['district_no'].isin(limited_districts)]
            df_seas5_year = df_seas5_year[df_seas5_year['district_no'].isin(limited_districts)]

            logging.info(f"Applying DISTRICT_LIMIT: Running simulations for only {len(limited_districts)} districts.")

        if df_static_year.empty or df_seas5_year.empty:
            logging.warning(f"Missing static or forecast data for year {year}. Skipping.");
            continue

        # Run simulations
        df_hist = run_historical_simulation(df_static_year, df_daily_hist_year, cropdata, year, CONFIG)
        df_fcst = run_forecast_simulation(df_static_year, df_seas5_year, wg, cropdata, year, CONFIG)

        if not df_hist.empty: all_hist_results.append(df_hist)
        if not df_fcst.empty: all_fcst_results.append(df_fcst)

    # 5. Perform the final, consolidated analysis
    if all_hist_results and all_fcst_results:
        final_hist_df = pd.concat(all_hist_results, ignore_index=True)
        final_fcst_df = pd.concat(all_fcst_results, ignore_index=True)
        analyze_and_plot_results(final_hist_df, final_fcst_df, CONFIG['FILE_PATHS']['OUTPUT_DIR'], CONFIG['START_YEAR'],
                                 CONFIG['END_YEAR'])
        logging.info("\n" + "=" * 70 + "\n✓ MULTI-YEAR PIPELINE COMPLETED SUCCESSFULLY!\n" + "=" * 70)
    else:
        logging.error("No simulation results were generated across all years. Aborting final analysis.")

    logging.shutdown()