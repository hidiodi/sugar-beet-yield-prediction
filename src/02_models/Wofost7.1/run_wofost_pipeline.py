# File: run_wofost_pipeline.py
# V3 REFACTORED: This script is now a pure "consumer" of pre-calculated data assets.
# All data preparation (weather generation, initial conditions, genetic parameters)
# has been moved to dedicated build scripts.

import datetime
import json
import pandas as pd
import numpy as np
import os
import logging
import sys
from pathlib import Path
from pcse.models import Wofost72_WLP_FD, Wofost72_PP
from pcse.base import ParameterProvider, WeatherDataProvider
from pcse.util import penman_monteith
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# Use the WOFOST_CONFIG dictionary from the central config file
CONFIG = config.WOFOST_CONFIG
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR

# ==============================================================================
# === SCRIPT STARTS HERE ===
# ==============================================================================
logging.getLogger().handlers = []
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)


class ParameterDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    def __setattr__(self, name, value):
        self[name] = value

class SimpleWeatherDataProvider(WeatherDataProvider):
    def __init__(self, weather_df, site_data):
        super().__init__()
        self.latitude = site_data['LAT']
        self.elevation = site_data['ELEV']
        weather_df = weather_df.copy()
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        self.store = {}
        for _, row in weather_df.iterrows():
            day = row['date'].date()
            tmin, tmax = float(row['tmin']), float(row['tmax'])
            precip_cm = float(row['precip']) / 10.0
            srad_mj_m2_day = float(row['srad'])
            irrad_kj_m2_day = srad_mj_m2_day * 1_000.0
            vap_kpa = float(row.get('vap', CONFIG['WEATHER_DEFAULTS']['VAPOR_PRESSURE']))
            vap_hpa = vap_kpa * 10.0
            wind = float(row.get('wind', CONFIG['WEATHER_DEFAULTS']['WIND_SPEED']))
            et0_mm = penman_monteith(day, self.latitude, self.elevation, tmin, tmax, irrad_kj_m2_day, vap_hpa, wind)
            self.store[(day, 0)] = ParameterDict({
                'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_cm,
                'IRRAD': srad_mj_m2_day * 1_000_000.0, 'VAP': vap_hpa, 'WIND': wind,
                'E0': et0_mm / 10.0, 'ES0': et0_mm / 10.0, 'ET0': et0_mm / 10.0, 'SNOWDEPTH': 0.0
            })

def _create_district_specific_parameters(static_site_row, crop_params_for_year, initial_condition_row):
    """
    V4 REFACTORED: Creates parameter providers using pre-loaded data rows.
    """
    sitedata = ParameterDict()
    soildata = ParameterDict()

    sitedata.LAT = static_site_row['latitude']
    sitedata.LON = static_site_row['longitude']
    sitedata.ELEV = static_site_row['avg_elevation']

    # Set DYNAMIC initial conditions from the pre-calculated file
    sitedata.WAV = initial_condition_row['WAV']

    # Static soil and site parameters
    for param in ['NOTINF', 'SSMAX', 'SMW', 'SMFCF', 'SM0', 'CRAIRC', 'K0', 'SOPE', 'KSUB', 'RDMSOL']:
        if param in ['NOTINF', 'SSMAX']:
            sitedata[param] = static_site_row[param]
        else:
            soildata[param] = static_site_row[param]

    sitedata.IFUNRN = 0.0
    sitedata.SSI = 0.0
    sitedata.SMLIM = soildata.SMFCF

    # Create a ParameterDict for crop parameters for this specific year
    cropdata = ParameterDict(crop_params_for_year)

    return ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata), sitedata

def _run_single_forecast_member(weather_for_member_df, crop_params_for_year, static_site_row, initial_condition_row):
    try:
        parameters, site_data = _create_district_specific_parameters(static_site_row, crop_params_for_year, initial_condition_row)
        weather_provider = SimpleWeatherDataProvider(weather_for_member_df, site_data)

        crop_start = pd.to_datetime(initial_condition_row['sowing_date']).date()
        crop_end = pd.to_datetime(initial_condition_row['CROP_END_DATE']).date().replace(year=crop_start.year)

        agromanagement = [{
            crop_start: ParameterDict({
                'CropCalendar': ParameterDict({
                    'crop_start_date': crop_start, 'crop_start_type': 'emergence',
                    'crop_end_date': crop_end, 'crop_end_type': 'harvest',
                    'max_duration': CONFIG['AGROMANAGEMENT']['MAX_DURATION']
                }), 'TimedEvents': None, 'StateEvents': None
            })
        }]

        model_wlp = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)
        model_wlp.run_till_terminate()
        output_wlp = pd.DataFrame(model_wlp.get_output()).set_index('day')
        yield_wlp = output_wlp.iloc[-1]['TWSO'] if not output_wlp.empty else 0

        model_pp = Wofost72_PP(parameters, weather_provider, agromanagement)
        model_pp.run_till_terminate()
        output_pp = pd.DataFrame(model_pp.get_output())
        yield_pp = output_pp.iloc[-1]['TWSO'] if not output_pp.empty else 0

        drought_stress_index = (yield_pp - yield_wlp) / yield_pp if yield_pp > 0 else 0.0

        def get_max_consecutive_run(boolean_series):
            if not boolean_series.any(): return 0
            runs = boolean_series.ne(boolean_series.shift()).cumsum()
            return boolean_series.groupby(runs).cumsum().max()

        summer_weather = weather_for_member_df[weather_for_member_df['date'].dt.month.isin([6, 7, 8])].copy()
        is_heatwave_day = summer_weather['tmax'] > 30
        consecutive_hot_days = get_max_consecutive_run(is_heatwave_day)
        is_dry_day = summer_weather['precip'] < 1
        consecutive_dry_days = get_max_consecutive_run(is_dry_day)

        days_to_anthesis = np.nan
        if 'DOA' in output_wlp.columns and (output_wlp['DOA'] is not None):
            first_anthesis_day = output_wlp[output_wlp['DOA'].notna()].index.min()
            if pd.notna(first_anthesis_day): days_to_anthesis = (first_anthesis_day - crop_start).days

        max_lai_achieved = output_wlp['LAI'].max() if 'LAI' in output_wlp.columns else 0.0
        cumulative_water_stress = (1 - output_wlp['TRA']).sum() if 'TRA' in output_wlp.columns else np.nan

        return {
            'yield_water_limited': yield_wlp, 'yield_potential': yield_pp,
            'consecutive_tmax_gt_30c': consecutive_hot_days,
            'consecutive_dry_days': consecutive_dry_days,
            'drought_stress_index': drought_stress_index,
            'simulation_failed': False, 'days_to_anthesis': days_to_anthesis,
            'max_lai_achieved': max_lai_achieved,
            'cumulative_water_stress': cumulative_water_stress
        }

    except Exception as e:
        logging.warning(f"Simulation failed for member: {e}")
        return {
            'yield_water_limited': np.nan, 'yield_potential': np.nan,
            'consecutive_tmax_gt_30c': np.nan, 'consecutive_dry_days': np.nan,
            'drought_stress_index': np.nan, 'simulation_failed': True,
            'days_to_anthesis': np.nan, 'max_lai_achieved': np.nan,
            'cumulative_water_stress': np.nan
        }

def run_forecast_simulation(df_static_year, df_forecast_weather_year, df_initial_conditions_year, all_crop_genes):
    full_ensemble_results = []
    crop_params_for_year = all_crop_genes[str(df_static_year['year'].iloc[0])]

    # Pre-build a dictionary for quick lookup
    static_site_map = df_static_year.set_index('district_no').to_dict('index')
    initial_conditions_map = df_initial_conditions_year.set_index('district_no').to_dict('index')

    # Group weather by simulation unit: district, year, and member
    grouped_weather = df_forecast_weather_year.groupby(['district_no', 'member'])

    tasks = []
    for (district_no, member), weather_for_member_df in tqdm(grouped_weather, desc=f"Preparing Forecast Jobs for {df_static_year['year'].iloc[0]}"):
        if district_no in static_site_map and district_no in initial_conditions_map:
            static_site_row = static_site_map[district_no]
            initial_condition_row = initial_conditions_map[district_no]

            tasks.append(delayed(_run_single_forecast_member)(
                weather_for_member_df, crop_params_for_year, static_site_row, initial_condition_row
            ))

    ensemble_outputs = Parallel(n_jobs=-1, backend='loky')(tasks)

    # Re-assemble results
    result_idx = 0
    for (district_no, member), _ in grouped_weather:
        if district_no in static_site_map and district_no in initial_conditions_map:
            result = ensemble_outputs[result_idx]
            full_ensemble_results.append({
                'year': df_static_year['year'].iloc[0], 'district_no': district_no, 'member': member,
                'yield_water_limited_dry_kgha': result['yield_water_limited'],
                'yield_potential_dry_kgha': result['yield_potential'],
                'drought_stress_index': result['drought_stress_index']
            })
            result_idx += 1

    return pd.DataFrame(full_ensemble_results)


def analyze_and_plot_ensemble_results(df_hist, df_fcst_ensemble, output_dir, start_year, end_year):
    """
    Analyzes and plots the results of the ensemble forecast.
    """
    logging.info("=" * 70 + "\n[ANALYSIS] Starting Final Analysis and Plotting\n" + "=" * 70)
    dmc = CONFIG['CONSTANTS']['DMC_SUGARBEET']
    if not df_fcst_ensemble.empty:
        fcst_output_path = output_dir / f'forecast_ensemble_{start_year}-{end_year}.csv'
        df_fcst_ensemble.to_csv(fcst_output_path, index=False)
        logging.info(f"Full forecast ensemble results saved to {fcst_output_path}")

    if df_fcst_ensemble.empty or df_hist.empty:
        logging.error("[ANALYSIS] No data in forecast or historical results. Cannot analyze.")
        return

    df_fcst_ensemble['yield_wlp_fresh_dt'] = (df_fcst_ensemble['yield_water_limited_dry_kgha'] / dmc) / 100.0
    df_hist['perfect_yield_dt'] = (df_hist['lintul_yield_perfect_weather'] / dmc) / 100.0
    df_fcst_agg = df_fcst_ensemble.groupby(['year', 'district_no']).agg(
        forecast_yield_mean=('yield_wlp_fresh_dt', 'mean'),
        sim_failure_rate=('simulation_failed', 'mean')
    ).reset_index()
    df_final = pd.merge(df_hist[['year', 'district_no', 'actual_yield', 'perfect_yield_dt']], df_fcst_agg, on=['year', 'district_no'])
    df_final_clean = df_final.dropna(subset=['actual_yield', 'perfect_yield_dt', 'forecast_yield_mean'])

    if df_final_clean.empty:
        logging.error("[ANALYSIS] No valid, non-NaN merged results were found.")
        return

    mae_p = mean_absolute_error(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt'])
    r2_p = r2_score(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt'])
    mae_f = mean_absolute_error(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'])
    r2_f = r2_score(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'])

    print(f"\n--- Overall Performance ---\nPerfect Weather: MAE={mae_p:.2f}, R²={r2_p:.3f}\nForecast Weather: MAE={mae_f:.2f}, R²={r2_f:.3f}\n")

def aggregate_and_save_extreme_weather_metrics(df_fcst_ensemble, output_path):
    """
    Calculates and saves distributional features for extreme weather and drought stress metrics.
    """
    logging.info("=" * 70 + "\n[ANALYSIS] Aggregating in-season risk features...\n" + "=" * 70)
    if df_fcst_ensemble.empty:
        logging.warning("[ANALYSIS] Forecast ensemble dataframe is empty. Skipping extreme metrics.")
        return
    aggs = {
        'consecutive_tmax_gt_30c': ['mean', 'std', lambda x: x.quantile(0.90)],
        'consecutive_dry_days': ['mean', 'std', lambda x: x.quantile(0.90)],
        'drought_stress_index': ['mean', 'std', lambda x: x.quantile(0.90)],
        'simulation_failed': ['mean'],
        'days_to_anthesis': ['mean', 'std'],
        'max_lai_achieved': ['mean', 'std', lambda x: x.quantile(0.10)],
        'cumulative_water_stress': ['mean', 'std', lambda x: x.quantile(0.90)]
    }
    df_extreme_metrics = df_fcst_ensemble.groupby(['year', 'district_no']).agg(aggs).reset_index()
    df_extreme_metrics.columns = ['_'.join(col).strip() for col in df_extreme_metrics.columns.values]
    df_extreme_metrics.to_csv(output_path, index=False)
    logging.info(f"[ANALYSIS] Risk features saved to {output_path}")

def run_historical_simulation(df_static_year, df_historical_weather_year, df_initial_conditions_year, all_crop_genes):
    results = []
    year = df_static_year['year'].iloc[0]
    crop_params_for_year = all_crop_genes[str(year)]

    for _, static_site_row in tqdm(df_static_year.iterrows(), total=len(df_static_year), desc=f"Historical Sim {year}"):
        district_no = static_site_row['district_no']
        weather_df = df_historical_weather_year[df_historical_weather_year['district_no'] == district_no]
        if weather_df.empty: continue

        try:
            initial_condition_row = df_initial_conditions_year[df_initial_conditions_year['district_no'] == district_no].iloc[0]
            parameters, site_data = _create_district_specific_parameters(static_site_row, crop_params_for_year, initial_condition_row)
            weather_provider = SimpleWeatherDataProvider(weather_df, site_data)

            crop_start = pd.to_datetime(initial_condition_row['sowing_date']).date()
            crop_end = pd.to_datetime(initial_condition_row['CROP_END_DATE']).date().replace(year=year)

            agromanagement = [{
                crop_start: ParameterDict({'CropCalendar': ParameterDict({'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end, 'crop_end_type': 'harvest', 'max_duration': CONFIG['AGROMANAGEMENT']['MAX_DURATION']}), 'TimedEvents': None, 'StateEvents': None})
            }]

            model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)
            model.run_till_terminate()
            simulated_yield = model.get_output()[-1]['TWSO'] if model.get_output() else np.nan
        except Exception as e:
            logging.error(f"[HISTORICAL] ERROR for district {district_no} in {year}: {e}", exc_info=True)
            simulated_yield = np.nan

        results.append({'year': year, 'district_no': district_no, 'actual_yield': static_site_row['kreisYield'], 'lintul_yield_perfect_weather': simulated_yield})
    return pd.DataFrame(results)


if __name__ == "__main__":
    # --- 1. SETUP LOGGING & PATHS ---
    logging.info("=" * 70 + "\nStarting Refactored WOFOST Pipeline (V3 Consumer)\n" + "=" * 70)
    os.makedirs(CONFIG['FILE_PATHS']['OUTPUT_DIR'], exist_ok=True)

    # --- 2. LOAD ALL PRE-CALCULATED DATA ASSETS ---
    logging.info("Loading all pre-calculated data assets...")
    try:
        df_static_all = pd.read_csv(PROCESSED_DATA_DIR / 'StaticSiteData.csv', dtype={'district_no': str})
        df_initial_conditions = pd.read_csv(PROCESSED_DATA_DIR / 'InitialConditions.csv', dtype={'district_no': str}, parse_dates=['sowing_date', 'CROP_END_DATE'])
        df_forecast_weather = pd.read_csv(PROCESSED_DATA_DIR / 'ForecastedWeather.csv', dtype={'district_no': str}, parse_dates=['date'])
        df_historical_weather = pd.read_csv(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'], dtype={'district_no': str}, parse_dates=['date'])

        with open(PROCESSED_DATA_DIR / 'SugarbeetGenes.json', 'r') as f:
            all_crop_genes = json.load(f)

    except FileNotFoundError as e:
        logging.error(f"FATAL: A required data asset was not found. Please run the build scripts. Error: {e}"); sys.exit(1)

    # --- 3. MAIN SIMULATION LOOP ---
    all_hist_results, all_fcst_results = [], []
    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        logging.info("=" * 70 + f"\nPROCESSING YEAR: {year}\n" + "=" * 70)

        # Slice data for the current year
        df_static_year = df_static_all[df_static_all['year'] == year].copy()
        df_initial_conditions_year = df_initial_conditions[df_initial_conditions['year'] == year].copy()
        df_forecast_weather_year = df_forecast_weather[df_forecast_weather['year'] == year].copy()
        df_historical_weather_year = df_historical_weather[df_historical_weather['date'].dt.year == year].copy()

        if df_static_year.empty or df_initial_conditions_year.empty:
            logging.warning(f"Missing static or initial conditions data for {year}. Skipping."); continue

        # Run simulations
        df_hist = run_historical_simulation(df_static_year, df_historical_weather_year, df_initial_conditions_year, all_crop_genes)
        df_fcst = run_forecast_simulation(df_static_year, df_forecast_weather_year, df_initial_conditions_year, all_crop_genes)

        if not df_hist.empty: all_hist_results.append(df_hist)
        if not df_fcst.empty: all_fcst_results.append(df_fcst)

    # --- 4. FINAL ANALYSIS ---
    if all_hist_results and all_fcst_results:
        final_hist_df = pd.concat(all_hist_results, ignore_index=True)
        final_fcst_df = pd.concat(all_fcst_results, ignore_index=True)

        aggregate_and_save_extreme_weather_metrics(final_fcst_df, CONFIG['FILE_PATHS']['EXTREME_WEATHER_METRICS_OUTPUT'])
        analyze_and_plot_ensemble_results(final_hist_df, final_fcst_df, CONFIG['FILE_PATHS']['OUTPUT_DIR'], CONFIG['START_YEAR'], CONFIG['END_YEAR'])

        logging.info("\n" + "=" * 70 + "\n✓ MULTI-YEAR PIPELINE COMPLETED SUCCESSFULLY!\n" + "=" * 70)
    else:
        logging.error("No simulation results were generated across all years. Aborting final analysis.")

    logging.shutdown()
