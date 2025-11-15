# File: run_wofost_pipeline.py

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
from sklearn.metrics import mean_absolute_error, r2_score
import argparse

# --- Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

CONFIG = config.WOFOST_CONFIG
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
FORECAST_PARTS_DIR = PROCESSED_DATA_DIR / 'forecast_weather_parts'

# --- Logging Setup ---
# Set our script's logger to INFO level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s',
                    stream=sys.stderr)


# --- Class Definitions (unchanged) ---
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


class SimpleWeatherDataProvider(WeatherDataProvider):
    def __init__(self, weather_df, site_data):
        super().__init__()
        self.latitude = site_data['LAT'];
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
            self.store[(day, 0)] = ParameterDict(
                {'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_cm,
                 'IRRAD': srad_mj_m2_day * 1_000_000.0, 'VAP': vap_hpa, 'WIND': wind, 'E0': et0_mm / 10.0,
                 'ES0': et0_mm / 10.0, 'ET0': et0_mm / 10.0, 'SNOWDEPTH': 0.0})


def _create_district_specific_parameters(static_site_row, crop_params_for_year, initial_condition_row):
    sitedata = ParameterDict();
    soildata = ParameterDict()
    sitedata.LAT = static_site_row['latitude']
    sitedata.LON = static_site_row['longitude']
    sitedata.ELEV = static_site_row['avg_elevation']
    sitedata.WAV = initial_condition_row['WAV']
    for param in ['NOTINF', 'SSMAX', 'SMW', 'SMFCF', 'SM0', 'CRAIRC', 'K0', 'SOPE', 'KSUB', 'RDMSOL']:
        if param in ['NOTINF', 'SSMAX']:
            sitedata[param] = static_site_row[param]
        else:
            soildata[param] = static_site_row[param]
    sitedata.IFUNRN = 0.0;
    sitedata.SSI = 0.0;
    sitedata.SMLIM = soildata.SMFCF
    cropdata = ParameterDict(crop_params_for_year)
    return ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata), sitedata


# --- THE MAIN FIXES ARE IN THIS FUNCTION ---
def _run_single_forecast_member(weather_for_member_df, crop_params_for_year, static_site_row, initial_condition_row):
    # --- FIX 1: Silence the PCSE logger within each worker process ---
    # This prevents any file lock errors from happening.
    logging.getLogger('pcse').setLevel(logging.CRITICAL)

    try:
        parameters, site_data = _create_district_specific_parameters(static_site_row, crop_params_for_year,
                                                                     initial_condition_row)
        weather_provider = SimpleWeatherDataProvider(weather_for_member_df, site_data)
        crop_start = pd.to_datetime(initial_condition_row['sowing_date']).date()
        crop_end = pd.to_datetime(initial_condition_row['CROP_END_DATE']).date().replace(year=crop_start.year)

        agromanagement = [{
            crop_start: ParameterDict({'CropCalendar': ParameterDict(
                {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
                 'crop_end_type': 'harvest', 'max_duration': CONFIG['AGROMANAGEMENT']['MAX_DURATION']}),
                                       'TimedEvents': None, 'StateEvents': None})
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
        return {'yield_water_limited': yield_wlp, 'yield_potential': yield_pp,
                'drought_stress_index': drought_stress_index, 'simulation_failed': False, 'error_message': None}
    except Exception as e:
        # --- FIX 2: Capture the actual error message for later analysis ---
        error_msg = f"{type(e).__name__}: {str(e)}"
        return {'yield_water_limited': np.nan, 'yield_potential': np.nan, 'drought_stress_index': np.nan,
                'simulation_failed': True, 'error_message': error_msg}


# --- This function is now focused on diagnosis of results ---
def analyze_and_plot_ensemble_results(df_hist, df_fcst_ensemble, output_dir, start_year, end_year):
    logging.info("=" * 70 + "\n[ANALYSIS] Starting Post-Simulation Analysis\n" + "=" * 70)

    # --- Diagnostic for Simulation Failures ---
    if 'error_message' in df_fcst_ensemble.columns:
        failed_sims = df_fcst_ensemble[df_fcst_ensemble['simulation_failed'] == True]
        if not failed_sims.empty:
            logging.warning(f"Found {len(failed_sims)} failed forecast simulations.")
            error_counts = failed_sims['error_message'].value_counts().head(5)
            logging.warning("Top 5 simulation error messages:")
            for msg, count in error_counts.items():
                logging.warning(f"  - [{count} times] {msg}")

    dmc = CONFIG['CONSTANTS']['DMC_SUGARBEET']
    df_fcst_ensemble['yield_wlp_fresh_dt'] = (df_fcst_ensemble['yield_water_limited'] / dmc) / 100.0
    df_hist['perfect_yield_dt'] = (df_hist['lintul_yield_perfect_weather'] / dmc) / 100.0

    df_fcst_agg = df_fcst_ensemble.groupby(['year', 'district_no']).agg(
        forecast_yield_mean=('yield_wlp_fresh_dt', 'mean'),
        sim_failure_rate=('simulation_failed', 'mean')).reset_index()

    df_final = pd.merge(df_hist[['year', 'district_no', 'actual_yield', 'perfect_yield_dt']], df_fcst_agg,
                        on=['year', 'district_no'], how='inner')

    df_final_clean = df_final.dropna(subset=['actual_yield', 'perfect_yield_dt', 'forecast_yield_mean'])

    if df_final_clean.empty:
        logging.error("[ANALYSIS] FINAL FAILURE: No valid merged and cleaned results found.")
        return

    mae_p = mean_absolute_error(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt'])
    r2_p = r2_score(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt'])
    mae_f = mean_absolute_error(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'])
    r2_f = r2_score(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'])
    print(
        f"\n--- Overall Performance ---\nPerfect Weather: MAE={mae_p:.2f}, R²={r2_p:.3f}\nForecast Weather: MAE={mae_f:.2f}, R²={r2_f:.3f}\n")


# --- All other functions (run_forecast_simulation, run_historical_simulation) are unchanged ---
def run_forecast_simulation(df_static_year, df_forecast_weather_year, df_initial_conditions_year, all_crop_genes):
    full_ensemble_results = [];
    crop_params_for_year = all_crop_genes[str(df_static_year['year'].iloc[0])]
    static_site_map = df_static_year.set_index('district_no').to_dict('index');
    initial_conditions_map = df_initial_conditions_year.set_index('district_no').to_dict('index')
    grouped_weather = df_forecast_weather_year.groupby(['district_no', 'member'])

    tasks = []
    for (district_no, member), weather_for_member_df in grouped_weather:
        if district_no in static_site_map and district_no in initial_conditions_map:
            static_site_row = static_site_map[district_no];
            initial_condition_row = initial_conditions_map[district_no]
            tasks.append(
                delayed(_run_single_forecast_member)(weather_for_member_df, crop_params_for_year, static_site_row,
                                                     initial_condition_row))

    if tasks:
        ensemble_outputs = Parallel(n_jobs=-1, backend='loky')(
            tqdm(tasks, desc=f"Forecast Sim {df_static_year['year'].iloc[0]}"))
    else:
        ensemble_outputs = []

    result_idx = 0
    for (district_no, member), _ in grouped_weather:
        if district_no in static_site_map and district_no in initial_conditions_map:
            if result_idx < len(ensemble_outputs):
                result = ensemble_outputs[result_idx];
                full_ensemble_results.append(
                    {'year': df_static_year['year'].iloc[0], 'district_no': district_no, 'member': member, **result})
                result_idx += 1

    return pd.DataFrame(full_ensemble_results)


def run_historical_simulation(df_static_year, df_historical_weather_year, df_initial_conditions_year, all_crop_genes):
    results = [];
    year = df_static_year['year'].iloc[0];
    crop_params_for_year = all_crop_genes[str(year)]
    for _, static_site_row in tqdm(df_static_year.iterrows(), total=len(df_static_year), desc=f"Historical Sim {year}"):
        district_no = static_site_row['district_no'];
        weather_df = df_historical_weather_year[df_historical_weather_year['district_no'] == district_no]
        if weather_df.empty: continue
        try:
            initial_condition_row = \
                df_initial_conditions_year[df_initial_conditions_year['district_no'] == district_no].iloc[0]
            parameters, site_data = _create_district_specific_parameters(static_site_row, crop_params_for_year,
                                                                         initial_condition_row)
            weather_provider = SimpleWeatherDataProvider(weather_df, site_data)
            crop_start = pd.to_datetime(initial_condition_row['sowing_date']).date();
            crop_end = pd.to_datetime(initial_condition_row['CROP_END_DATE']).date().replace(year=year)

            agromanagement = [{
                crop_start: ParameterDict({'CropCalendar': ParameterDict(
                    {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
                     'crop_end_type': 'harvest', 'max_duration': CONFIG['AGROMANAGEMENT']['MAX_DURATION']}),
                                           'TimedEvents': None, 'StateEvents': None})
            }]

            model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement);
            model.run_till_terminate()
            simulated_yield = model.get_output()[-1]['TWSO'] if model.get_output() else np.nan
        except Exception as e:
            simulated_yield = np.nan
        results.append({'year': year, 'district_no': district_no, 'actual_yield': static_site_row['kreisYield'],
                        'lintul_yield_perfect_weather': simulated_yield})
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the WOFOST pipeline for a range of years.")
    parser.add_argument("-l", "--limit-years", type=int, default=None,
                        help="Limit the run to a specific number of years for testing.")
    args = parser.parse_args()

    logging.info("=" * 70 + "\nStarting WOFOST Pipeline (V4 Partition Consumer)\n" + "=" * 70)
    output_dir = Path(CONFIG['FILE_PATHS']['OUTPUT_DIR']);
    output_dir.mkdir(exist_ok=True)

    try:
        logging.info("Loading static data assets...")
        df_static_all = pd.read_csv(PROCESSED_DATA_DIR / 'StaticSiteData.csv', dtype={'district_no': str})
        df_initial_conditions = pd.read_csv(PROCESSED_DATA_DIR / 'InitialConditions.csv', dtype={'district_no': str},
                                            parse_dates=['sowing_date', 'CROP_END_DATE'])

        weather_path = Path(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'])
        weather_files = list(weather_path.glob("*.csv"))
        df_historical_weather = pd.concat((pd.read_csv(f, parse_dates=['date'], dtype={'district_no': str}) for f in
                                           tqdm(weather_files, desc="Loading Historical Weather")), ignore_index=True)
        with open(PROCESSED_DATA_DIR / 'SugarbeetGenes.json', 'r') as f:
            all_crop_genes = json.load(f)
    except Exception as e:
        logging.error(f"FATAL: Error loading static assets. Error: {e}", exc_info=True);
        sys.exit(1)

    all_years = range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1)
    if args.limit_years:
        years_to_process = list(all_years)[:args.limit_years]
        logging.warning(
            f"--- !!! TEST RUN !!! Limiting to the first {args.limit_years} year(s): {years_to_process} ---")
    else:
        years_to_process = all_years

    all_hist_results, all_fcst_results = [], []
    for year in years_to_process:
        logging.info("=" * 70 + f"\nPROCESSING YEAR: {year}\n" + "=" * 70)

        df_static_year = df_static_all[df_static_all['year'] == year].copy()
        df_initial_conditions_year = df_initial_conditions[df_initial_conditions['year'] == year].copy()
        df_historical_weather_year = df_historical_weather[df_historical_weather['date'].dt.year == year].copy()

        logging.info(f"Loading partitioned forecast weather for {year}...")
        forecast_files_for_year = list(FORECAST_PARTS_DIR.glob(f'*_{year}.parquet'))
        if not forecast_files_for_year:
            logging.warning(f"No forecast weather files found for {year}. Skipping forecast simulation.");
            continue

        df_forecast_weather_year = pd.concat(
            (pd.read_parquet(f) for f in tqdm(forecast_files_for_year, desc=f"Loading Forecast Parts {year}")),
            ignore_index=True
        )

        if df_static_year.empty: logging.warning(f"Missing static data for {year}. Skipping."); continue

        df_hist = run_historical_simulation(df_static_year, df_historical_weather_year, df_initial_conditions_year,
                                            all_crop_genes)
        df_fcst = run_forecast_simulation(df_static_year, df_forecast_weather_year, df_initial_conditions_year,
                                          all_crop_genes)

        if not df_hist.empty: all_hist_results.append(df_hist)
        if not df_fcst.empty: all_fcst_results.append(df_fcst)

    if all_hist_results and all_fcst_results:
        final_hist_df = pd.concat(all_hist_results, ignore_index=True)
        final_fcst_df = pd.concat(all_fcst_results, ignore_index=True)

        # --- NEW: Save the full ensemble output for the dashboard ---
        try:
            ensemble_output_path = output_dir / f"forecast_ensemble_{CONFIG['START_YEAR']}-{CONFIG['END_YEAR']}.csv"
            logging.info(f"Saving full ensemble results to {ensemble_output_path}...")
            final_fcst_df.to_csv(ensemble_output_path, index=False)
            logging.info("Save successful.")
        except Exception as e:
            logging.error(f"Failed to save ensemble output file: {e}")
        # --- END NEW ---

        analyze_and_plot_ensemble_results(final_hist_df, final_fcst_df, output_dir, CONFIG['START_YEAR'],
                                          CONFIG['END_YEAR'])
        logging.info("\n" + "=" * 70 + "\n✓ MULTI-YEAR PIPELINE COMPLETED SUCCESSFULLY!\n" + "=" * 70)
    else:
        logging.error("No simulation results generated. Aborting final analysis.")
