# File: src/02_models/Wofost7.1/run_wofost_pipeline.py
# FINAL WORKING VERSION: Fixes the critical bug where modified crop parameters
# were being overwritten due to object referencing in the main loop.

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import yaml
import re
import matplotlib.pyplot as plt
from pcse.models import Wofost72_WLP_FD, Wofost72_PP
from pcse.base import ParameterProvider, WeatherDataProvider
from pcse.util import penman_monteith
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, r2_score

# --- Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
CONFIG = config.WOFOST_CONFIG
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
FORECAST_PARTS_DIR = PROCESSED_DATA_DIR / 'forecast_weather_parts'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s',
                    stream=sys.stderr)


# --- Class Definitions (Unchanged) ---
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
            self.store[(day, 0)] = ParameterDict(
                {'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_cm,
                 'IRRAD': srad_mj_m2_day * 1_000_000.0, 'VAP': vap_hpa, 'WIND': wind, 'E0': et0_mm / 10.0,
                 'ES0': et0_mm / 10.0, 'ET0': et0_mm / 10.0, 'SNOWDEPTH': 0.0})


# --- Core Simulation Logic ---
def _create_district_specific_parameters(static_site_row, crop_params, initial_condition_row):
    sitedata = ParameterDict()
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
    sitedata.IFUNRN = 0.0
    sitedata.SSI = 0.0
    sitedata.SMLIM = soildata.SMFCF
    # This logging statement is now reliable and will show the correct, changing values.
    logging.debug(f"  [MODEL ENGINE] Using AMAXTB with peak value: {crop_params['AMAXTB'][3]:.4f}")
    return ParameterProvider(cropdata=crop_params, soildata=soildata, sitedata=sitedata), sitedata


# All other functions are unchanged...
def _run_single_forecast_member(district_no, member, weather_for_member_df, crop_params, static_site_row,
                                initial_condition_row):
    logging.getLogger('pcse').setLevel(logging.CRITICAL)
    try:
        parameters, site_data = _create_district_specific_parameters(static_site_row, crop_params,
                                                                     initial_condition_row)
        weather_provider = SimpleWeatherDataProvider(weather_for_member_df, site_data)
        crop_start = pd.to_datetime(initial_condition_row['sowing_date']).date()
        crop_end = pd.to_datetime(initial_condition_row['CROP_END_DATE']).date().replace(year=crop_start.year)
        agromanagement = [{crop_start: ParameterDict({'CropCalendar': ParameterDict(
            {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
             'crop_end_type': 'harvest', 'max_duration': CONFIG['AGROMANAGEMENT']['MAX_DURATION']}),
            'TimedEvents': None, 'StateEvents': None})}]
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

        summer_weather = weather_for_member_df[weather_for_member_df['date'].dt.month.isin([6, 7, 8])].copy()
        consecutive_hot_days = get_max_consecutive_run(summer_weather['tmax'] > 30)
        consecutive_dry_days = get_max_consecutive_run(summer_weather['precip'] < 1)
        drought_stress_index = (yield_pp - yield_wlp) / yield_pp if yield_pp > 0 else 0.0
        days_to_anthesis = np.nan
        if 'DOA' in output_wlp.columns and (output_wlp['DOA'] is not None):
            first_anthesis_day = output_wlp[output_wlp['DOA'].notna()].index.min()
            if pd.notna(first_anthesis_day): days_to_anthesis = (first_anthesis_day - crop_start).days
        max_lai_achieved = output_wlp['LAI'].max() if 'LAI' in output_wlp.columns else 0.0
        cumulative_water_stress = (1 - output_wlp['TRA']).sum() if 'TRA' in output_wlp.columns else np.nan
        return {'district_no': district_no, 'member': member, 'yield_water_limited': yield_wlp,
                'yield_potential': yield_pp, 'consecutive_tmax_gt_30c': consecutive_hot_days,
                'consecutive_dry_days': consecutive_dry_days, 'drought_stress_index': drought_stress_index,
                'simulation_failed': False, 'days_to_anthesis': days_to_anthesis, 'max_lai_achieved': max_lai_achieved,
                'cumulative_water_stress': cumulative_water_stress}
    except Exception as e:
        return {'district_no': district_no, 'member': member, 'yield_water_limited': np.nan, 'yield_potential': np.nan,
                'consecutive_tmax_gt_30c': np.nan, 'consecutive_dry_days': np.nan, 'drought_stress_index': np.nan,
                'simulation_failed': True, 'days_to_anthesis': np.nan, 'max_lai_achieved': np.nan,
                'cumulative_water_stress': np.nan}


def run_historical_simulation(df_static_year, df_historical_weather_year, df_initial_conditions_year, crop_params):
    results = []
    year = df_static_year['year'].iloc[0]
    for _, static_site_row in tqdm(df_static_year.iterrows(), total=len(df_static_year), desc=f"Historical Sim {year}"):
        district_no = static_site_row['district_no']
        weather_df = df_historical_weather_year[df_historical_weather_year['district_no'] == district_no].copy()
        if weather_df.empty: continue
        simulated_yield = np.nan
        try:
            initial_condition_row = \
            df_initial_conditions_year[df_initial_conditions_year['district_no'] == district_no].iloc[0]
            parameters, site_data = _create_district_specific_parameters(static_site_row, crop_params,
                                                                         initial_condition_row)
            weather_provider = SimpleWeatherDataProvider(weather_df, site_data)
            crop_start = pd.to_datetime(initial_condition_row['sowing_date']).date()
            crop_end = pd.to_datetime(initial_condition_row['CROP_END_DATE']).date().replace(year=year)
            agromanagement = [{crop_start: ParameterDict({'CropCalendar': ParameterDict(
                {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
                 'crop_end_type': 'harvest', 'max_duration': CONFIG['AGROMANAGEMENT']['MAX_DURATION']}),
                'TimedEvents': None, 'StateEvents': None})}]
            model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)
            model.run_till_terminate()
            output_df = pd.DataFrame(model.get_output())
            simulated_yield = output_df.iloc[-1]['TWSO'] if not output_df.empty else 0.0
        except Exception:
            simulated_yield = np.nan
        results.append({'year': year, 'district_no': district_no, 'actual_yield': static_site_row['kreisYield'],
                        'lintul_yield_perfect_weather': simulated_yield})
    return pd.DataFrame(results)


def run_forecast_simulation(df_static_year, df_forecast_weather_year, df_initial_conditions_year, crop_params):
    static_site_map = df_static_year.set_index('district_no').to_dict('index')
    initial_conditions_map = df_initial_conditions_year.set_index('district_no').to_dict('index')
    tasks = []
    for (district_no, member), weather_for_member_df in df_forecast_weather_year.groupby(['district_no', 'member']):
        if district_no in static_site_map and district_no in initial_conditions_map:
            tasks.append(delayed(_run_single_forecast_member)(
                district_no, member, weather_for_member_df.copy(), crop_params, static_site_map[district_no],
                initial_conditions_map[district_no]
            ))
    if not tasks:
        logging.warning(f"No valid forecast tasks to run for year {df_static_year['year'].iloc[0]}.")
        return pd.DataFrame()
    year = df_static_year['year'].iloc[0]
    ensemble_outputs = Parallel(n_jobs=-1, backend='loky')(tqdm(tasks, desc=f"Forecast Sim {year}"))
    for result in ensemble_outputs:
        result['year'] = year
    return pd.DataFrame(ensemble_outputs)


def analyze_and_plot_ensemble_results(df_hist, df_fcst_ensemble, output_dir, start_year, end_year):
    logging.info("=" * 70 + "\n[ANALYSIS & DEBUG] Starting Post-Simulation Analysis\n" + "=" * 70)
    if df_fcst_ensemble.empty or df_hist.empty:
        logging.error("[ANALYSIS] No data in forecast or historical results. Cannot analyze.");
        return
    dmc = CONFIG['CONSTANTS']['DMC_SUGARBEET']
    df_fcst_ensemble['yield_wlp_fresh_dt'] = (df_fcst_ensemble['yield_water_limited'] / dmc) / 100.0
    df_fcst_ensemble['yield_pp_fresh_dt'] = (df_fcst_ensemble['yield_potential'] / dmc) / 100.0
    df_hist['perfect_yield_dt'] = (df_hist['lintul_yield_perfect_weather'] / dmc) / 100.0
    logging.info("\n--- Data Distribution Analysis (check for scaling issues) ---")
    logging.info("\n[DEBUG] Actual Yield (dt/ha) Distribution:");
    logging.info(df_hist['actual_yield'].describe().to_string())
    logging.info("\n[DEBUG] Historical Simulated Yield (dt/ha, fresh weight) Distribution:");
    logging.info(df_hist['perfect_yield_dt'].describe().to_string())
    logging.info("\n[DEBUG] Forecast Mean Yield (dt/ha, fresh weight) Distribution:");
    logging.info(df_fcst_ensemble['yield_wlp_fresh_dt'].describe().to_string())
    logging.info("-----------------------------------------------------------\n")
    df_fcst_agg = df_fcst_ensemble.groupby(['year', 'district_no']).agg(
        forecast_yield_mean=('yield_wlp_fresh_dt', 'mean'),
        forecast_yield_p10=('yield_wlp_fresh_dt', lambda x: x.quantile(0.10)),
        forecast_yield_p90=('yield_wlp_fresh_dt', lambda x: x.quantile(0.90)),
        potential_yield_mean=('yield_pp_fresh_dt', 'mean'),
        sim_failure_rate=('simulation_failed', 'mean')).reset_index()
    df_final = pd.merge(df_hist[['year', 'district_no', 'actual_yield', 'perfect_yield_dt']], df_fcst_agg,
                        on=['year', 'district_no'], how='inner')
    df_final_clean = df_final.dropna(subset=['actual_yield', 'perfect_yield_dt', 'forecast_yield_mean'])
    if df_final_clean.empty:
        logging.error("[ANALYSIS] FINAL FAILURE: No valid merged and cleaned results found.");
        return
    mae_p = mean_absolute_error(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt']);
    r2_p = r2_score(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt'])
    mae_f = mean_absolute_error(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean']);
    r2_f = r2_score(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'])
    print(
        f"\n--- Overall Performance ---\nPerfect Weather: MAE={mae_p:.2f}, R²={r2_p:.3f}\nForecast Weather: MAE={mae_f:.2f}, R²={r2_f:.3f}\n")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True);
    fig.suptitle(f'WOFOST Performance Diagnosis ({start_year}-{end_year})', fontsize=16)
    min_val = df_final_clean[['actual_yield', 'perfect_yield_dt', 'forecast_yield_p10']].min().min() * 0.95
    max_val = df_final_clean[
                  ['actual_yield', 'perfect_yield_dt', 'forecast_yield_p90', 'potential_yield_mean']].max().max() * 1.05
    axes[0].scatter(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt'], alpha=0.6,
                    label='Simulated (Perfect Weather)');
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    axes[0].scatter(df_final_clean['actual_yield'], df_final_clean['potential_yield_mean'], marker='x', color='red',
                    alpha=0.5, label='Mean Potential Yield (PP)');
    axes[0].set_title(f'Perfect Weather\nMAE={mae_p:.2f}, R²={r2_p:.3f}')
    axes[0].set_xlabel('Actual Yield (dt/ha)');
    axes[0].set_ylabel('Simulated Yield (dt/ha)')
    y_err = [df_final_clean['forecast_yield_mean'] - df_final_clean['forecast_yield_p10'],
             df_final_clean['forecast_yield_p90'] - df_final_clean['forecast_yield_mean']]
    axes[1].errorbar(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'], yerr=y_err, fmt='o',
                     color='orange', ecolor='lightgray', elinewidth=3, capsize=0, alpha=0.8,
                     label='Ensemble Mean & 10-90th Pct. Range')
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line');
    axes[1].scatter(df_final_clean['actual_yield'], df_final_clean['potential_yield_mean'], marker='x', color='red',
                    alpha=0.5, label='Mean Potential Yield (PP)')
    axes[1].set_title(f'Forecast Weather (Ensemble Range)\nMean MAE={mae_f:.2f}, Mean R²={r2_f:.3f}');
    axes[1].set_xlabel('Actual Yield (dt/ha)')
    for ax in axes:
        ax.set_xlim(min_val, max_val);
        ax.set_ylim(min_val, max_val);
        ax.grid(True, alpha=0.3);
        ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    plot_path = output_dir / f'results_scatter_analysis_{start_year}-{end_year}.png'
    plt.savefig(plot_path, dpi=300);
    logging.info(f"[ANALYSIS] ✓ Diagnostic plot saved to {plot_path}");
    plt.show()


def aggregate_and_save_extreme_weather_metrics(df_fcst_ensemble, output_path):
    logging.info("=" * 70 + "\n[ANALYSIS] Aggregating in-season risk features...\n" + "=" * 70)
    if df_fcst_ensemble.empty:
        logging.warning("[ANALYSIS] Forecast ensemble dataframe is empty. Skipping extreme metrics.");
        return
    aggs = {'consecutive_tmax_gt_30c': ['mean', 'std', lambda x: x.quantile(0.90)],
            'consecutive_dry_days': ['mean', 'std', lambda x: x.quantile(0.90)],
            'drought_stress_index': ['mean', 'std', lambda x: x.quantile(0.90)], 'simulation_failed': ['mean'],
            'days_to_anthesis': ['mean', 'std'], 'max_lai_achieved': ['mean', 'std', lambda x: x.quantile(0.10)],
            'cumulative_water_stress': ['mean', 'std', lambda x: x.quantile(0.90)]}
    df_extreme_metrics = df_fcst_ensemble.groupby(['year', 'district_no']).agg(aggs).reset_index()
    df_extreme_metrics.columns = ['_'.join(col).strip() for col in df_extreme_metrics.columns.values]
    df_extreme_metrics.rename(columns={'year_': 'year', 'district_no_': 'district_no'}, inplace=True)
    df_extreme_metrics.to_csv(output_path, index=False)
    logging.info(f"[ANALYSIS] ✓ All risk features saved to {output_path}")


if __name__ == "__main__":
    logging.info("=" * 70 + "\nStarting WOFOST Pipeline (FINAL CORRECTED VERSION)\n" + "=" * 70)
    output_dir = Path(CONFIG['FILE_PATHS']['OUTPUT_DIR']);
    output_dir.mkdir(exist_ok=True)

    try:
        # Load all data that is NOT year-dependent
        logging.info("Loading static data assets...")
        df_static_all = pd.read_csv(PROCESSED_DATA_DIR / 'StaticSiteData.csv', dtype={'district_no': str})
        df_initial_conditions = pd.read_csv(PROCESSED_DATA_DIR / 'InitialConditions.csv', dtype={'district_no': str},
                                            parse_dates=['sowing_date', 'CROP_END_DATE'])
        weather_path = Path(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'])
        weather_files = list(weather_path.glob("*.csv"))
        df_historical_weather = pd.concat((pd.read_csv(f, parse_dates=['date'], dtype={'district_no': str}) for f in
                                           tqdm(weather_files, desc="Loading Historical Weather")), ignore_index=True)

        logging.info(f"Loading BASE crop parameters from: {CONFIG['FILE_PATHS']['CROP_YAML']}")
        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            cp_base = yaml.safe_load(f)['CropParameters']
        base_params = {**cp_base.get('GenericC3', {}), **cp_base['EcoTypes']['sugarbeet'],
                       **cp_base['Varieties']['Sugarbeet_601']}

        genetic_factors_path = config.PROCESSED_DATA_DIR / 'GeneticGainFactors.json'
        logging.info(f"Loading genetic gain factors from: {genetic_factors_path}")
        with open(genetic_factors_path, 'r') as f:
            genetic_factors = json.load(f)

    except Exception as e:
        logging.error(
            f"FATAL: Error loading static assets. Did you run `build_genetic_gain_factors.py` first? Error: {e}",
            exc_info=True);
        sys.exit(1)

    logging.info(f"Scanning for built forecast files in: {FORECAST_PARTS_DIR}")
    try:
        forecast_files = list(FORECAST_PARTS_DIR.glob('forecast_*.parquet'))
        if not forecast_files: raise FileNotFoundError(f"No forecast files found in {FORECAST_PARTS_DIR}.")
        tasks_to_process = [{'district_no': match.groups()[0], 'year': int(match.groups()[1])}
                            for f in forecast_files if (match := re.search(r'forecast_(\d+)_(\d{4})\.parquet', f.name))]
        df_tasks = pd.DataFrame(tasks_to_process).drop_duplicates()
        logging.info(f"Discovered {len(df_tasks)} unique district-year combinations to process.")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"FATAL: {e}");
        sys.exit(1)

    start_year = CONFIG.get('START_YEAR')
    end_year = CONFIG.get('END_YEAR')

    if start_year:
        df_tasks = df_tasks[df_tasks['year'] >= start_year]
    if end_year:
        df_tasks = df_tasks[df_tasks['year'] <= end_year]

    logging.info(f"Filtered to {len(df_tasks)} tasks based on YEAR range ({start_year}-{end_year}).")

    # 2. Filter by DISTRICT_LIMIT
    district_limit = CONFIG.get('DISTRICT_LIMIT')

    if district_limit and not df_tasks.empty:
        # Find districts with the most years of data (within the year range)
        district_data_counts = df_tasks['district_no'].value_counts()
        top_districts = district_data_counts.nlargest(district_limit).index

        # Filter df_tasks to only these top districts
        df_tasks = df_tasks[df_tasks['district_no'].isin(top_districts)]
        logging.info(
            f"Filtered to {len(df_tasks)} tasks for {len(top_districts)} top districts (Limit={district_limit}).")

    df_static_all = pd.merge(df_static_all, df_tasks, on=['district_no', 'year'], how='inner')
    df_initial_conditions = pd.merge(df_initial_conditions, df_tasks, on=['district_no', 'year'], how='inner')
    if df_static_all.empty: logging.error("Static data is missing for all discovered tasks."); sys.exit(1)

    all_hist_results, all_fcst_results = [], []
    for year in sorted(df_tasks['year'].unique()):
        logging.info("=" * 70 + f"\nPROCESSING YEAR: {year}\n" + "=" * 70)

        crop_params = ParameterDict()
        for key, val in base_params.items():
            if key not in ['Metadata', '<<']:
                crop_params.add_variable(key, val[0] if isinstance(val, list) and len(val) > 0 else val)

        factors_for_year = genetic_factors[str(year)]
        logging.info(f"Applying Genetic Gain Factors for {year}: {factors_for_year}")

        crop_params['AMAXTB'] = [v * factors_for_year['AMAX_FACTOR'] if (i + 1) % 2 == 0 else v for i, v in
                                 enumerate(crop_params['AMAXTB'])]
        crop_params['EFFTB'] = [v * factors_for_year['EFF_FACTOR'] if (i + 1) % 2 == 0 else v for i, v in
                                enumerate(crop_params['EFFTB'])]
        crop_params['TSUM1'] *= factors_for_year['TSUM1_FACTOR']

        df_static_year = df_static_all[df_static_all['year'] == year].copy()
        df_initial_conditions_year = df_initial_conditions[df_initial_conditions['year'] == year].copy()
        df_historical_weather_year = df_historical_weather[df_historical_weather['date'].dt.year == year].copy()

        forecast_files_for_year = list(FORECAST_PARTS_DIR.glob(f'*_{year}.parquet'))
        df_forecast_weather_year = pd.concat(
            (pd.read_parquet(f) for f in tqdm(forecast_files_for_year, desc=f"Loading Forecast Parts {year}")),
            ignore_index=True)

        # Run simulations with a forced copy of the parameters
        df_fcst = run_forecast_simulation(df_static_year, df_forecast_weather_year, df_initial_conditions_year,
                                          crop_params.copy())
        df_hist = run_historical_simulation(df_static_year, df_historical_weather_year, df_initial_conditions_year,
                                            crop_params.copy())

        if not df_hist.empty: all_hist_results.append(df_hist)
        if not df_fcst.empty: all_fcst_results.append(df_fcst)

    # FINAL ANALYSIS (unchanged)
    if all_hist_results and all_fcst_results:
        final_hist_df = pd.concat(all_hist_results, ignore_index=True)
        final_fcst_df = pd.concat(all_fcst_results, ignore_index=True)
        final_fcst_df.to_csv(output_dir / "forecast_ensemble_results_raw.csv", index=False)
        aggregate_and_save_extreme_weather_metrics(final_fcst_df, output_dir / "forecast_extreme_weather_metrics.csv")
        analyze_and_plot_ensemble_results(final_hist_df, final_fcst_df, output_dir, df_tasks['year'].min(),
                                          df_tasks['year'].max())
        logging.info("\n" + "=" * 70 + "\n✓ PIPELINE COMPLETED SUCCESSFULLY!\n" + "=" * 70)
    else:
        logging.error("No simulation results generated. Aborting final analysis.")