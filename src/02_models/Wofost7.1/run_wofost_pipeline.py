# File: src/02_models/Wofost7.1/run_wofost_pipeline.py
# Description: Runs the WOFOST Simulation Loop (Smart ESP Version).
#              UPDATED (v6.0): Adds "Mechanism-Informed" Risk Scanning.
#              Calculates: Yield, Sowing Failure Risk, Anoxia Potential, Terminal Freeze, Harvest Respiration.

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
    """
    Optimized Weather Provider for Pandas DataFrames.
    Calculates ET0 on the fly using Penman-Monteith.
    """

    def __init__(self, weather_df, site_data):
        super().__init__()
        self.latitude = site_data['LAT']
        self.elevation = site_data['ELEV']

        # Speed optimization: Convert to dictionary records once
        weather_df = weather_df.copy()
        weather_df['date'] = pd.to_datetime(weather_df['date'])

        self.store = {}

        for row in weather_df.itertuples():
            day = row.date.date()
            tmin, tmax = float(row.tmin), float(row.tmax)
            precip_cm = float(row.precip) / 10.0  # mm to cm
            srad_mj_m2_day = float(row.srad)  # MJ/m2/day
            irrad_kj_m2_day = srad_mj_m2_day * 1_000.0

            # Defaults if missing in analog file
            vap_kpa = float(getattr(row, 'vap', CONFIG['WEATHER_DEFAULTS']['VAPOR_PRESSURE']))
            wind = float(getattr(row, 'wind', CONFIG['WEATHER_DEFAULTS']['WIND_SPEED']))

            vap_hpa = vap_kpa * 10.0

            et0_mm = penman_monteith(day, self.latitude, self.elevation, tmin, tmax, irrad_kj_m2_day, vap_hpa, wind)

            self.store[(day, 0)] = ParameterDict({
                'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_cm,
                'IRRAD': srad_mj_m2_day * 1_000_000.0, 'VAP': vap_hpa, 'WIND': wind,
                'E0': et0_mm / 10.0, 'ES0': et0_mm / 10.0, 'ET0': et0_mm / 10.0, 'SNOWDEPTH': 0.0
            })


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

    return ParameterProvider(cropdata=crop_params, soildata=soildata, sitedata=sitedata), sitedata


def _run_single_forecast_member(district_no, member, weather_for_member_df, crop_params, static_site_row,
                                initial_condition_row):
    logging.getLogger('pcse').setLevel(logging.CRITICAL)
    try:
        # --- 1. Setup Simulation ---
        parameters, site_data = _create_district_specific_parameters(static_site_row, crop_params,
                                                                     initial_condition_row)
        weather_provider = SimpleWeatherDataProvider(weather_for_member_df, site_data)

        crop_start = pd.to_datetime(initial_condition_row['sowing_date']).date()
        crop_end = pd.to_datetime(initial_condition_row['CROP_END_DATE']).date().replace(year=crop_start.year)

        # Validation: Ensure weather covers simulation
        first_weather = weather_for_member_df['date'].dt.date.min()
        last_weather = weather_for_member_df['date'].dt.date.max()

        if crop_start < first_weather or crop_end > last_weather:
            if (last_weather - crop_start).days < 90:
                return {'district_no': district_no, 'member': member, 'simulation_failed': True}
            crop_end = min(crop_end, last_weather)

        agromanagement = [{crop_start: ParameterDict({
            'CropCalendar': ParameterDict({
                'crop_start_date': crop_start, 'crop_start_type': 'emergence',
                'crop_end_date': crop_end, 'crop_end_type': 'harvest',
                'max_duration': CONFIG['AGROMANAGEMENT']['MAX_DURATION']
            }), 'TimedEvents': None, 'StateEvents': None
        })}]

        # --- 2. Run Models (Potential & Water Limited) ---
        model_wlp = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)
        model_wlp.run_till_terminate()
        output_wlp = pd.DataFrame(model_wlp.get_output()).set_index('day')
        loss_factor = CONFIG['CONSTANTS'].get('HARVEST_LOSS_FACTOR', 1.0)
        yield_wlp = (output_wlp.iloc[-1]['TWSO'] * loss_factor) if not output_wlp.empty else 0

        model_pp = Wofost72_PP(parameters, weather_provider, agromanagement)
        model_pp.run_till_terminate()
        output_pp = pd.DataFrame(model_pp.get_output())
        yield_pp = (output_pp.iloc[-1]['TWSO'] * loss_factor) if not output_pp.empty else 0

        # --- 3. Bio-Physical Risk Scanner (The "Mechanism-Informed" Update) ---

        # Prepare weather subsets
        w_df = weather_for_member_df.copy()
        w_df['month'] = w_df['date'].dt.month

        # A. Operational Risk: Spring Mud (March-April)
        # Count days with > 5mm rain. High count = Trafficability Failure.
        spring_mask = w_df['month'].isin([3, 4])
        spring_mud_days = w_df[spring_mask & (w_df['precip'] > 5.0)].shape[0]

        # B. Hydraulic Risk: Anoxia Potential (June-July)
        # Count 3-day periods with > 25mm rain. (Heavy rain on Clay = Death)
        summer_mask = w_df['month'].isin([6, 7])
        if summer_mask.any():
            summer_df = w_df[summer_mask].copy()
            summer_df['precip_3d'] = summer_df['precip'].rolling(3).sum()
            heavy_rain_events = (summer_df['precip_3d'] > 25.0).sum()
        else:
            heavy_rain_events = 0

        # C. Terminal Risk: Killing Frost (Oct-Nov)
        # Did Tmin drop below -2C? If yes, harvest is compromised.
        fall_mask = w_df['month'].isin([10, 11])
        terminal_frost_occurrence = 1 if (w_df[fall_mask]['tmin'] < -2.0).any() else 0

        # D. Thermodynamics: Harvest Respiration (Sept 15 - Nov 15)
        # Sum of (Tmean - 10) for days > 10C. Sugar Burn.
        harvest_mask = (w_df['month'] == 10) | (w_df['month'] == 11) | \
                       ((w_df['month'] == 9) & (w_df['date'].dt.day >= 15))
        harvest_temps = w_df[harvest_mask]['tmean'] if 'tmean' in w_df.columns else (w_df[harvest_mask]['tmin'] +
                                                                                     w_df[harvest_mask]['tmax']) / 2
        harvest_respiration_gdd = (harvest_temps - 10).clip(lower=0).sum()

        # Standard Metrics
        cumulative_water_stress = (1 - output_wlp['TRA']).sum() if 'TRA' in output_wlp.columns else np.nan
        max_lai = output_wlp['LAI'].max() if 'LAI' in output_wlp.columns else 0.0

        return {
            'district_no': district_no, 'member': member,
            'yield_water_limited': yield_wlp, 'yield_potential': yield_pp,
            'cumulative_water_stress': cumulative_water_stress,
            'max_lai_achieved': max_lai,

            # NEW MECHANISTIC METRICS
            'spring_mud_days': spring_mud_days,
            'summer_heavy_rain_events': heavy_rain_events,
            'terminal_frost_occurrence': terminal_frost_occurrence,
            'harvest_respiration_gdd': harvest_respiration_gdd,

            'simulation_failed': False
        }

    except Exception as e:
        return {'district_no': district_no, 'member': member, 'simulation_failed': True}


def run_historical_simulation(df_static_year, df_historical_weather_year, df_initial_conditions_year, crop_params):
    results = []
    year = df_static_year['year'].iloc[0]
    for _, static_site_row in tqdm(df_static_year.iterrows(), total=len(df_static_year), desc=f"Historical Sim {year}"):
        district_no = static_site_row['district_no']
        weather_df = df_historical_weather_year[df_historical_weather_year['district_no'] == district_no].copy()
        if weather_df.empty: continue
        try:
            initial_condition_row = \
            df_initial_conditions_year[df_initial_conditions_year['district_no'] == district_no].iloc[0]
            res = _run_single_forecast_member(
                district_no, "historical", weather_df, crop_params, static_site_row, initial_condition_row
            )
            if not res.get('simulation_failed', True):
                results.append({
                    'year': year, 'district_no': district_no,
                    'actual_yield': static_site_row['kreisYield'],
                    'lintul_yield_perfect_weather': res['yield_water_limited']
                })
        except Exception:
            continue
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
    if not tasks: return pd.DataFrame()

    year = df_static_year['year'].iloc[0]
    ensemble_outputs = Parallel(n_jobs=-1, backend='loky')(tqdm(tasks, desc=f"Forecast Sim {year}"))
    valid_results = [res for res in ensemble_outputs if not res.get('simulation_failed', False)]

    df_res = pd.DataFrame(valid_results)
    if not df_res.empty: df_res['year'] = year
    return df_res


def aggregate_and_save_extreme_weather_metrics(df_fcst_ensemble, output_path):
    logging.info("=" * 70 + "\n[ANALYSIS] Aggregating Bio-Physical Risk Features...\n" + "=" * 70)
    if df_fcst_ensemble.empty: return

    def p90(x): return x.quantile(0.90)

    p90.__name__ = 'p90'

    def p10(x): return x.quantile(0.10)

    p10.__name__ = 'p10'

    # NEW Aggregations for the "Mechanism-Informed" Features
    aggs = {
        'simulation_failed': ['mean'],
        'cumulative_water_stress': ['mean', 'std', p90],

        # NEW: Probability of Operational Failures
        'spring_mud_days': ['mean'],  # Average days of delay/mud
        'summer_heavy_rain_events': ['mean'],  # Anoxia Risk (Probability if scaled)
        'terminal_frost_occurrence': ['mean'],  # Probability of Frost Event (0-1)
        'harvest_respiration_gdd': ['mean']  # Expected heat stress at harvest
    }

    # Apply only to existing columns
    valid_aggs = {k: v for k, v in aggs.items() if k in df_fcst_ensemble.columns}

    df_metrics = df_fcst_ensemble.groupby(['year', 'district_no']).agg(valid_aggs).reset_index()
    df_metrics.columns = ['_'.join(col).strip() for col in df_metrics.columns.values]
    df_metrics.rename(columns={'year_': 'year', 'district_no_': 'district_no'}, inplace=True)

    # Rename for clarity in Feature Engineering script
    rename_map = {
        'spring_mud_days_mean': 'prob_sowing_failure',  # Proxy
        'summer_heavy_rain_events_mean': 'anoxia_events',
        'terminal_frost_occurrence_mean': 'prob_terminal_freeze',
        'harvest_respiration_gdd_mean': 'harvest_respiration_risk'
    }
    df_metrics.rename(columns=rename_map, inplace=True)

    df_metrics.to_csv(output_path, index=False)
    logging.info(f"[ANALYSIS] ✓ Risk features saved to {output_path}")


def main():
    logging.info("=" * 70 + "\nStarting WOFOST Pipeline (Smart ESP + Risk Scanner)\n" + "=" * 70)
    output_dir = Path(CONFIG['FILE_PATHS']['OUTPUT_DIR'])
    output_dir.mkdir(exist_ok=True)

    try:
        logging.info("Loading static assets...")
        df_static_all = pd.read_csv(PROCESSED_DATA_DIR / 'StaticSiteData.csv', dtype={'district_no': str})
        df_initial_conditions = pd.read_csv(PROCESSED_DATA_DIR / 'InitialConditions.csv', dtype={'district_no': str},
                                            parse_dates=['sowing_date', 'CROP_END_DATE'])
        weather_path = Path(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'])
        weather_files = list(weather_path.glob("*.csv"))
        df_historical_weather = pd.concat((pd.read_csv(f, parse_dates=['date'], dtype={'district_no': str}) for f in
                                           tqdm(weather_files, desc="Loading History")), ignore_index=True)

        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            cp_base = yaml.safe_load(f)['CropParameters']
        base_params = {**cp_base.get('GenericC3', {}), **cp_base['EcoTypes']['sugarbeet'],
                       **cp_base['Varieties']['Sugarbeet_601']}

        with open(config.PROCESSED_DATA_DIR / 'GeneticGainFactors.json', 'r') as f:
            genetic_factors = json.load(f)

    except Exception as e:
        logging.error(f"FATAL: {e}", exc_info=True)
        sys.exit(1)

    forecast_files = list(FORECAST_PARTS_DIR.glob('forecast_*.parquet'))
    if not forecast_files:
        logging.error("No forecast files found.")
        sys.exit(1)

    # Task Discovery & Filtering
    tasks = []
    for f in forecast_files:
        match = re.search(r'forecast_(\d+)_(\d{4})\.parquet', f.name)
        if match: tasks.append({'district_no': match.group(1), 'year': int(match.group(2))})
    df_tasks = pd.DataFrame(tasks).drop_duplicates()

    start_year, end_year = CONFIG.get('START_YEAR'), CONFIG.get('END_YEAR')
    if start_year: df_tasks = df_tasks[df_tasks['year'] >= start_year]
    if end_year: df_tasks = df_tasks[df_tasks['year'] <= end_year]

    df_static_all = pd.merge(df_static_all, df_tasks, on=['district_no', 'year'], how='inner')
    df_initial_conditions = pd.merge(df_initial_conditions, df_tasks, on=['district_no', 'year'], how='inner')

    all_fcst_results, all_hist_results = [], []

    for year in sorted(df_tasks['year'].unique()):
        logging.info(f"Processing Year: {year}")

        # Genetic Gain
        crop_params = ParameterDict()
        for k, v in base_params.items():
            if k not in ['Metadata', '<<']: crop_params.add_variable(k,
                                                                     v[0] if isinstance(v, list) and len(v) > 0 else v)

        factors = genetic_factors.get(str(year), {})
        if 'AMAXTB' in crop_params:
            crop_params['AMAXTB'] = [
                v * factors.get('AMAX_FACTOR', 1.0) if (i + 1) % 2 == 0 else v
                for i, v in enumerate(crop_params['AMAXTB'])
            ]

        # --- NEW: Apply EFF (Light Use Efficiency) - CRITICAL FOR SLOPE ---
        if 'EFFTB' in crop_params:
            crop_params['EFFTB'] = [
                v * factors.get('EFF_FACTOR', 1.0) if (i + 1) % 2 == 0 else v
                for i, v in enumerate(crop_params['EFFTB'])
            ]

        crop_params['TSUM1'] *= factors.get('TSUM1_FACTOR', 1.0)
        crop_params['CVO'] *= factors.get('CVO_FACTOR', 1.0)

        # Load Forecasts
        year_files = list(FORECAST_PARTS_DIR.glob(f'*_{year}.parquet'))
        if year_files:
            df_fcst_weather = pd.concat((pd.read_parquet(f) for f in year_files), ignore_index=True)
            df_fcst = run_forecast_simulation(
                df_static_all[df_static_all['year'] == year],
                df_fcst_weather,
                df_initial_conditions[df_initial_conditions['year'] == year],
                crop_params.copy()
            )
            if not df_fcst.empty: all_fcst_results.append(df_fcst)

        # Load Historical (Validation)
        df_hist_w = df_historical_weather[df_historical_weather['date'].dt.year == year]
        df_hist = run_historical_simulation(
            df_static_all[df_static_all['year'] == year],
            df_hist_w,
            df_initial_conditions[df_initial_conditions['year'] == year],
            crop_params.copy()
        )
        if not df_hist.empty: all_hist_results.append(df_hist)

    # Save
    if all_fcst_results:
        final_fcst = pd.concat(all_fcst_results, ignore_index=True)
        final_fcst.to_csv(output_dir / "forecast_ensemble_results_raw.csv", index=False)
        aggregate_and_save_extreme_weather_metrics(final_fcst, output_dir / "forecast_extreme_weather_metrics.csv")

    if all_hist_results:
        pd.concat(all_hist_results, ignore_index=True).to_csv(output_dir / "historical_validation_results.csv",
                                                              index=False)

    logging.info("✓ Pipeline Completed.")


if __name__ == "__main__":
    main()