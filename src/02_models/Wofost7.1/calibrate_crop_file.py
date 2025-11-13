import datetime
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import os
import logging
import sys
from pcse.util import penman_monteith
from pcse.models import Wofost72_PP
from pcse.base import ParameterProvider, WeatherDataProvider
from tqdm import tqdm
import geopandas as gpd
import random
import optuna  # <-- The optimizer
from tqdm import tqdm

# --- Ensure the project root is in the Python path ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
try:
    from src import config
except ImportError:
    print("FATAL: Could not import src.config. Make sure this script is in the correct directory.")
    sys.exit(1)

CONFIG = config.WOFOST_CONFIG
# The starting year for the genetic gain trend (e.g., 1980)
GAIN_START_YEAR = CONFIG['START_YEAR']

# --- Setup Logging ---
logging.getLogger().handlers = []
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)
# Silence Optuna's trial-by-trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger('pcse').setLevel(logging.WARNING)


# ==============================================================================
# === 1. COPIED CLASSES FROM YOUR SCRIPT (Needed by the simulator) ===
# ==============================================================================

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
                precip_cm = float(row['precip']) / 10.0
                srad_mj_m2_day = float(row['srad'])
                irrad_j_m2_day = srad_mj_m2_day * 1_000_000.0
                irrad_kj_m2_day = srad_mj_m2_day * 1_000.0
                vap_kpa = float(row.get('vap', fallback_vap_kpa))
                vap_hpa = vap_kpa * 10.0
                et0_mm = penman_monteith(day, self.latitude, self.elevation, tmin, tmax, irrad_kj_m2_day, vap_hpa, wind)
                et0_cm = et0_mm / 10.0
                self.store[(day, 0)] = ParameterDict(
                    {'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_cm,
                     'IRRAD': irrad_j_m2_day, 'VAP': vap_hpa, 'WIND': wind, 'E0': et0_cm, 'ES0': et0_cm, 'ET0': et0_cm,
                     'SNOWDEPTH': 0.0})
            except Exception as e:
                pass  # Suppress errors during simulation


class DynamicSowingManager:
    def __init__(self, sowing_window_start_month=3, sowing_window_start_day=15,
                 sowing_window_end_month=4, sowing_window_end_day=30,
                 temp_threshold_c=7.0, temp_avg_period_days=7):
        self.start_month = sowing_window_start_month
        self.start_day = sowing_window_start_day
        self.end_month = sowing_window_end_month
        self.end_day = sowing_window_end_day
        self.threshold = temp_threshold_c
        self.period = temp_avg_period_days

    def find_sowing_date(self, weather_df_year: pd.DataFrame) -> datetime.date:
        df = weather_df_year.copy()
        if df.empty or 'date' not in df.columns:
            return datetime.date(2000, self.start_month, self.start_day)  # Fallback

        df['mean_temp'] = (df['tmin'] + df['tmax']) / 2
        df['temp_ma'] = df['mean_temp'].rolling(window=self.period, min_periods=self.period).mean()
        try:
            year = df['date'].dt.year.iloc[0]
            window_start = datetime.date(year, self.start_month, self.start_day)
            window_end = datetime.date(year, self.end_month, self.end_day)
        except IndexError:
            return datetime.date(2000, self.start_month, self.start_day)  # Fallback
        sow_mask = (
                (df['date'].dt.date >= window_start) &
                (df['date'].dt.date <= window_end) &
                (df['temp_ma'] >= self.threshold)
        )
        potential_sow_days = df[sow_mask]
        if not potential_sow_days.empty:
            return potential_sow_days['date'].dt.date.iloc[0]
        else:
            return window_end


# ==============================================================================
# === 2. GLOBAL VARIABLES FOR CALIBRATION (To be filled by setup) ===
# ==============================================================================

# The raw crop parameters loaded from the YAML file
base_cropdata_yaml = None
# A list of all historical data points (dicts) to test against
calibration_tasks = []


# ==============================================================================
# === 3. OPTIMIZATION FUNCTIONS (The new logic) ===
# ==============================================================================

def run_single_pp_simulation(
        year_specific_cropdata,
        sitedata,
        soildata,
        weather_df,
        site_info,
        sowing_date,
        harvest_date
):
    """
    Runs a single Wofost72_PP simulation.
    This is the "black box" our optimizer will tune.
    """
    try:
        if sowing_date is None or harvest_date is None or weather_df.empty:
            return 0.0

        parameters = ParameterProvider(cropdata=year_specific_cropdata, sitedata=sitedata, soildata=soildata)
        weather_provider = SimpleWeatherDataProvider(weather_df, site_info)

        agromanagement = [{
            sowing_date: ParameterDict({
                'CropCalendar': ParameterDict({
                    'crop_start_date': sowing_date, 'crop_start_type': 'emergence',
                    'crop_end_date': harvest_date, 'crop_end_type': 'harvest',
                    'max_duration': 300
                }),
                'TimedEvents': None,
                'StateEvents': None
            })
        }]

        model_pp = Wofost72_PP(parameters, weather_provider, agromanagement)
        model_pp.run_till_terminate()
        output = model_pp.get_output()

        simulated_yield = output[-1]['TWSO'] if (output and len(output) > 0) else 0.0
        return simulated_yield

    except Exception as e:
        # Optuna might guess impossible parameters (e.g., TSUM1 > TSUM2)
        # We must catch this and return a "failure" score (0 yield)
        return 0.0


def objective(trial):
    """
    This is the core "scoring" function for Optuna.
    It will be called 1000s of times.
    """
    global base_cropdata_yaml, calibration_tasks

    # === A. Define the search space: "Guess" the base and gain_rate values ===

    # RUE (Radiation-Use Efficiency) - This is the most important
    RUE_base = trial.suggest_float('RUE_base', 1.8, 2.5)
    RUE_gain = trial.suggest_float('RUE_gain', 0.0, 0.03)  # (0 to 3% gain per year)

    # TSUM1 (Emergence to Anthesis) - May get shorter over time
    TSUM1_base = trial.suggest_float('TSUM1_base', 600, 900)
    TSUM1_gain = trial.suggest_float('TSUM1_gain', -2.0, 0.0)  # (e.g., -1.0 = 1 degree-day shorter per year)

    # AMAX (Max CO2 Assimilation Rate) - The "engine power"
    AMAX_base = trial.suggest_float('AMAX_base', 30.0, 50.0)
    AMAX_gain = trial.suggest_float('AMAX_gain', 0.0, 0.2)

    total_penalty = 0.0
    total_gap = 0.0

    # === B. Test these parameters against a random sample of the data ===
    n_samples = 250  # Run 250 random simulations per trial
    if len(calibration_tasks) < n_samples:
        sampled_tasks = calibration_tasks
    else:
        sampled_tasks = random.sample(calibration_tasks, n_samples)

    for task in sampled_tasks:

        # --- 1. Calculate the effective parameters for THIS task's year ---
        year = task['year']
        years_elapsed = year - GAIN_START_YEAR

        effective_RUE = RUE_base + (years_elapsed * RUE_gain)
        effective_TSUM1 = TSUM1_base + (years_elapsed * TSUM1_gain)
        effective_AMAX = AMAX_base + (years_elapsed * AMAX_gain)

        # --- 2. Build the year-specific cropdata object ---
        trial_cropdata = base_cropdata_yaml.copy()
        trial_cropdata['RUE'] = effective_RUE
        # TSUM1 is a list, so we modify its first value
        if isinstance(trial_cropdata['TSUM1'], list) and len(trial_cropdata['TSUM1']) > 0:
            trial_cropdata['TSUM1'][0] = effective_TSUM1
        else:
            trial_cropdata['TSUM1'] = effective_TSUM1  # Fallback

        # AMAX is stored as a list, we must modify the list
        if 'AMAXTB' in trial_cropdata and trial_cropdata['AMAXTB'] is not None:
            amax_table = trial_cropdata['AMAXTB']

            # --- THIS IS THE FIX ---
            # Iterate over the "key" indices
            for i in range(0, len(amax_table), 2):
                # Check if a corresponding "value" index exists
                if (i + 1) < len(amax_table):
                    amax_table[i + 1] = effective_AMAX
            # --- END OF FIX ---

            trial_cropdata['AMAXTB'] = amax_table

        # --- 3. Run the simulation ---
        sim_yield = run_single_pp_simulation(
            trial_cropdata,
            task['sitedata'],
            task['soildata'],
            task['weather_df'],
            task['site_info'],
            task['sowing_date'],
            task['harvest_date']
        )

        actual_yield = task['actual_yield_kg_ha']

        # --- 4. Calculate our custom score ---
        if sim_yield <= 0:  # Model failed, massive penalty
            total_penalty += (actual_yield ** 2) * 1e6
            continue

        diff = sim_yield - actual_yield

        if diff < 0:
            # PENALTY: Simulated yield is BELOW actual. This is a "violation".
            total_penalty += (diff ** 2) * 1e6  # Large quadratic punishment
        else:
            # GAP: Simulated yield is the "ceiling". This is good.
            total_gap += diff  # We want to minimize this gap

    # The final score is the sum of all punishments and all "gaps"
    final_score = (total_penalty + total_gap) / len(sampled_tasks)
    return final_score


def load_all_calibration_data():
    """
    V4: Fixed the 'sitedtata' typo.
    """
    global base_cropdata_yaml, calibration_tasks
    logging.info("Starting to load all historical data for calibration...")

    # --- 1. Load Base Cropdata (from YAML) ---
    logging.info("Loading and parsing base crop YAML file...")
    try:
        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            cp = yaml.safe_load(f)['CropParameters']
        p = {**cp.get('GenericC3', {}), **cp['EcoTypes']['sugarbeet'], **cp['Varieties']['Sugarbeet_601']}
        base_cropdata_yaml = ParameterDict()
        for key, val in p.items():
            if key not in ['Metadata', '<<']:
                base_cropdata_yaml.add_variable(key, val)
        logging.info(f"Successfully loaded base crop params. Test: TSUM1={base_cropdata_yaml.get('TSUM1', ['N/A'])[0]}")
    except Exception as e:
        logging.error(f"FATAL: Could not load or parse crop YAML. Error: {e}", exc_info=True);
        sys.exit(1)

    # --- 2. Load All Static and Yield Data ---
    logging.info("Loading and merging all static/yield data sources...")
    try:
        df_yield = pd.read_csv(CONFIG['FILE_PATHS']['YIELD_DATA'], dtype={'district_no': str})
        df_yield.rename(columns={'yield': 'kreisYield'}, inplace=True)
        df_static_physics = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_SOIL_FEATURES'], dtype={'district_no': str})
        gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH)
        gdf_districts_coords = gdf_districts[['id', 'geometry']].copy()
        gdf_districts_coords['latitude'] = gdf_districts_coords.geometry.centroid.y
        gdf_districts_coords['longitude'] = gdf_districts_coords.geometry.centroid.x
        gdf_districts_coords.rename(columns={'id': 'district_no'}, inplace=True)
        gdf_districts_coords['district_no'] = gdf_districts_coords['district_no'].astype(str).str.zfill(5)
        df_wav = pd.read_csv(CONFIG['FILE_PATHS']['INITIAL_CONDITIONS'], dtype={'district_no': str})
        df_static_base = pd.merge(df_static_physics, gdf_districts_coords[['district_no', 'latitude', 'longitude']],
                                  on='district_no')
        df_static_all = pd.merge(df_yield, df_static_base, on=['district_no'], how='inner')
        df_static_all = pd.merge(df_static_all, df_wav, on=['year', 'district_no'], how='left')
        df_static_all['WAV'] = df_static_all['WAV'].fillna(10.0)
        df_static_all.dropna(subset=['kreisYield'], inplace=True)
        df_static_all = df_static_all[df_static_all['kreisYield'] > 0]

        start_year = CONFIG['START_YEAR']
        end_year = CONFIG['END_YEAR']
        district_limit = CONFIG.get('DISTRICT_LIMIT', None)

        logging.info(f"Applying config: START_YEAR={start_year}, END_YEAR={end_year}, DISTRICT_LIMIT={district_limit}")

        df_static_all = df_static_all[
            (df_static_all['year'] >= start_year) & (df_static_all['year'] <= end_year)
            ].copy()

        if district_limit is not None and district_limit > 0:
            limited_districts = df_static_all['district_no'].unique()[:district_limit]
            df_static_all = df_static_all[df_static_all['district_no'].isin(limited_districts)]
            logging.info(f"Filtered to {len(limited_districts)} districts.")

        if df_static_all.empty:
            logging.error(
                f"FATAL: No data found for the specified year range ({start_year}-{end_year}) and district limit.")
            sys.exit(1)

    except Exception as e:
        logging.error(f"FATAL: Error during static data loading: {e}", exc_info=True);
        sys.exit(1)

    # --- 3. Initialize Sowing Manager ---
    sowing_manager = DynamicSowingManager()

    # --- 4. Loop Through Years and Build Tasks ---
    logging.info("Building calibration task list (this may take a moment)...")
    dmc = CONFIG['CONSTANTS']['DMC_SUGARBEET']

    district_parameter_cache = {}

    for _, row in df_static_all.iterrows():
        district_no = row['district_no']
        if district_no in district_parameter_cache:
            continue

        sitedata = ParameterDict()
        soildata = ParameterDict()
        sitedata.add_variable('LAT', row['latitude'])
        sitedata.add_variable('LON', row['longitude'])
        sitedata.add_variable('ELEV', row['avg_elevation'])
        sitedata.add_variable('WAV', row['WAV'])
        sitedata.add_variable('NOTINF', row['NOTINF'])
        sitedata.add_variable('SSMAX', row['SSMAX'])
        soil_params = ['SMW', 'SMFCF', 'SM0', 'CRAIRC', 'K0', 'SOPE', 'KSUB', 'RDMSOL']
        for param in soil_params:
            if param in row:
                soildata.add_variable(param, row[param])
        sitedata.add_variable('IFUNRN', 0.0)
        sitedata.add_variable('SSI', 0.0)
        sitedata.add_variable('SMLIM', soildata.get('SMFCF', 0.1))

        district_parameter_cache[district_no] = {'sitedata': sitedata, 'soildata': soildata}

    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        df_static_year = df_static_all[df_static_all['year'] == year].copy()
        if df_static_year.empty:
            continue

        hist_weather_path = CONFIG['FILE_PATHS'][
                                'HISTORICAL_DAILY_WEATHER_DIR'] / f"historical_daily_weather_era5_{year}.csv"
        try:
            df_daily_hist_year = pd.read_csv(hist_weather_path, parse_dates=['date'], dtype={'district_no': str})
            df_daily_hist_year['district_no'] = df_daily_hist_year['district_no'].astype(str).str.zfill(5)
        except FileNotFoundError:
            logging.warning(f"Weather for {year} not found. Skipping year.");
            continue

        dynamic_sowing_dates = {}
        for district_no in df_static_year['district_no'].unique():
            district_weather_for_sowing = df_daily_hist_year[df_daily_hist_year['district_no'] == district_no]
            if not district_weather_for_sowing.empty:
                sowing_date = sowing_manager.find_sowing_date(district_weather_for_sowing)
                dynamic_sowing_dates[district_no] = sowing_date

        for _, row in df_static_year.iterrows():
            district_no = row['district_no']
            if district_no not in district_parameter_cache:
                continue

            weather_df = df_daily_hist_year[df_daily_hist_year['district_no'] == district_no]
            if weather_df.empty: continue

            sowing_date = dynamic_sowing_dates.get(district_no)
            if sowing_date is None:
                continue

            params = district_parameter_cache[district_no]
            params['sitedata']['WAV'] = row['WAV']

            task = {
                "id": f"{year}_{district_no}",
                "year": year,
                "actual_yield_kg_ha": row['kreisYield'] * (100 / dmc),

                # --- THIS IS THE FIX ---
                "sitedata": params['sitedata'],  # Was 'sitedtata'
                # --- END OF FIX ---

                "soildata": params['soildata'],
                "site_info": {'LAT': row['latitude'], 'LON': row['longitude'], 'ELEV': row['avg_elevation']},
                "weather_df": weather_df,
                "sowing_date": sowing_date,
                "harvest_date": CONFIG['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)
            }
            calibration_tasks.append(task)

    logging.info(f"Successfully loaded {len(calibration_tasks)} total district-years for calibration.")

if __name__ == "__main__":

    # --- 1. SET LOGGING LEVEL FOR PCSE ---
    # This is your fix to silence the "Starting crop..." spam
    logging.getLogger('pcse').setLevel(logging.WARNING)
    # --- END OF FIX ---

    # 2. Load all the data into the global `calibration_tasks` list
    load_all_calibration_data()

    if not calibration_tasks or not base_cropdata_yaml:
        logging.error("FATAL: Data loading failed. Cannot run optimizer.")
        sys.exit(1)

    # 3. Start the Optuna optimization
    logging.info("=" * 70)
    n_trials = CONFIG['OPTIMIZATION']['N_TRIALS']
    logging.info(f"Starting Optuna optimization... This will run {n_trials} trials.")
    logging.info("This may take several hours. Grab a coffee.")
    logging.info("=" * 70)

    study = optuna.create_study(direction='minimize')

    # --- 4. SETUP TQDM PROGRESS BAR ---
    pbar = tqdm(total=n_trials, desc="Calibrating Crop Gains")


    def tqdm_callback(study, trial):
        """This function is called by Optuna after each trial."""
        pbar.update(1)


    # --- END TQDM SETUP ---

    try:
        # We pass our new callback to the optimizer
        study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])

    except KeyboardInterrupt:
        logging.warning("Optimization stopped early by user.")
    except Exception as e:
        logging.error(f"Optimization failed with error: {e}", exc_info=True)
    finally:
        # Ensure the progress bar is closed
        pbar.close()

    # 5. Print the final results
    print("\n" + "=" * 80)
    print("           OPTIMIZATION FINISHED           ")
    print("=" * 80)

    if study.best_trial is None:
        logging.error("No trials were completed successfully. Cannot show results.")
    else:
        print(f"Best score (minimum error): {study.best_value}")
        print("Best parameters found:")

        # Format the output perfectly to be pasted into config.py
        print("\n\n--- COPY AND PASTE THIS INTO YOUR src/config.py ---")
        print(f"'GENETIC_GAIN_PARAMS': {{")
        print(f"    'START_YEAR': {GAIN_START_YEAR},")

        # Get the best parameters
        best_params = study.best_params

        # RUE
        print(
            f"    'RUE': {{'base': {best_params.get('RUE_base', 1.9)}, 'gain_rate': {best_params.get('RUE_gain', 0.0)}}},")
        # TSUM1
        print(
            f"    'TSUM1': {{'base': {best_params.get('TSUM1_base', 700)}, 'gain_rate': {best_params.get('TSUM1_gain', 0.0)}}},")
        # AMAX
        print(
            f"    'AMAX': {{'base': {best_params.get('AMAX_base', 35.0)}, 'gain_rate': {best_params.get('AMAX_gain', 0.0)}}},")

        print(f"}}")
        print("----------------------------------------------------\n\n")