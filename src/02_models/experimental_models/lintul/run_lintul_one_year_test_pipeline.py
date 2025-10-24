# File: run_lintul_one_year_test_pipeline.py
# Description: A consolidated script to run the full one-year test pipeline.
# It loads data, trains the weather generator, runs a historical simulation,
# runs an ensemble forecast simulation, and analyzes the final results.

import datetime
import yaml
import pandas as pd
import numpy as np
import os
import logging
from pcse.util import penman_monteith
from numpy.f2py.auxfuncs import throw_error

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')

from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

# --- PCSE Imports ---
from pcse.models import Wofost72_WLP_FD  # <-- FIX: Use WOFOST 7.2
from pcse.base import ParameterProvider, WeatherDataProvider

TEST_YEAR = 2018

# --- Input File Paths ---
HISTORICAL_DAILY_WEATHER_PATH = 'data/02_intermediate/historical_daily_weather_era5_2018_TEST.csv'
STATIC_FEATURES_PATH = 'data/05_model_input/stage1_preseason_features.csv'
SEAS5_MEMBER_FEATURES_PATH = 'data/02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv'
CROP_YAML_PATH = 'data/01_raw/sugarbeet.yaml'

# --- Output File Paths ---
OUTPUT_DIR = 'data/06_model_output/one_year_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FINAL_COMPARISON_CSV_PATH = os.path.join(OUTPUT_DIR, f'final_comparison_{TEST_YEAR}_TEST.csv')


# ==============================================================================
# PARAMETER DICTIONARY CLASS FOR PCSE
# ==============================================================================

class ParameterDict(dict):
    """
    Dictionary that mimics PCSE parameter objects with add_variable method.
    It includes a custom .copy() method to ensure its type is preserved
    when copied internally by PCSE.
    """

    def add_variable(self, name, value, description=""):
        """Add a variable to the dictionary."""
        self[name] = value

    def __getattr__(self, name):
        """Allow attribute-style access to dict items."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Allow attribute-style setting of dict items."""
        self[name] = value

    def copy(self):
        """
        Overrides the default dict.copy() method.
        Ensures that the copied object is also a ParameterDict instance.
        """
        return ParameterDict(self)


# ==============================================================================
# SIMPLE WEATHER DATA CONTAINER FOR PCSE
# ==============================================================================
class SimpleWeatherDataProvider(WeatherDataProvider):
    """
    Simple weather data provider for PCSE LINTUL3.
    Inherits from WeatherDataProvider base class.
    """

    def __init__(self, weather_df, latitude=52.0, longitude=10.0):
        WeatherDataProvider.__init__(self)

        self.latitude = latitude
        self.longitude = longitude
        self.elevation = 50.0
        self.description = "Custom weather data from local CSV"
        self.angstA = 0.25
        self.angstB = 0.5

        weather_df = weather_df.copy()
        weather_df['date'] = pd.to_datetime(weather_df['date'])

        self.store = {}
        logging.info(f"[WEATHER] Initializing provider with {len(weather_df)} rows")
        logging.info(f"[WEATHER] Date range: {weather_df['date'].min()} to {weather_df['date'].max()}")

        for _, row in weather_df.iterrows():
            try:
                day = row['date'].date()

                # Basic weather data
                tmin = float(row['tmin'])
                tmax = float(row['tmax'])
                irrad = float(row['srad']) / 10.0  # Convert to kJ/m²/day
                precip = float(row['precip']) / 10000.0  # Convert to mm
                # Use default/placeholder values if needed, or get them from your CSV
                vap = 10.0  # Placeholder kPa
                wind = 2.0  # Placeholder m/s

                # --- FIX: Calculate ET0, E0, ES0 ---
                # --- FIX: Calculate ET0 (assuming function returns only ET0 in mm/day) ---
                et0_mm = penman_monteith(
                    day,
                    self.latitude,
                    self.elevation,
                    tmin,
                    tmax,
                    irrad,
                    vap,
                    wind
                )

                # --- FIX: Convert ET0 to cm/day and assign reasonable values to E0, ES0 ---
                et0 = et0_mm / 10.0  # Convert ET0 to cm/day
                # Simple approximation: Assume E0 and ES0 are similar to ET0 initially
                # More complex calculations exist, but this is often sufficient.
                e0 = et0
                es0 = et0

                data_dict = ParameterDict({
                    'DAY': day,
                    'LAT': self.latitude,
                    'TMIN': tmin,
                    'TMAX': tmax,
                    'RAIN': precip,
                    'IRRAD': irrad,
                    'VAP': vap,
                    'WIND': wind,
                    'SNOWDEPTH': 0.0,
                    # --- FIX: Add ET variables in cm/day ---
                    'E0': e0,
                    'ES0': es0,
                    'ET0': et0
                })

                self.store[(day, 0)] = data_dict

                # Log first 3 days for verification
                if day.day <= 3:
                    logging.info(f"[WEATHER] {day}: TMIN={data_dict['TMIN']:.1f}°C, "
                                 f"TMAX={data_dict['TMAX']:.1f}°C, RAIN={data_dict['RAIN']:.1f}mm, "
                                 f"IRRAD={data_dict['IRRAD']:.1f}kJ/m²/day")

            except Exception as e:
                logging.error(f"[WEATHER] CRITICAL: Failed processing row for date: {row.get('date')}")
                logging.error(f"[WEATHER] Error: {e}")
                logging.error(f"[WEATHER] Row data: {row}")
                raise e

        logging.info(f"[WEATHER] Successfully loaded {len(self.store)} days into weather provider")


# ==============================================================================
# WEATHER GENERATOR CLASS
# ==============================================================================

class WeatherGenerator:
    """
    Stochastic weather generator conditioned on monthly climate forecasts.
    """

    def __init__(self):
        self.stats = defaultdict(dict)
        self.PRECIP_THRESHOLD_MM = 0.3

    def fit(self, daily_df: pd.DataFrame):
        """Learn monthly statistics from historical data."""
        logging.info("[WEATHER_GEN] Fitting Weather Generator...")
        print("[WEATHER_GEN] Fitting Weather Generator...")
        daily_df = daily_df.copy()
        daily_df['month'] = daily_df['date'].dt.month
        daily_df['is_wet'] = (daily_df['precip'] > self.PRECIP_THRESHOLD_MM).astype(int)

        total_groups = len(daily_df.groupby(['district_no', 'month']))
        logging.info(f"[WEATHER_GEN] Processing {total_groups} district-month combinations")
        print(f"[WEATHER_GEN] Processing {total_groups} district-month combinations")

        for (district_no, month), group in tqdm(daily_df.groupby(['district_no', 'month']),
                                                desc="Learning Weather Patterns", disable=False):
            # Markov chain transitions for precipitation
            p01 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 1)).sum()
            p00 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 0)).sum()
            p11 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 1)).sum()
            p10 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 0)).sum()

            prob_wet_given_dry = p01 / (p01 + p00) if (p01 + p00) > 0 else 0.1
            prob_wet_given_wet = p11 / (p11 + p10) if (p11 + p10) > 0 else 0.5

            wet_day_precip = group[group['is_wet'] == 1]['precip']

            self.stats[(district_no, month)] = {
                'p_wet_given_dry': prob_wet_given_dry,
                'p_wet_given_wet': prob_wet_given_wet,
                'precip_wet_day_mean': wet_day_precip.mean() if len(wet_day_precip) > 0 else 1.0,
                'precip_wet_day_std': wet_day_precip.std() if len(wet_day_precip) > 1 else 0.5,
                'precip_mean': group['precip'].mean(),
                'tmin_mean': group['tmin'].mean(),
                'tmin_std': max(group['tmin'].std(), 0.5),
                'tmax_mean': group['tmax'].mean(),
                'tmax_std': max(group['tmax'].std(), 0.5),
                'srad_mean': group['srad'].mean(),
                'srad_std': max(group['srad'].std(), 0.5),
            }

        logging.info(f"[WEATHER_GEN] Learned statistics for {len(self.stats)} district-month pairs")
        print(f"[WEATHER_GEN] Learned statistics for {len(self.stats)} district-month pairs\n")

    def generate(self, district_no: str, start_date_str: str, end_date_str: str, monthly_anomalies: dict):
        """Generate synthetic daily weather with bias correction."""
        dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
        generated_data = []
        yesterday_was_wet = np.random.rand() < 0.5

        for date in dates:
            month = date.month
            key = (str(district_no).zfill(5), month)

            if key not in self.stats:
                continue

            month_stats = self.stats[key]

            # Precipitation (Markov chain)
            transition_prob = (month_stats['p_wet_given_wet'] if yesterday_was_wet
                               else month_stats['p_wet_given_dry'])
            today_is_wet = np.random.rand() < transition_prob

            precip = 0.0
            if today_is_wet:
                precip = np.random.normal(month_stats['precip_wet_day_mean'],
                                          month_stats['precip_wet_day_std'])
                precip = max(0, precip)

            # Temperature
            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std'])
            tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin:
                tmax = tmin + abs(np.random.normal(0, 1.0))

            # Solar radiation
            srad = np.random.normal(month_stats['srad_mean'], month_stats['srad_std'])
            srad = max(0.1, srad)

            generated_data.append({
                'date': date,
                'tmin': tmin,
                'tmax': tmax,
                'precip': precip,
                'srad': srad
            })
            yesterday_was_wet = today_is_wet

        if not generated_data:
            return pd.DataFrame()

        synthetic_df = pd.DataFrame(generated_data)
        synthetic_df['month'] = synthetic_df['date'].dt.month

        # Bias correction to match forecast anomalies
        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month
            key = (str(district_no).zfill(5), month)

            if key not in self.stats:
                continue

            # Temperature: additive correction
            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0)
            synth_tmean = (synthetic_df.loc[month_mask, 'tmin'].mean() +
                           synthetic_df.loc[month_mask, 'tmax'].mean()) / 2
            hist_tmean = (self.stats[key]['tmin_mean'] + self.stats[key]['tmax_mean']) / 2
            temp_correction = (hist_tmean + temp_anomaly) - synth_tmean

            synthetic_df.loc[month_mask, 'tmin'] += temp_correction
            synthetic_df.loc[month_mask, 'tmax'] += temp_correction

            # Precipitation: multiplicative correction
            precip_anomaly_factor = 1.0 + monthly_anomalies.get(f'precip_anomaly_{month}', 0)
            synth_precip = synthetic_df.loc[month_mask, 'precip'].sum()
            hist_precip = self.stats[key]['precip_mean'] * month_mask.sum()
            target_precip = hist_precip * precip_anomaly_factor

            if synth_precip > 0:
                synthetic_df.loc[month_mask, 'precip'] *= (target_precip / synth_precip)

        return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad']]

# ==============================================================================
# PART 1: DATA LOADING AND MODEL SETUP
# ==============================================================================

def load_data_and_setup_model(year):
    """Load all input data and prepare LINTUL3 parameters."""
    logging.info("=" * 70)
    logging.info("[SETUP] Loading input data and setting up model")
    logging.info("=" * 70)

    try:
        df_static = pd.read_csv(STATIC_FEATURES_PATH)
        df_daily_hist = pd.read_csv(HISTORICAL_DAILY_WEATHER_PATH, parse_dates=['date'])
        df_seas5 = pd.read_csv(SEAS5_MEMBER_FEATURES_PATH)

        logging.info(f"[DATA] Raw data loaded:")
        logging.info(f"[DATA]   - Static features: {len(df_static)} rows")
        logging.info(f"[DATA]   - Historical weather: {len(df_daily_hist)} rows")
        logging.info(f"[DATA]   - SEAS5 forecasts: {len(df_seas5)} rows")

    except FileNotFoundError as e:
        logging.error(f"[DATA] FATAL: Input file not found. {e}")
        return None, None, None, None

    # Filter for test year
    df_static = df_static[df_static['year'] == year].copy()
    df_seas5 = df_seas5[df_seas5['year'] == year].copy()

    # Standardize district codes
    logging.info("[DATA] Standardizing district_no columns to zero-padded strings")
    df_static['district_no'] = pd.to_numeric(df_static['district_no'], errors='coerce').astype('Int64').astype(
        str).str.zfill(5)
    df_daily_hist['district_no'] = pd.to_numeric(df_daily_hist['district_no'], errors='coerce').astype('Int64').astype(
        str).str.zfill(5)
    df_seas5['district_no'] = pd.to_numeric(df_seas5['district_no'], errors='coerce').astype('Int64').astype(
        str).str.zfill(5)

    logging.info(f"[DATA] Filtered for year {year}:")
    logging.info(f"[DATA]   - Static records: {len(df_static)}")
    logging.info(f"[DATA]   - Unique districts in static: {df_static['district_no'].nunique()}")
    logging.info(f"[DATA]   - SEAS5 records: {len(df_seas5)}")
    logging.info(
        f"[DATA]   - SEAS5 members per district: {len(df_seas5) // df_seas5['district_no'].nunique() if len(df_seas5) > 0 else 0}")

    # Log sample data
    logging.info("\n[DATA] Sample static features (first 3 districts):")
    sample_static = df_static[['district_no', 'kreisYield']].head(3)
    for _, row in sample_static.iterrows():
        logging.info(f"[DATA]   District {row['district_no']}: Actual yield = {row['kreisYield']:.2f} kg/ha")

    logging.info("\n[DATA] Sample SEAS5 forecast (first record):")
    if len(df_seas5) > 0:
        first_seas5 = df_seas5.iloc[0]
        logging.info(f"[DATA]   District {first_seas5['district_no']}, Member {first_seas5.get('member', 'N/A')}")
        logging.info(f"[DATA]   Spring temp anomaly: {first_seas5.get('spring_temp_anomaly_forecast', 'N/A'):.3f}°C")
        logging.info(f"[DATA]   Spring precip anomaly: {first_seas5.get('spring_precip_anomaly_forecast', 'N/A'):.3f}")
        logging.info(f"[DATA]   Summer temp anomaly: {first_seas5.get('summer_temp_anomaly_forecast', 'N/A'):.3f}°C")
        logging.info(f"[DATA]   Summer precip anomaly: {first_seas5.get('summer_precip_anomaly_forecast', 'N/A'):.3f}")

    try:
        # --- Load and parse crop YAML ---
        logging.info("\n[PARAMS] Loading crop parameters from YAML")
        with open(CROP_YAML_PATH, 'r') as f:
            full_yaml = yaml.safe_load(f)

        if 'CropParameters' not in full_yaml:
            raise ValueError("No CropParameters section found in YAML file")

        crop_params = full_yaml['CropParameters']
        available_varieties = [k for k in crop_params.keys() if not k.startswith('Generic')]

        if not available_varieties:
            raise ValueError("No crop varieties found in YAML")

        variety_name = available_varieties[0]
        variety_data = crop_params[variety_name]
        logging.info(f"[PARAMS] Using crop variety: {variety_name}")

        # --- Create ParameterDicts for all parameter groups ---
        cropdata = ParameterDict()
        soildata = ParameterDict()
        sitedata = ParameterDict()

        # --- Helper: recursively add variables ---
        def add_parameters_recursive(param_dict, data):
            """Recursively add all parameters (including nested dicts).
               Assumes YAML format [value, description, units] or nested dicts.
            """
            for key, val in data.items():
                if isinstance(val, dict):
                    # Recurse into nested dictionaries (like Metadata)
                    add_parameters_recursive(param_dict, val)
                elif isinstance(val, list) and len(val) > 0:
                    # --- FIX: ALWAYS take the first element as the value ---
                    # Handles both simple values [3.0, ...] and tables [[[...],[...]], ...]
                    value = val[0]
                    try:
                        param_dict.add_variable(key, value)
                    except Exception as e:
                        logging.warning(f"[PARAMS] Skipping crop parameter '{key}' during add_variable: {e}")
                # Optional: Handle cases where 'val' is not a dict or list (if any)
                # else:
                #     logging.debug(f"[PARAMS] Skipping non-list/non-dict item: {key}")

        # --- Fill crop parameters recursively ---
        generic_data = {}
        if 'GenericC3' in crop_params:
            generic_data.update(crop_params['GenericC3'])
        if 'Generic' in crop_params:
            generic_data.update(crop_params['Generic'])

        if generic_data:
            logging.info("[PARAMS] Loading generic crop parameters...")
            add_parameters_recursive(cropdata, generic_data)

        # --- Load variety-specific parameters (will overwrite generic) ---
        logging.info(f"[PARAMS] Loading parameters for variety: {variety_name}")
        add_parameters_recursive(cropdata, variety_data)
        logging.info(f"[PARAMS] Loaded {len(cropdata)} crop parameters (flattened)")

        key_params = ['TSUM1', 'TSUM2', 'TSUMEM', 'RGRLAI', 'LAICR']
        logging.info("[PARAMS] Key crop parameters:")
        for param in key_params:
            if param in cropdata:
                logging.info(f"[PARAMS]   {param} = {cropdata[param]}")

        # --- Define soil parameters ---
        soildata.add_variable('SMW', 0.10)  # Wilting point
        soildata.add_variable('SMFCF', 0.30)  # Field capacity
        soildata.add_variable('SM0', 0.40)  # Saturation
        soildata.add_variable('CRAIRC', 0.06)  # Critical air content
        soildata.add_variable('RDMSOL', 150.0)  # <-- FIX: Add Max Soil Rooting Depth (cm)
        soildata.add_variable('KSUB', 10.0)  # Max percolation rate subsoil (cm/day)
        soildata.add_variable('SOPE', 10.0)  # Max percolation rate root zone (cm/day)

        logging.info("[PARAMS] Soil parameters:")
        logging.info(f"[PARAMS]   SMW (Wilting point) = {soildata['SMW']}")
        logging.info(f"[PARAMS]   SMFCF (Field capacity) = {soildata['SMFCF']}")
        logging.info(f"[PARAMS]   SM0 (Saturation) = {soildata['SM0']}")

        # --- Define site parameters ---
        sitedata.add_variable('LAT', 52.0)
        sitedata.add_variable('LON', 10.0)
        sitedata.add_variable('ELEV', 50.0)
        sitedata.add_variable('WAV', 100.0)  # Initial water amount (mm)
        sitedata.add_variable('SMLIM', 0.30)  # Initial soil moisture (set to SMFCF)
        sitedata.add_variable('IFUNRN', 0.0)  # <-- FIX: Add IFUNRN flag
        sitedata.add_variable('NOTINF', 0.0)  # <-- FIX: Add NOTINF flag
        sitedata.add_variable('SSI', 0.0)  # <-- FIX: Add Initial Surface Storage (cm)
        sitedata.add_variable('SSMAX', 0.0)  # <-- FIX: Add Max Surface Storage (cm)

        logging.info("[PARAMS] Site parameters:")
        logging.info(f"[PARAMS]   Latitude = {sitedata['LAT']}°")
        logging.info(f"[PARAMS]   Longitude = {sitedata['LON']}°")
        logging.info(f"[PARAMS]   Elevation = {sitedata['ELEV']} m")

        # --- Sanity check ---
        assert isinstance(cropdata, ParameterDict)
        assert isinstance(soildata, ParameterDict)
        assert isinstance(sitedata, ParameterDict)

        # --- Build ParameterProvider ---
        parameters = ParameterProvider(
            cropdata=cropdata,
            soildata=soildata,
            sitedata=sitedata
        )

        logging.info("[PARAMS] ✓ Parameters loaded successfully\n")

    except Exception as e:
        logging.error(f"[PARAMS] FATAL: Could not load parameters: {e}", exc_info=True)
        return None, None, None, None

    return df_static, df_daily_hist, df_seas5, parameters


# ==============================================================================
# PART 2: HISTORICAL SIMULATION (PERFECT WEATHER)
# ==============================================================================

def run_historical_simulation(df_static, df_daily_hist, parameters, year):
    """Run LINTUL3 for each district using observed weather."""
    logging.info("=" * 70)
    logging.info(f"[HISTORICAL] Running Historical Simulation for {year}")
    logging.info("=" * 70)

    # CRITICAL: Filter historical weather by year first
    df_daily_hist_year = df_daily_hist[df_daily_hist['date'].dt.year == year].copy()
    logging.info(f"[HISTORICAL] Filtered weather data to year {year}: {len(df_daily_hist_year)} records")

    if df_daily_hist_year.empty:
        logging.error(f"[HISTORICAL] FATAL: No weather data found for year {year}!")
        return pd.DataFrame()

    logging.info(
        f"[HISTORICAL] Weather date range: {df_daily_hist_year['date'].min().date()} to {df_daily_hist_year['date'].max().date()}")
    logging.info(f"[HISTORICAL] Unique districts in weather data: {df_daily_hist_year['district_no'].nunique()}")

    results = []

    # Disable tqdm for first district to see logs
    iterator = enumerate(df_static.iterrows())

    for idx, (_, row) in iterator:
        district_no = row['district_no']

        # Log first district in detail (before tqdm starts suppressing output)
        if idx == 0:
            print(f"\n[HISTORICAL] Processing first district: {district_no}")
            print(f"[HISTORICAL] Actual yield: {row['kreisYield']:.2f} kg/ha")

        # Get weather data for this district AND year
        weather_df = df_daily_hist_year[df_daily_hist_year['district_no'] == district_no].copy()
        if weather_df.empty:
            print(f"[HISTORICAL] No weather for district {district_no} in year {year}")
            continue

        if idx == 0:
            print(f"[HISTORICAL] Found {len(weather_df)} days of weather data")
            print(
                f"[HISTORICAL] Weather date range for district: {weather_df['date'].min().date()} to {weather_df['date'].max().date()}")

        try:
            # Create weather provider
            if idx == 0:
                print(f"[HISTORICAL] Creating weather provider...")
            weather_provider = SimpleWeatherDataProvider(weather_df)

            # Agromanagement: planting March 15, harvest October 20
            crop_start = datetime.date(year, 3, 15)
            crop_end = datetime.date(year, 10, 20)

            if idx == 0:
                print(f"[HISTORICAL] Crop calendar: {crop_start} to {crop_end}")
                print(f"[HISTORICAL] Starting LINTUL3 model...")

            crop_calendar = ParameterDict({
                'crop_start_date': crop_start,
                'crop_start_type': 'emergence',
                'crop_end_date': crop_end,
                'crop_end_type': 'harvest',
                'max_duration': 250
            })

            agromanagement = [{
                crop_start: ParameterDict({
                    'CropCalendar': crop_calendar,
                    'TimedEvents': None,
                    'StateEvents': None
                })
            }]

            # Run Wofost72_WLP_FD
            model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)

            if idx == 0:
                print(f"[HISTORICAL] Running model simulation...")

            model.run_till_terminate()

            # Extract yield
            output = model.get_output()
            if output:
                simulated_yield = output[-1]['TWSO']  # kg/ha
                if idx == 0:
                    print(f"[HISTORICAL] Simulated yield: {simulated_yield:.2f} kg/ha")
                    print(f"[HISTORICAL] Simulation completed with {len(output)} output records")
            else:
                simulated_yield = np.nan
                if idx == 0:
                    print(f"[HISTORICAL] No output from simulation")

            results.append({
                'year': year,
                'district_no': district_no,
                'actual_yield': row['kreisYield'],
                'lintul_yield_perfect_weather': simulated_yield
            })

        except Exception as e:
            print(f"[HISTORICAL] ERROR for district {district_no}: {e}")
            if idx == 0:
                import traceback
                print(f"[HISTORICAL] Full traceback:")
                traceback.print_exc()
            results.append({
                'year': year,
                'district_no': district_no,
                'actual_yield': row['kreisYield'],
                'lintul_yield_perfect_weather': np.nan
            })

        # Show progress only after first district
        if idx == 0:
            print(
                f"[HISTORICAL] First district complete. Continuing with tqdm for remaining {len(df_static) - 1} districts...")
            # Now wrap remaining iterations with tqdm
            remaining = list(enumerate(df_static.iloc[1:].iterrows(), start=1))
            for idx, (_, row) in tqdm(remaining, desc="Historical Sim", initial=1, total=len(df_static)):
                district_no = row['district_no']
                weather_df = df_daily_hist_year[df_daily_hist_year['district_no'] == district_no].copy()
                if weather_df.empty:
                    continue

                try:
                    weather_provider = SimpleWeatherDataProvider(weather_df)
                    crop_start = datetime.date(year, 3, 15)
                    crop_end = datetime.date(year, 10, 20)
                    crop_calendar = ParameterDict({
                        'crop_start_date': crop_start,
                        'crop_start_type': 'emergence',
                        'crop_end_date': crop_end,
                        'crop_end_type': 'harvest',
                        'max_duration': 250
                    })
                    agromanagement = [{
                        crop_start: ParameterDict({
                            'CropCalendar': crop_calendar,
                            'TimedEvents': None,
                            'StateEvents': None
                        })
                    }]
                    model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)
                    model.run_till_terminate()
                    output = model.get_output()
                    if output:
                        simulated_yield = output[-1]['TWSO']
                    else:
                        simulated_yield = np.nan
                    results.append({
                        'year': year,
                        'district_no': district_no,
                        'actual_yield': row['kreisYield'],
                        'lintul_yield_perfect_weather': simulated_yield
                    })
                except Exception as e:
                    results.append({
                        'year': year,
                        'district_no': district_no,
                        'actual_yield': row['kreisYield'],
                        'lintul_yield_perfect_weather': np.nan
                    })
            break  # Exit the outer loop since we've processed all

    logging.info(f"[HISTORICAL] Completed {len(results)} simulations\n")
    print(f"[HISTORICAL] Completed {len(results)} simulations\n")
    return pd.DataFrame(results)


# ==============================================================================
# PART 3: ENSEMBLE FORECAST SIMULATION
# ==============================================================================

def run_forecast_simulation(df_static, df_daily_hist, df_seas5, parameters, year):
    """Run ensemble LINTUL3 simulations using synthetic weather."""
    logging.info("=" * 70)
    logging.info(f"[FORECAST] Running Forecast Simulation for {year}")
    logging.info("=" * 70)

    # Train weather generator on historical data (can use all years or just training years)
    # For this test, we'll use all available historical data to learn patterns
    wg = WeatherGenerator()
    logging.info(f"[FORECAST] Training weather generator on {len(df_daily_hist)} historical records")
    logging.info(f"[FORECAST] Historical data years: {sorted(df_daily_hist['date'].dt.year.unique())}")
    wg.fit(df_daily_hist)

    forecast_results = []
    district_count = 0

    for district_no, group in tqdm(df_seas5.groupby('district_no'), desc="Forecast Sim"):
        district_count += 1

        # Log first district in detail
        if district_count == 1:
            logging.info(f"\n[FORECAST] Processing first district: {district_no}")
            logging.info(f"[FORECAST] Ensemble size: {len(group)} members")

        ensemble_yields = []

        # Run simulation for each SEAS5 member
        for member_idx, (_, member) in enumerate(group.iterrows()):

            # Build monthly anomaly dictionary
            monthly_anomalies = {}
            for month in range(3, 11):
                if month <= 6:  # Spring (Mar-Jun)
                    temp_anom = member.get('spring_temp_anomaly_forecast', 0)
                    precip_anom = member.get('spring_precip_anomaly_forecast', 0)
                else:  # Summer (Jul-Oct)
                    temp_anom = member.get('summer_temp_anomaly_forecast', 0)
                    precip_anom = member.get('summer_precip_anomaly_forecast', 0)

                monthly_anomalies[f'temp_anomaly_{month}'] = temp_anom
                monthly_anomalies[f'precip_anomaly_{month}'] = precip_anom

            # Log first member
            if district_count == 1 and member_idx == 0:
                logging.info(f"[FORECAST] First member anomalies:")
                logging.info(f"[FORECAST]   Spring: T={temp_anom:.3f}°C, P={precip_anom:.3f}")
                logging.info(f"[FORECAST]   Summer: T={member.get('summer_temp_anomaly_forecast', 0):.3f}°C, "
                             f"P={member.get('summer_precip_anomaly_forecast', 0):.3f}")

            # Generate synthetic weather
            synth_weather = wg.generate(
                district_no,
                f'{year}-03-01',
                f'{year}-10-31',
                monthly_anomalies
            )

            if synth_weather.empty:
                ensemble_yields.append(np.nan)
                continue

            if district_count == 1 and member_idx == 0:
                logging.info(f"[FORECAST] Generated {len(synth_weather)} days of synthetic weather")
                logging.info(f"[FORECAST] Sample synthetic weather (first 3 days):")
                for i in range(min(3, len(synth_weather))):
                    row = synth_weather.iloc[i]
                    logging.info(f"[FORECAST]   {row['date'].date()}: "
                                 f"TMIN={row['tmin']:.1f}°C, TMAX={row['tmax']:.1f}°C, "
                                 f"PRECIP={row['precip']:.1f}mm, SRAD={row['srad']:.1f}J/m²/day")

            try:
                # Create weather provider
                weather_provider = SimpleWeatherDataProvider(synth_weather)

                # Agromanagement
                crop_start = datetime.date(year, 3, 15)
                crop_end = datetime.date(year, 10, 20)

                crop_calendar = ParameterDict({
                    'crop_start_date': crop_start,
                    'crop_start_type': 'emergence',
                    'crop_end_date': crop_end,
                    'crop_end_type': 'harvest',
                    'max_duration': 250
                })

                agromanagement = [{
                    crop_start: ParameterDict({
                        'CropCalendar': crop_calendar,
                        'TimedEvents': None,
                        'StateEvents': None
                    })
                }]

                # Run model
                model = Wofost72_WLP_FD(parameters, weather_provider, agromanagement)
                model.run_till_terminate()

                output = model.get_output()
                if output:
                    member_yield = output[-1]['TWSO']
                    ensemble_yields.append(member_yield)

                    if district_count == 1 and member_idx == 0:
                        logging.info(f"[FORECAST] First member yield: {member_yield:.2f} kg/ha")
                else:
                    ensemble_yields.append(np.nan)

            except Exception as e:
                logging.error(f"[FORECAST] Error for district {district_no}, member {member_idx}: {e}")
                ensemble_yields.append(np.nan)

        # Compute ensemble mean and std
        valid_yields = [y for y in ensemble_yields if not np.isnan(y)]

        if valid_yields:
            forecast_mean = np.mean(valid_yields)
            forecast_std = np.std(valid_yields)

            if district_count == 1:
                logging.info(f"[FORECAST] Ensemble statistics:")
                logging.info(f"[FORECAST]   Valid members: {len(valid_yields)}/{len(ensemble_yields)}")
                logging.info(f"[FORECAST]   Mean yield: {forecast_mean:.2f} kg/ha")
                logging.info(f"[FORECAST]   Std dev: {forecast_std:.2f} kg/ha")
        else:
            forecast_mean = np.nan
            forecast_std = np.nan

        forecast_results.append({
            'year': year,
            'district_no': district_no,
            'lintul_yield_forecast_weather': forecast_mean,
            'forecast_uncertainty_std': forecast_std
        })

    logging.info(f"[FORECAST] Completed {len(forecast_results)} district ensembles\n")
    return pd.DataFrame(forecast_results)


# ==============================================================================
# PART 4: ANALYSIS AND VISUALIZATION
# ==============================================================================

def analyze_and_plot_results(df_hist, df_fcst, year):
    """Merge results and create analysis plots."""
    logging.info("=" * 70)
    logging.info("[ANALYSIS] Analyzing Results")
    logging.info("=" * 70)

    # Merge
    df_final = pd.merge(df_hist, df_fcst, on=['year', 'district_no'])
    logging.info(f"[ANALYSIS] Merged data: {len(df_final)} districts")

    # Check for missing values before dropping
    missing_actual = df_final['actual_yield'].isna().sum()
    missing_perfect = df_final['lintul_yield_perfect_weather'].isna().sum()
    missing_forecast = df_final['lintul_yield_forecast_weather'].isna().sum()

    logging.info(f"[ANALYSIS] Missing values before cleanup:")
    logging.info(f"[ANALYSIS]   Actual yield: {missing_actual}")
    logging.info(f"[ANALYSIS]   Perfect weather simulation: {missing_perfect}")
    logging.info(f"[ANALYSIS]   Forecast weather simulation: {missing_forecast}")

    df_final = df_final.dropna()

    if df_final.empty:
        logging.error("[ANALYSIS] No valid results to analyze!")
        return

    logging.info(f"[ANALYSIS] Valid results after cleanup: {len(df_final)} districts")

    # Save
    df_final.to_csv(FINAL_COMPARISON_CSV_PATH, index=False)
    logging.info(f"[ANALYSIS] ✓ Results saved to {FINAL_COMPARISON_CSV_PATH}")

    # Convert kg/ha to dt/ha (1 dt = 100 kg)
    df_final['actual_yield_dt'] = df_final['actual_yield'] / 100
    df_final['perfect_yield_dt'] = df_final['lintul_yield_perfect_weather'] / 100
    df_final['forecast_yield_dt'] = df_final['lintul_yield_forecast_weather'] / 100

    print("\n" + "=" * 70)
    print(f"FINAL ANALYSIS FOR {year}")
    print("=" * 70)
    print("\n--- Sample Results (dt/ha) ---")
    print(df_final[['district_no', 'actual_yield_dt', 'perfect_yield_dt', 'forecast_yield_dt']].head())

    print("\n--- Statistical Summary (dt/ha) ---")
    print(df_final[['actual_yield_dt', 'perfect_yield_dt', 'forecast_yield_dt']].describe())

    # Metrics
    print("\n--- Performance Metrics ---")
    mae_perfect = mean_absolute_error(df_final['actual_yield_dt'], df_final['perfect_yield_dt'])
    mae_forecast = mean_absolute_error(df_final['actual_yield_dt'], df_final['forecast_yield_dt'])
    r2_perfect = r2_score(df_final['actual_yield_dt'], df_final['perfect_yield_dt'])
    r2_forecast = r2_score(df_final['actual_yield_dt'], df_final['forecast_yield_dt'])

    print(f"  Perfect Weather:  MAE = {mae_perfect:.2f} dt/ha, R² = {r2_perfect:.3f}")
    print(f"  Forecast Weather: MAE = {mae_forecast:.2f} dt/ha, R² = {r2_forecast:.3f}")

    logging.info(f"\n[ANALYSIS] Performance Summary:")
    logging.info(f"[ANALYSIS]   Perfect Weather:  MAE = {mae_perfect:.2f} dt/ha, R² = {r2_perfect:.3f}")
    logging.info(f"[ANALYSIS]   Forecast Weather: MAE = {mae_forecast:.2f} dt/ha, R² = {r2_forecast:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'LINTUL3 Performance Comparison - {year}', fontsize=14, fontweight='bold')

    min_val = df_final[['actual_yield_dt', 'perfect_yield_dt', 'forecast_yield_dt']].min().min()
    max_val = df_final[['actual_yield_dt', 'perfect_yield_dt', 'forecast_yield_dt']].max().max()

    # Perfect weather
    axes[0].scatter(df_final['actual_yield_dt'], df_final['perfect_yield_dt'], alpha=0.6, s=40)
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
    axes[0].set_title(f'Perfect Weather\nMAE={mae_perfect:.2f}, R²={r2_perfect:.3f}')
    axes[0].set_xlabel('Actual Yield (dt/ha)')
    axes[0].set_ylabel('Simulated Yield (dt/ha)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Forecast weather
    axes[1].scatter(df_final['actual_yield_dt'], df_final['forecast_yield_dt'],
                    alpha=0.6, s=40, color='orange')
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
    axes[1].set_title(f'Forecast Weather\nMAE={mae_forecast:.2f}, R²={r2_forecast:.3f}')
    axes[1].set_xlabel('Actual Yield (dt/ha)')
    axes[1].set_ylabel('Simulated Yield (dt/ha)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'results_{year}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"[ANALYSIS] ✓ Plot saved to {plot_path}")
    plt.show()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    logging.info("=" * 70)
    logging.info(f"LINTUL3 ONE-YEAR TEST PIPELINE - {TEST_YEAR}")
    logging.info("=" * 70)

    # Load data and setup
    df_static, df_daily_hist, df_seas5, parameters = load_data_and_setup_model(TEST_YEAR)

    print("\n" + "=" * 70)
    print("--- DATA VALIDATION (MAIN BLOCK) ---")

    if parameters is None:
        print("[FATAL] 'parameters' is None. Setup failed inside load_data_and_setup_model.")
    else:
        print("[SUCCESS] 'parameters' object was created successfully.")

    print(f"\n[df_static] Shape: {df_static.shape}")
    if not df_static.empty:
        print(f"[df_static] Head:\n{df_static.head()}")

    print(f"\n[df_seas5] Shape: {df_seas5.shape}")
    if not df_seas5.empty:
        print(f"[df_seas5] Head:\n{df_seas5.head()}")

    print(f"\n[df_daily_hist] Shape: {df_daily_hist.shape}")
    if not df_daily_hist.empty:
        print(f"[df_daily_hist] Head:\n{df_daily_hist.head()}")
        print(f"\n[CRITICAL CHECK] Historical Weather Date Range:")
        print(f"  First available date: {df_daily_hist['date'].min().date()}")  # <--- FIXED (removed timestamp)
        print(f"  Last available date:  {df_daily_hist['date'].max().date()}")  # <--- FIXED (removed timestamp)
    else:
        print("[FATAL] df_daily_hist is EMPTY. This is the cause of the error.")

    print("=" * 70 + "\n")

    if parameters is not None:
        # Historical simulation
        df_hist = run_historical_simulation(df_static, df_daily_hist, parameters, TEST_YEAR)

        # Forecast simulation
        df_fcst = run_forecast_simulation(df_static, df_daily_hist, df_seas5, parameters, TEST_YEAR)

        # Analysis
        if not df_hist.empty and not df_fcst.empty:
            analyze_and_plot_results(df_hist, df_fcst, TEST_YEAR)
            logging.info("\n" + "=" * 70)
            logging.info("✓ PIPELINE COMPLETED SUCCESSFULLY!")
            logging.info("=" * 70)
        else:
            logging.error("Simulations failed - no results to analyze")
    else:
        logging.error("Pipeline aborted - setup failed")