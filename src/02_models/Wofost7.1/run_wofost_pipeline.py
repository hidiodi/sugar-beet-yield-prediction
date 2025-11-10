# File: run_wofost_pipeline.py
# Refactored to use central configuration from src.config

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

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src import config

# Use the WOFOST_CONFIG dictionary from the central config file
CONFIG = config.WOFOST_CONFIG

# ==============================================================================
# === SCRIPT STARTS HERE ===
# ==============================================================================
logging.getLogger().handlers = [];
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
handler.setFormatter(formatter);
logging.getLogger().addHandler(handler);
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
        super().__init__();
        self.latitude = site_data['LAT'];
        self.longitude = site_data['LON'];
        self.elevation = site_data['ELEV']
        self.angstA = 0.25;
        self.angstB = 0.5;
        weather_df = weather_df.copy();
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        self.store = {};
        fallback_wind = CONFIG['WEATHER_DEFAULTS']['WIND_SPEED'];
        fallback_vap_kpa = CONFIG['WEATHER_DEFAULTS']['VAPOR_PRESSURE']
        for _, row in weather_df.iterrows():
            try:
                day = row['date'].date();
                tmin = float(row['tmin']);
                tmax = float(row['tmax'])
                wind = float(row.get('wind', fallback_wind));

                # --- START: CORRECTED UNIT CONVERSIONS ---

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

                # --- END: CORRECTED UNIT CONVERSIONS ---

                et0_mm = penman_monteith(day, self.latitude, self.elevation, tmin, tmax, irrad_kj_m2_day, vap_hpa,
                                         wind);
                et0_cm = et0_mm / 10.0

                self.store[(day, 0)] = ParameterDict(
                    {'DAY': day, 'LAT': self.latitude, 'TMIN': tmin, 'TMAX': tmax, 'RAIN': precip_cm,
                     'IRRAD': irrad_j_m2_day, 'VAP': vap_hpa, 'WIND': wind, 'E0': et0_cm, 'ES0': et0_cm, 'ET0': et0_cm,
                     'SNOWDEPTH': 0.0})
            except Exception as e:
                logging.error(
                    f"CRITICAL: Failed processing weather row for date: {row.get('date')}. Error: {e}");
                raise e


class WeatherGenerator:
    def __init__(self):
        self.stats = defaultdict(dict);
        self.PRECIP_THRESHOLD_MM = CONFIG['WEATHER_GENERATOR'][
            'PRECIP_THRESHOLD_MM'];
        self.MIN_SRAD = CONFIG['WEATHER_GENERATOR']['MIN_SRAD']

    def fit(self, daily_df: pd.DataFrame):
        daily_df = daily_df.copy();
        daily_df['district_no'] = daily_df['district_no'].astype(str).str.zfill(5);
        daily_df['month'] = daily_df['date'].dt.month;
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
            p01 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 1)).sum();
            p00 = ((group['is_wet'].shift(1) == 0) & (group['is_wet'] == 0)).sum()
            p11 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 1)).sum();
            p10 = ((group['is_wet'].shift(1) == 1) & (group['is_wet'] == 0)).sum()
            prob_wet_given_dry = p01 / (p01 + p00) if (p01 + p00) > 0 else 0.1;
            prob_wet_given_wet = p11 / (p11 + p10) if (p11 + p10) > 0 else 0.5
            wet_day_precip = group[group['is_wet'] == 1]['precip']
            if len(wet_day_precip) > 2:
                a, loc, b = gamma.fit(wet_day_precip, floc=0);
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

                # --- NEWLY ADDED LINES ---
                'wind_mean': group['wind'].mean(),
                'wind_std': max(group['wind'].std(), 0.5)  # Wind can vary quite a bit
                # --- END NEW ---
            }

    def generate(self, district_no: str, start_date_str: str, end_date_str: str, monthly_anomalies: dict):
        dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D');
        generated_data = [];
        yesterday_was_wet = np.random.rand() < 0.5
        for date in dates:
            month, key = date.month, (str(district_no).zfill(5), date.month)
            if key not in self.stats: continue
            month_stats = self.stats[key];

            # --- Get historical stats ---
            transition_prob = month_stats['p_wet_given_wet'] if yesterday_was_wet else month_stats['p_wet_given_dry'];
            today_is_wet = np.random.rand() < transition_prob;
            precip = 0.0
            if today_is_wet: alpha = month_stats['precip_gamma_shape']; beta = month_stats[
                'precip_gamma_scale']; precip = max(0, gamma.rvs(a=alpha, scale=beta, size=1)[0])

            tmin = np.random.normal(month_stats['tmin_mean'], month_stats['tmin_std']);
            tmax = np.random.normal(month_stats['tmax_mean'], month_stats['tmax_std'])
            if tmax < tmin: tmax = tmin + abs(np.random.normal(0, 1.0))
            srad = max(self.MIN_SRAD, np.random.normal(month_stats['srad_mean'], month_stats['srad_std']));
            vap = max(0.1, np.random.normal(month_stats['vap_mean'], month_stats['vap_std']))

            # --- NEWLY ADDED LINE ---
            # Generate wind speed (m/s) from historical stats
            # Ensure wind speed is not negative
            wind = max(0.0, np.random.normal(month_stats['wind_mean'], month_stats['wind_std']))
            # --- END NEW ---

            generated_data.append(
                {'date': date, 'tmin': tmin, 'tmax': tmax, 'precip': precip, 'srad': srad, 'vap': vap, 'wind': wind});
            yesterday_was_wet = today_is_wet

        if not generated_data: return pd.DataFrame()

        synthetic_df = pd.DataFrame(generated_data);
        synthetic_df['month'] = synthetic_df['date'].dt.month

        for month in synthetic_df['month'].unique():
            month_mask = synthetic_df['month'] == month;
            key = (str(district_no).zfill(5), month)
            if key not in self.stats: continue

            # --- Temperature Anomaly (no change) ---
            temp_anomaly = monthly_anomalies.get(f'temp_anomaly_{month}', 0);
            hist_tmean = (self.stats[key]['tmin_mean'] + self.stats[key]['tmax_mean']) / 2;
            synth_tmean = (synthetic_df.loc[month_mask, 'tmin'].mean() + synthetic_df.loc[
                month_mask, 'tmax'].mean()) / 2
            temp_correction = (hist_tmean + temp_anomaly) - synth_tmean;
            synthetic_df.loc[month_mask, ['tmin', 'tmax']] += temp_correction;

            # --- Precipitation Anomaly (no change) ---
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

            # (Note: We are not yet applying wind *anomalies*,
            # just simulating the historical climatology, which is a huge improvement.)

        # --- NEW: Make sure 'wind' is in the output dataframe ---
        return synthetic_df[['date', 'tmin', 'tmax', 'precip', 'srad', 'vap', 'wind']]


def analyze_all_inputs(config, df_static_all, df_seas5_all, full_hist_weather_df, cropdata):
    """
    FINAL VERSION: Performs a diagnostic analysis of all model inputs,
    now that all known unit-conversion bugs are fixed.
    """
    logging.info("=" * 80)
    logging.info("--- STARTING FINAL INPUT DATA ANALYSIS (v3) ---")
    logging.info("=" * 80)

    analysis_passed = True

    # --- 1. Analyze Static CONFIG Dictionary ---
    try:
        logging.info("--- [1/5] Analyzing CONFIG Dictionary ---")
        logging.info(f"    Years: {config['START_YEAR']} to {config['END_YEAR']}")
        logging.info(f"    AgroManagement:")
        logging.info(f"        Start Date: {config['AGROMANAGEMENT']['CROP_START_DATE']}")
        logging.info(f"        End Date:   {config['AGROMANAGEMENT']['CROP_END_DATE']}")
    except Exception as e:
        logging.error(f"    [FAIL] Error analyzing CONFIG: {e}", exc_info=True)
        analysis_passed = False

    # --- 2. Analyze Loaded Crop Parameters (from sugarbeet.yaml) ---
    try:
        logging.info("--- [2/5] Analyzing Loaded CropData (from YAML) ---")
        cvl = cropdata.get('CVL', 'MISSING')
        tsum1 = cropdata.get('TSUM1', 'MISSING')
        amaxtb = cropdata.get('AMAXTB', 'MISSING')
        logging.info(f"    [OK] CVL: {cvl}, TSUM1: {tsum1}, AMAXTB type: {type(amaxtb)}")
    except Exception as e:
        logging.error(f"    [FAIL] Error analyzing CropData: {e}", exc_info=True)
        analysis_passed = False

    # --- 3. Analyze Merged Static Data (Yield & Soil) ---
    try:
        logging.info("--- [3/5] Analyzing Static Data (df_static_all) ---")
        sand_max = df_static_all[config['SOIL_COLUMN_MAPPING']['sand']].max()
        bdod_mean = df_static_all[config['SOIL_COLUMN_MAPPING']['bdod']].mean()
        if sand_max > 1.0 and bdod_mean < 2.0:
            logging.info(f"    [OK] Soil data (Sand max: {sand_max}, Bdod mean: {bdod_mean}) looks plausible.")
        else:
            logging.warning(f"    [WARNING] Soil data seems off (Sand max: {sand_max}, Bdod mean: {bdod_mean}).")
            analysis_passed = False
    except Exception as e:
        logging.error(f"    [FAIL] Error analyzing Static Data: {e}", exc_info=True)
        analysis_passed = False

    # --- 4. Analyze Forecast Anomaly Data (SEAS5) ---
    try:
        logging.info("--- [4/5] Analyzing Forecast Data (df_seas5_all) ---")
        precip_cols = [c for c in df_seas5_all.columns if 'precip_anomaly_forecast' in c]
        min_precip_anom = df_seas5_all[precip_cols].min().min()
        if min_precip_anom <= -1.0:
            logging.info(f"    [INFO] Min precip anomaly is {min_precip_anom:.2f}. This is extreme,")
            logging.info(f"    ... but the new WeatherGenerator.generate() function correctly handles this.")
        else:
            logging.info(f"    [OK] Precip anomaly range seems plausible.")
    except Exception as e:
        logging.error(f"    [FAIL] Error analyzing Forecast Data: {e}", exc_info=True)
        analysis_passed = False

    # --- 5. Analyze Historical Weather Data (ERA5) ---
    try:
        logging.info("--- [5/5] Analyzing Historical Weather (full_hist_weather_df) ---")
        weather_stats = full_hist_weather_df[['tmin', 'tmax', 'precip', 'srad']].describe().T

        tmin_mean = weather_stats.loc['tmin', 'mean']
        precip_mean = weather_stats.loc['precip', 'mean']
        srad_mean = weather_stats.loc['srad', 'mean']

        if tmin_mean > 200:
            logging.error(f"    [FAIL] Mean TMIN is {tmin_mean} (Kelvin?). Model expects Celsius.")
            analysis_passed = False
        else:
            logging.info(f"    [OK] TMIN mean is {tmin_mean:.2f} (Celsius).")

        if precip_mean > 100.0:
            logging.error(f"    [FAIL] Mean precip is {precip_mean}. Unit is not mm/day. Rerun build_weather_data.py.")
            analysis_passed = False
        else:
            logging.info(f"    [OK] Precip mean is {precip_mean:.2f} (mm/day).")

        # Check for srad in MJ/m²/day
        if 8.0 < srad_mean < 18.0:
            logging.info(f"    [OK] Srad mean is {srad_mean:.2f} (MJ/m²/day).")
        else:
            logging.warning(f"    [WARNING] Srad mean is {srad_mean:.2f}. Expected 8-18 for MJ/m²/day.")

    except Exception as e:
        logging.error(f"    [FAIL] Error analyzing Historical Weather: {e}", exc_info=True)
        analysis_passed = False

    # --- Final Summary ---
    logging.info("=" * 80)
    if analysis_passed:
        logging.info("--- ANALYSIS COMPLETE: All data looks correct and plausible. ---")
    else:
        logging.error("--- ANALYSIS FAILED: Errors found. See log. ---")
    logging.info("=" * 80)

    return analysis_passed


def _calculate_soil_hydraulic_properties(sand_frac, clay_frac, som_frac, bdod):
    """
    Calculates soil hydraulic properties using Saxton & Rawls (2006) equations.
    Inputs MUST be fractions (0-1), not percentages.

    - sand_frac: Sand fraction (0-1) e.g., 0.47
    - clay_frac: Clay fraction (0-1) e.g., 0.23
    - som_frac: Soil Organic Matter fraction (0-1) e.g., 0.025
    - bdod: Bulk Density (g/cm^3)
    """

    # --- 1. Calculate Texture-Based Properties (in Percent) ---
    # The Saxton-Rawls equations are built to use percentages (e.g., 47, 23, 2.5)
    sand_pct = sand_frac * 100.0
    clay_pct = clay_frac * 100.0
    som_pct = som_frac * 100.0

    # Eq. 3: Wilting Point (WP) based on texture (value is in percent)
    wp_texture_pct = -0.024 * sand_pct + 0.487 * clay_pct + 0.006 * som_pct + \
                     0.005 * sand_pct * som_pct - 0.013 * clay_pct * som_pct + \
                     0.068 * sand_pct * clay_pct + 0.031

    # Eq. 5: Field Capacity (FC) based on texture (value is in percent)
    fc_texture_pct = -0.251 * sand_pct + 0.195 * clay_pct + 0.011 * som_pct + \
                     0.006 * sand_pct * som_pct - 0.027 * clay_pct * som_pct + \
                     0.452 * sand_pct * clay_pct + 0.299

    # --- 2. Convert to Fractions (0-1) for Adjustment Formulas ---
    # This was the bug: the adjustment formulas need fractions, not percentages.
    wp_texture_frac = wp_texture_pct / 100.0
    fc_texture_frac = fc_texture_pct / 100.0

    # --- 3. Calculate Bulk Density Adjustment (as a fractional adjustment) ---
    # Using 1.35 g/cm^3 as a reference bulk density
    bd_adj_factor = (1.7 - bdod / 1.35)

    # Eq. 4: WP adjustment (this formula expects fractions)
    wp_adj = (0.14 * wp_texture_frac - 0.02) * bd_adj_factor

    # Eq. 6: FC adjustment (this formula expects fractions)
    fc_adj = (1.283 * fc_texture_frac ** 2 - 0.374 * fc_texture_frac - 0.015) * bd_adj_factor

    # --- 4. Calculate Final Values (as fractions) ---
    smw_frac = wp_texture_frac + wp_adj
    smfcf_frac = fc_texture_frac + fc_adj

    # --- 5. Calculate Saturation (SM0) (as a fraction) ---
    porosity = 1.0 - (bdod / CONFIG['CONSTANTS']['SOIL_PARTICLE_DENSITY'])

    # Eq. 8: Saturation (SM0) based on porosity and adjusted FC
    # (This formula also expects all inputs to be fractions)
    sm0_frac = smfcf_frac + (porosity - smfcf_frac) * (1.0 - (0.6 / (1.0 + (smfcf_frac / 0.6) ** 5)) ** (1.0 / 5.0))

    # --- 6. Final safety checks (all values are fractions) ---
    smw = max(0.01, smw_frac)
    smfcf = max(smw + 0.01, smfcf_frac)
    sm0 = max(smfcf + 0.01, sm0_frac)
    crairc = max(0.005, sm0 - smfcf)  # Air capacity

    return {'SMW': smw, 'SMFCF': smfcf, 'SM0': sm0, 'CRAIRC': crairc}


def _create_district_specific_parameters(static_row, cropdata):
    sitedata = ParameterDict()
    latitude = static_row.get('latitude', CONFIG['GENERIC_SITE']['LATITUDE'])
    longitude = static_row.get('longitude', CONFIG['GENERIC_SITE']['LONGITUDE'])
    elevation = static_row.get('avg_elevation', CONFIG['GENERIC_SITE']['ELEVATION'])
    sitedata.add_variable('LAT', latitude)
    sitedata.add_variable('LON', longitude)
    sitedata.add_variable('ELEV', elevation)

    soildata = ParameterDict()
    soil_map = CONFIG['SOIL_COLUMN_MAPPING']

    try:
        # Calculate soil properties using the standard equations
        sand = static_row[soil_map['sand']] / 100.0
        clay = static_row[soil_map['clay']] / 100.0
        som = static_row[soil_map['som']] / 100.0
        bdod = static_row[soil_map['bdod']]
        calculated_soil_params = _calculate_soil_hydraulic_properties(sand, clay, som, bdod)

        for key, value in calculated_soil_params.items():
            soildata.add_variable(key, value)

    except (KeyError, TypeError) as e:
        logging.error(f"FATAL: Missing soil data for district {static_row.get('district_no', 'N/A')}. Error: {e}")
        raise e

    # Add default soil constants
    for key, value in CONFIG['SOIL_DEFAULTS_AND_CONSTANTS'].items():
        soildata.add_variable(key, value)

    # ==============================================================================
    # === S T A R T   O F   T H E   C O R R E C T   F I X ===
    # ==============================================================================
    #
    # PROBLEM: The model incorrectly penalizes high-quality, water-retentive soils
    # by simulating severe oxygen stress (waterlogging) during periods of high
    # rainfall. This creates a yield ceiling that specifically affects the best districts.
    # The internal calculation of CRAIRC (Critical Air Content) is flawed for these soils.
    #
    # SOLUTION: We will globally disable the oxygen stress factor for ALL simulations.
    # We do this by overriding the calculated CRAIRC value and setting it to a
    # minimally small number (0.01). This prevents the model from ever applying the
    # growth-limiting penalty, allowing yield to be determined by the primary
    # drivers of light, temperature, and water availability.

    original_crairc = soildata.get('CRAIRC', 'N/A')  # Get the flawed calculated value for logging
    soildata.add_variable('CRAIRC', 0.01)  # Globally disable oxygen stress

    # ==============================================================================
    # === E N D   O F   T H E   C O R R E C T   F I X ===
    # ==============================================================================

    smfc = soildata['SMFCF']
    smw = soildata['SMW']
    rdi = CONFIG['CONSTANTS']['INITIAL_ROOTING_DEPTH_CM']
    rdmsol = soildata['RDMSOL']
    smlim = smfc

    # WAV calculation MUST come after all soil data is set
    wav = (smfc - smw) * rdmsol

    sitedata.add_variable('SMLIM', smlim)
    sitedata.add_variable('WAV', wav)
    sitedata.add_variable('IFUNRN', 0.0)
    sitedata.add_variable('NOTINF', 0.0)
    sitedata.add_variable('SSI', 0.0)
    sitedata.add_variable('SSMAX', 0.0)

    try:
        logging.info(f"  Dist: {static_row['district_no']} | Yield: {static_row['kreisYield']:<8} | " +
                     f"WAV: {wav:<8.2f} | Orig_CRAIRC: {original_crairc:<8.4f} -> Final_CRAIRC: {soildata.CRAIRC:<8.4f}")
    except KeyError:
        pass  # Don't fail if kreisYield is not in the row (e.g., during forecast)

    return ParameterProvider(cropdata=cropdata, soildata=soildata, sitedata=sitedata), sitedata


# =====================================================================

def run_historical_simulation(df_static_year, df_daily_hist_year, cropdata, year, cfg):
    results = [];
    for _, row in tqdm(df_static_year.iterrows(), total=len(df_static_year), desc=f"Historical Sim {year}"):
        district_no = row['district_no'];
        weather_df = df_daily_hist_year[df_daily_hist_year['district_no'] == district_no].copy()
        if weather_df.empty: continue
        try:
            # This function will now do the logging
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
            logging.error(f"[HISTORICAL] ERROR for district {district_no} in {year}: {e}",
                          exc_info=True);
            simulated_yield = np.nan
        results.append({'year': year, 'district_no': district_no, 'actual_yield': row['kreisYield'],
                        'lintul_yield_perfect_weather': simulated_yield})
    return pd.DataFrame(results)


def _run_single_forecast_member(member_row, district_no, year, wg, parameters, site_data, cfg, apply_anomalies=True):
    try:
        # === START OF FIX: Load seasonal anomalies and map to months ===
        spring_temp_anomaly = member_row.get('spring_temp_anomaly_forecast', 0)
        summer_temp_anomaly = member_row.get('summer_temp_anomaly_forecast', 0)

        spring_precip_anomaly = member_row.get('spring_precip_anomaly_forecast', 0)
        summer_precip_anomaly = member_row.get('summer_precip_anomaly_forecast', 0)

        monthly_anomalies = {}
        # *** MODIFICATION: Conditionally apply anomalies ***
        if apply_anomalies:
            for month in range(3, 11):  # March to October
                if month in [3, 4, 5]:  # Spring
                    monthly_anomalies[f'temp_anomaly_{month}'] = spring_temp_anomaly
                    monthly_anomalies[f'precip_anomaly_{month}'] = spring_precip_anomaly

                elif month in [6, 7, 8]:  # Summer
                    monthly_anomalies[f'temp_anomaly_{month}'] = summer_temp_anomaly
                    monthly_anomalies[f'precip_anomaly_{month}'] = summer_precip_anomaly

                else:  # Autumn (Sep, Oct)
                    # Best guess: extend the summer forecast as it's the latest available
                    monthly_anomalies[f'temp_anomaly_{month}'] = summer_temp_anomaly
                    monthly_anomalies[f'precip_anomaly_{month}'] = summer_precip_anomaly

        start_date, end_date = f'{year}-03-01', f'{year}-11-30';
        # === END OF FIX ===

        synth_weather = wg.generate(district_no, start_date, end_date, monthly_anomalies)
        if synth_weather.empty: return {'member': member_row.get('seas5_member', 'N/A'), 'yield_water_limited': 0.0,
                                        'yield_potential': 0.0, 'consecutive_tmax_gt_30c': np.nan,
                                        'consecutive_dry_days': np.nan, 'drought_stress_index': 1.0,
                                        'simulation_failed': True, 'days_to_anthesis': np.nan, 'max_lai_achieved': 0.0,
                                        'cumulative_water_stress': np.nan}

        weather_provider = SimpleWeatherDataProvider(synth_weather, site_data);
        crop_start = cfg['AGROMANAGEMENT']['CROP_START_DATE'].replace(year=year);
        crop_end = cfg['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)
        agromanagement = [{crop_start: ParameterDict({'CropCalendar': ParameterDict(
            {'crop_start_date': crop_start, 'crop_start_type': 'emergence', 'crop_end_date': crop_end,
             'crop_end_type': 'harvest', 'max_duration': cfg['AGROMANAGEMENT']['MAX_DURATION']}), 'TimedEvents': None,
            'StateEvents': None})}]

        model_wlp = Wofost72_WLP_FD(parameters, weather_provider, agromanagement);
        model_wlp.run_till_terminate();
        output_wlp = pd.DataFrame(model_wlp.get_output()).set_index('day');
        yield_wlp = output_wlp.iloc[-1]['TWSO'] if not output_wlp.empty else 0

        model_pp = Wofost72_PP(parameters, weather_provider, agromanagement);
        model_pp.run_till_terminate();
        output_pp = pd.DataFrame(model_pp.get_output());
        yield_pp = output_pp.iloc[-1]['TWSO'] if not output_pp.empty else 0

        def get_max_consecutive_run(boolean_series):
            if not boolean_series.any(): return 0
            runs = boolean_series.ne(boolean_series.shift()).cumsum();
            return boolean_series.groupby(runs).cumsum().max()

        summer_weather = synth_weather[synth_weather['date'].dt.month.isin([6, 7, 8])].copy()
        is_heatwave_day = summer_weather['tmax'] > 30
        consecutive_hot_days = get_max_consecutive_run(is_heatwave_day);

        is_dry_day = summer_weather[
                         'precip'] < 1
        consecutive_dry_days = get_max_consecutive_run(is_dry_day)

        drought_stress_index = (yield_pp - yield_wlp) / yield_pp if yield_pp > 0 else 0.0;
        days_to_anthesis = np.nan
        if 'DOA' in output_wlp.columns and (output_wlp['DOA'] is not None):
            first_anthesis_day = output_wlp[output_wlp['DOA'].notna()].index.min()
            if pd.notna(first_anthesis_day): days_to_anthesis = (first_anthesis_day - crop_start).days

        max_lai_achieved = output_wlp['LAI'].max() if 'LAI' in output_wlp.columns else 0.0;
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
    logging.disable(logging.INFO);
    district_params = {row['district_no']: _create_district_specific_parameters(row, cropdata) for _, row in
                       df_static_year.iterrows()};
    logging.disable(logging.NOTSET)
    for district_no, group in tqdm(df_seas5_year.groupby('district_no'), desc=f"Forecast Sim {year}"):
        if district_no not in district_params: continue
        parameters, site_data = district_params[district_no];
        wg = district_wgs.get(district_no, WeatherGenerator())
        # *** MODIFICATION: Check if this is an "expert" district to avoid double-counting anomalies ***
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
    mae_p = mean_absolute_error(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt']);
    r2_p = r2_score(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt']);
    mae_f = mean_absolute_error(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean']);
    r2_f = r2_score(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'])

    print("\n--- Overall Performance Metrics (Fresh Weight dt/ha, based on Ensemble Mean) ---");
    print(f"  Perfect Weather (WLP):  MAE = {mae_p:.2f}, R² = {r2_p:.3f}");
    print(f"  Forecast Weather (WLP): MAE = {mae_f:.2f}, R² = {r2_f:.3f}\n")

    # --- 8. Create Diagnostic Plots (only if data is valid) ---
    logging.info("[ANALYSIS] Generating diagnostic plots (if any valid data exists)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True);
    fig.suptitle(f'WOFOST Performance Diagnosis ({start_year}-{end_year})', fontsize=16)

    min_val_data = df_final_clean[
        ['actual_yield', 'perfect_yield_dt', 'forecast_yield_p10', 'potential_yield_mean']].min()
    max_val_data = df_final_clean[
        ['actual_yield', 'perfect_yield_dt', 'forecast_yield_p90', 'potential_yield_mean']].max()
    min_val = min_val_data.min() * 0.95
    max_val = max_val_data.max() * 1.05

    # === PLOT 1: PERFECT WEATHER ===
    axes[0].scatter(df_final_clean['actual_yield'], df_final_clean['perfect_yield_dt'], alpha=0.6,
                    label='Simulated (Perfect Weather)');
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line');
    axes[0].scatter(df_final_clean['actual_yield'], df_final_clean['potential_yield_mean'],
                    marker='x', color='red', alpha=0.5, label='Mean Potential Yield (PP)')
    axes[0].set_title(f'Perfect Weather\nMAE={mae_p:.2f}, R²={r2_p:.3f}');
    axes[0].set_xlabel('Actual Yield (dt/ha)');
    axes[0].set_ylabel('Simulated Yield (dt/ha)')

    # === PLOT 2: FORECAST WEATHER (ENSEMBLE) ===
    lower_error = df_final_clean['forecast_yield_mean'] - df_final_clean['forecast_yield_p10'];
    upper_error = df_final_clean['forecast_yield_p90'] - df_final_clean['forecast_yield_mean'];
    y_err = [lower_error.values, upper_error.values]

    axes[1].errorbar(df_final_clean['actual_yield'], df_final_clean['forecast_yield_mean'], yerr=y_err, fmt='o',
                     color='orange',
                     ecolor='lightgray', elinewidth=3, capsize=0, alpha=0.8,
                     label='Ensemble Mean & 10-90th Pct. Range');
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line');
    axes[1].scatter(df_final_clean['actual_yield'], df_final_clean['potential_yield_mean'],
                    marker='x', color='red', alpha=0.5, label='Mean Potential Yield (PP)')
    axes[1].set_title(f'Forecast Weather (Ensemble Range)\nMean MAE={mae_f:.2f}, Mean R²={r2_f:.3f}');
    axes[1].set_xlabel('Actual Yield (dt/ha)')

    # --- 9. Finalize and Save Plot ---
    for ax in axes:
        ax.set_xlim(min_val, max_val);
        ax.set_ylim(min_val, max_val);
        ax.grid(True, alpha=0.3);
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    plot_path = output_dir / f'results_scatter_with_POTENTIAL_{start_year}-{end_year}.png';
    plt.savefig(plot_path, dpi=300);
    logging.info(f"[ANALYSIS] ✓ Diagnostic plot (with potential yield) saved to {plot_path}");
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
    # This function is in its original state
    pass


if __name__ == "__main__":
    pcse_log_dir = Path.home() / ".pcse" / "logs"
    pcse_log_file = pcse_log_dir / "pcse.log"
    if pcse_log_file.exists():
        try:
            # Try to delete the old log file
            os.remove(pcse_log_file)
        except PermissionError:
            pass  # If it's locked, we can't do much, but we'll try

    # Disable the file handler to stop the PermissionErrors
    logging.getLogger('pcse').handlers = [h for h in logging.getLogger('pcse').handlers if
                                          not isinstance(h, logging.FileHandler)]
    os.makedirs(CONFIG['FILE_PATHS']['OUTPUT_DIR'], exist_ok=True)

    # ==============================================================================
    # === MODIFIED DATA LOADING AS PER FINAL INSTRUCTIONS ===
    # ==============================================================================
    logging.info("=" * 70 + "\nLoading and merging raw data sources...\n" + "=" * 70)
    try:
        # 1. Load the raw yield data
        df_yield = pd.read_csv(CONFIG['FILE_PATHS']['YIELD_DATA'])
        df_yield.rename(columns={'yield': 'kreisYield'}, inplace=True)
        logging.info(f"Loaded {len(df_yield)} records from yield data file.")

        # 2. Load the correct soil data
        df_soil = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_SOIL_FEATURES'])
        logging.info(f"Loaded {len(df_soil)} records from static soil features file.")

        # 3. Merge the two data sources to create the master static data table
        df_yield['district_no'] = df_yield['district_no'].astype(str).str.zfill(5)
        df_soil['district_no'] = df_soil['district_no'].astype(str).str.zfill(5)
        df_static_all = pd.merge(df_yield, df_soil, on='district_no', how='inner')
        logging.info(f"Merged data into a single table with {len(df_static_all)} records.")

        # 4. Load the forecast data
        df_seas5_all = pd.read_csv(CONFIG['FILE_PATHS']['SEAS5_MEMBER_FEATURES'])
        df_seas5_all['district_no'] = pd.to_numeric(df_seas5_all['district_no'], errors='coerce').astype(
            'Int64').astype(str).str.zfill(5)

        # 5. Filter and synchronize data
        df_static_all.dropna(subset=['kreisYield'], inplace=True)
        df_static_all = df_static_all[df_static_all['kreisYield'] > 0]

        valid_combinations = df_static_all[['year', 'district_no']].drop_duplicates()
        df_seas5_all = pd.merge(df_seas5_all, valid_combinations, on=['year', 'district_no'], how='inner')

        logging.info(f"Proceeding with {len(df_static_all)} valid district-year records across all data sources.")

    except FileNotFoundError as e:
        logging.error(f"FATAL: A required data file was not found. Error: {e}");
        sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: An error occurred during data loading and merging. Error: {e}");
        sys.exit(1)
    # ==============================================================================
    # === END OF DATA LOADING MODIFICATIONS ===
    # ==============================================================================

    # --- The rest of the script is in its original state ---

    all_hist_dfs = []
    logging.info("Loading all available yearly weather files into memory for analog search...")
    for year_to_load in range(1981, CONFIG['END_YEAR'] + 1):
        hist_weather_path = os.path.join(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'],
                                         f"historical_daily_weather_era5_{year_to_load}.csv")
        if os.path.exists(hist_weather_path):
            df = pd.read_csv(hist_weather_path, parse_dates=['date']);
            df['year'] = year_to_load;
            all_hist_dfs.append(df)
    if not all_hist_dfs: logging.error("FATAL: No historical weather files found. Aborting."); sys.exit()
    full_hist_weather_df = pd.concat(all_hist_dfs, ignore_index=True)
    logging.info(
        f"[DIAGNOSTIC] NaN count for 'vap': {full_hist_weather_df['vap'].isna().sum()} / {len(full_hist_weather_df)}")
    logging.info(
        f"[DIAGNOSTIC] NaN count for 'wind': {full_hist_weather_df['wind'].isna().sum()} / {len(full_hist_weather_df)}")
    full_hist_weather_df['district_no'] = full_hist_weather_df['district_no'].astype(str).str.zfill(5)
    full_hist_weather_df['month'] = full_hist_weather_df['date'].dt.month

    # === START: NEW CORRECTED PARAMETER LOADING (Fix for TraitError) ===

    logging.info("Loading and merging crop parameters from sugarbeet.yaml...")
    try:
        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            crop_params_all = yaml.safe_load(f)['CropParameters']

        # 1. Load the base dictionaries
        base_params = crop_params_all.get('GenericC3', {})
        ecotype_params = crop_params_all['EcoTypes']['sugarbeet']
        variety_params = crop_params_all['Varieties']['Sugarbeet_601']

        # 2. Merge them in the correct order of inheritance
        # (Variety overrides Ecotype, which overrides Base C3)
        final_params_dict = {**base_params, **ecotype_params, **variety_params}

        # 3. Instantiate the final ParameterDict
        cropdata = ParameterDict()

        # 4. Unpack the [value, description, unit] lists
        #    This is the crucial step that fixes the TraitError.
        for key, val in final_params_dict.items():
            if key in ['Metadata', '<<']:
                continue  # Skip YAML/metadata keys

            # The YAML structure is [value, description, unit]
            # We just want the value, which is the first element (val[0]).
            # This works for floats (CVL), ints (IDSL), and lists (AMAXTB).
            if isinstance(val, list) and len(val) > 0:
                cropdata.add_variable(key, val[0], "Loaded from YAML")
            else:
                # This handles any stray keys that don't follow the list format
                pass

        logging.info(f"Successfully loaded and merged parameters for variety: Sugarbeet_601")
        logging.info(f"Test load: CVL = {cropdata.CVL} (Type: {type(cropdata.CVL)})")

    except Exception as e:
        logging.error(f"FATAL: Could not load or parse crop YAML file. Error: {e}", exc_info=True)
        sys.exit(1)

    if not analyze_all_inputs(CONFIG, df_static_all, df_seas5_all, full_hist_weather_df, cropdata):
        logging.error("Input data analysis failed. Aborting pipeline.")
        logging.shutdown()
        sys.exit(1)

    all_hist_results = []
    all_fcst_results = []
    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        logging.info("=" * 70 + f"\nPROCESSING YEAR: {year}\n" + "=" * 70)
        hist_weather_path = os.path.join(CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'],
                                         f"historical_daily_weather_era5_{year}.csv")
        try:
            df_daily_hist_year = pd.read_csv(hist_weather_path, parse_dates=['date'])
            df_daily_hist_year['district_no'] = df_daily_hist_year['district_no'].astype(str).str.zfill(5)
        except FileNotFoundError:
            logging.error(f"FATAL: Weather file for {year} not found at {hist_weather_path}. Skipping year.");
            continue

        df_static_year = df_static_all[df_static_all['year'] == year].copy()
        df_seas5_year = df_seas5_all[df_seas5_all['year'] == year].copy()

        if CONFIG['DISTRICT_LIMIT'] is not None:
            limited_districts = df_static_year['district_no'].unique()[:CONFIG['DISTRICT_LIMIT']]
            df_static_year = df_static_year[df_static_year['district_no'].isin(limited_districts)]
            df_seas5_year = df_seas5_year[df_seas5_year['district_no'].isin(limited_districts)]
            logging.info(f"Applying DISTRICT_LIMIT: Running for {len(limited_districts)} districts.")

        if df_static_year.empty or df_seas5_year.empty:
            logging.warning(f"Missing static or forecast data for year {year}. Skipping.");
            continue

        past_weather_df = full_hist_weather_df[full_hist_weather_df['year'] < year].copy()
        district_weather_generators = {}
        expert_districts = set()  # MODIFICATION: Keep track of expert districts
        for district_no in tqdm(df_static_year['district_no'].unique(), desc=f"Fitting Expert WGs for {year}"):
            district_past_weather = past_weather_df[past_weather_df['district_no'] == district_no]

            if len(district_past_weather['year'].unique()) < CONFIG['ANALOG_YEAR_CONFIG']['MIN_YEARS_FOR_FIT']:
                # Not enough historical data, use a generic WG trained on what it can get
                wg_expert = WeatherGenerator()
                if not district_past_weather.empty:
                    wg_expert.fit(district_past_weather)
                district_weather_generators[district_no] = wg_expert

            else:
                # --- START: CORRECTED ANALOG YEAR LOGIC ---
                expert_districts.add(district_no)  # MODIFICATION: Mark this as an expert district

                # 1. Calculate historical climatology (long-term average)
                climatology = district_past_weather.groupby('month')[['tmin', 'tmax', 'precip']].mean()
                climatology['temp'] = (climatology['tmin'] + climatology['tmax']) / 2

                # 2. Get target forecast ANOMALIES for the current year
                target_forecast = df_seas5_year[df_seas5_year['district_no'] == district_no]
                target_anomalies = {
                    month: {
                        'temp': target_forecast.get(
                            f'spring_temp_anomaly_forecast' if month in [3, 4, 5] else f'summer_temp_anomaly_forecast',
                            pd.Series(0.0)).mean(),
                        'precip': target_forecast.get(f'spring_precip_anomaly_forecast' if month in [3, 4,
                                                                                                     5] else f'summer_precip_anomaly_forecast',
                                                      pd.Series(0.0)).mean()
                    } for month in range(3, 11)
                }

                # 3. Calculate the TARGET ABSOLUTE weather for the forecast year
                target_weather = climatology.copy()
                for month in range(3, 11):
                    if month in target_weather.index:
                        target_weather.loc[month, 'temp'] += target_anomalies[month]['temp']
                        target_weather.loc[month, 'precip'] += target_anomalies[month]['precip']

                # 4. Calculate historical yearly averages (ABSOLUTE values)
                yearly_avg = district_past_weather.groupby(['year', 'month'])[
                    ['precip', 'tmin', 'tmax']].mean().reset_index()
                yearly_avg['temp'] = (yearly_avg['tmin'] + yearly_avg['tmax']) / 2

                # 5. Pivot historical data to have one row per year
                hist_pivot = yearly_avg.pivot_table(index='year', columns='month', values=['temp', 'precip']);
                hist_pivot.columns = [f'{val}_{month}' for val, month in hist_pivot.columns]

                # 6. Create a target series for comparison
                target_series = {}
                for month in range(3, 11):
                    if month in target_weather.index:
                        target_series[f'temp_{month}'] = target_weather.loc[month, 'temp']
                        target_series[f'precip_{month}'] = target_weather.loc[month, 'precip']

                common_cols = hist_pivot.columns.intersection(target_series.keys())

                if not common_cols.any():
                    wg_expert = WeatherGenerator();
                    wg_expert.fit(district_past_weather);
                    district_weather_generators[district_no] = wg_expert;
                    continue

                # 7. Find best analog years by comparing ABSOLUTE weather
                aligned_target = pd.Series(target_series)[common_cols];
                distances = np.sqrt(
                    np.sum((hist_pivot[common_cols].dropna() - aligned_target) ** 2, axis=1)).sort_values()
                analog_years = distances.head(CONFIG['ANALOG_YEAR_CONFIG']['NUM_ANALOGS']).index.tolist();
                analog_weather_data = district_past_weather[district_past_weather['year'].isin(analog_years)]

                # 8. Train the expert WG on the weather of those best-matching years
                wg_expert = WeatherGenerator();
                wg_expert.fit(analog_weather_data);
                district_weather_generators[district_no] = wg_expert
                # --- END: CORRECTED ANALOG YEAR LOGIC ---

        df_hist = run_historical_simulation(df_static_year, df_daily_hist_year, cropdata, year, CONFIG)
        # MODIFICATION: Pass the set of expert districts to the forecast runner
        df_fcst = run_forecast_simulation(df_static_year, df_seas5_year, district_weather_generators, cropdata, year,
                                          CONFIG, expert_districts)
        if not df_hist.empty: all_hist_results.append(df_hist)
        if not df_fcst.empty: all_fcst_results.append(df_fcst)

    if all_hist_results and all_fcst_results:
        final_hist_df = pd.concat(all_hist_results, ignore_index=True)
        final_fcst_df = pd.concat(all_fcst_results, ignore_index=True)
        aggregate_and_save_extreme_weather_metrics(final_fcst_df,
                                                   CONFIG['FILE_PATHS']['EXTREME_WEATHER_METRICS_OUTPUT'])
        analyze_and_plot_ensemble_results(final_hist_df, final_fcst_df, CONFIG['FILE_PATHS']['OUTPUT_DIR'],
                                          CONFIG['START_YEAR'], CONFIG['END_YEAR'])
        logging.info("\n" + "=" * 70 + "\n✓ MULTI-YEAR PIPELINE COMPLETED SUCCESSFULLY!\n" + "=" * 70)
    else:
        logging.error("No simulation results were generated across all years. Aborting final analysis.")

    logging.shutdown()
