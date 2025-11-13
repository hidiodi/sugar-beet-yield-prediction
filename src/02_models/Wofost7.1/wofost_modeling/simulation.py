import datetime
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from pcse.models import Wofost72_WLP_FD, Wofost72_PP

from .data_providers import SimpleWeatherDataProvider
from .parameters import ParameterDict, _create_district_specific_parameters
from .weather import WeatherGenerator

def run_historical_simulation(df_static_year, df_daily_hist_year, cropdata,
                              year, cfg, dynamic_sowing_dates):
    """
    MODIFIED: Runs historical simulation using DYNAMIC sowing dates.
    """
    results = []

    # Get the static end date from config ONCE
    # We assume harvest date is still fixed for now.
    crop_end_date_template = cfg['AGROMANAGEMENT']['CROP_END_DATE']

    for _, row in tqdm(df_static_year.iterrows(),
                       total=len(df_static_year),
                       desc=f"Historical Sim {year}"):
        district_no = row['district_no']
        weather_df = df_daily_hist_year[
            df_daily_hist_year['district_no'] == district_no].copy()
        if weather_df.empty:
            continue

        try:
            parameters, site_data = _create_district_specific_parameters(
                row, cropdata)
            weather_provider = SimpleWeatherDataProvider(weather_df, site_data)

            # --- START OF THE CRITICAL CHANGE ---
            # Get the dynamic sowing date from the dictionary
            crop_start = dynamic_sowing_dates.get(district_no)

            # Fallback in case a date was not calculated
            if crop_start is None:
                logging.warning(
                    f"No dynamic sowing date for {district_no} in {year}. "
                    "Falling back to static date.")
                crop_start = cfg['AGROMANAGEMENT']['CROP_START_DATE'].replace(
                    year=year)

            # The harvest date remains the same for this year
            crop_end = crop_end_date_template.replace(year=year)
            # --- END OF THE CRITICAL CHANGE ---

            agromanagement = [{
                crop_start: ParameterDict({
                    'CropCalendar': ParameterDict({
                        'crop_start_date': crop_start,
                        'crop_start_type': 'emergence',
                        'crop_end_date': crop_end,
                        'crop_end_type': 'harvest',
                        'max_duration':
                            cfg['AGROMANAGEMENT']['MAX_DURATION']
                    }),
                    'TimedEvents': None,
                    'StateEvents': None
                })
            }]

            model = Wofost72_WLP_FD(
                parameters, weather_provider, agromanagement)
            model.run_till_terminate()
            output = model.get_output()
            simulated_yield = output[-1]['TWSO'] if output else np.nan

        except Exception as e:
            logging.error(
                f"[HISTORICAL] ERROR for district {district_no} in {year}: {e}",
                exc_info=True)
            simulated_yield = np.nan

        results.append({
            'year': year,
            'district_no': district_no,
            'actual_yield': row['kreisYield'],
            'lintul_yield_perfect_weather': simulated_yield
        })

    return pd.DataFrame(results)


def _run_single_forecast_member(member_row, district_no, year, wg, parameters,
                                site_data, cfg, apply_anomalies=True):
    try:
        spring_temp_anomaly = member_row.get(
            'spring_temp_anomaly_forecast', 0)
        summer_temp_anomaly = member_row.get(
            'summer_temp_anomaly_forecast', 0)
        spring_precip_anomaly = member_row.get(
            'spring_precip_anomaly_forecast', 0)
        summer_precip_anomaly = member_row.get(
            'summer_precip_anomaly_forecast', 0)

        monthly_anomalies = {}
        if apply_anomalies:
            for month in range(3, 11):
                if month in [3, 4, 5]:
                    monthly_anomalies[f'temp_anomaly_{month}'] = \
                        spring_temp_anomaly
                    monthly_anomalies[f'precip_anomaly_{month}'] = \
                        spring_precip_anomaly
                elif month in [6, 7, 8]:
                    monthly_anomalies[f'temp_anomaly_{month}'] = \
                        summer_temp_anomaly
                    monthly_anomalies[f'precip_anomaly_{month}'] = \
                        summer_precip_anomaly
                else:
                    monthly_anomalies[f'temp_anomaly_{month}'] = \
                        summer_temp_anomaly
                    monthly_anomalies[f'precip_anomaly_{month}'] = \
                        summer_precip_anomaly

        start_date = f'{year}-03-01'
        end_date = f'{year}-11-30'
        synth_weather = wg.generate(
            district_no, start_date, end_date, monthly_anomalies)

        # Check if weather generation was successful
        if synth_weather.empty or len(synth_weather) < \
                (datetime.date(year, 11, 30) - datetime.date(year, 3, 1)).days:
            logging.warning(
                f"[FORECAST_WORKER] Weather generation failed or incomplete "
                f"for dist {district_no}, member "
                f"{member_row.get('seas5_member', 'N/A')}. "
                "Returning default failure.")
            return {
                'member': member_row.get('seas5_member', 'N/A'),
                'yield_water_limited': 0.0,
                'yield_potential': 0.0,
                'consecutive_tmax_gt_30c': np.nan,
                'consecutive_dry_days': np.nan,
                'drought_stress_index': 1.0,
                'simulation_failed': True,
                'days_to_anthesis': np.nan,
                'max_lai_achieved': 0.0,
                'cumulative_water_stress': np.nan
            }

        weather_provider = SimpleWeatherDataProvider(synth_weather, site_data)
        crop_start = cfg['AGROMANAGEMENT']['CROP_START_DATE'].replace(
            year=year)
        crop_end = cfg['AGROMANAGEMENT']['CROP_END_DATE'].replace(year=year)
        agromanagement = [{
            crop_start: ParameterDict({
                'CropCalendar': ParameterDict({
                    'crop_start_date': crop_start,
                    'crop_start_type': 'emergence',
                    'crop_end_date': crop_end,
                    'crop_end_type': 'harvest',
                    'max_duration':
                        cfg['AGROMANAGEMENT']['MAX_DURATION']
                }),
                'TimedEvents': None,
                'StateEvents': None
            })
        }]

        model_wlp = Wofost72_WLP_FD(
            parameters, weather_provider, agromanagement)
        model_wlp.run_till_terminate()
        output_wlp = pd.DataFrame(model_wlp.get_output()).set_index('day')
        yield_wlp = output_wlp.iloc[-1]['TWSO'] if not output_wlp.empty else 0

        model_pp = Wofost72_PP(parameters, weather_provider, agromanagement)
        model_pp.run_till_terminate()
        output_pp = pd.DataFrame(model_pp.get_output())
        yield_pp = output_pp.iloc[-1]['TWSO'] if not output_pp.empty else 0

        def get_max_consecutive_run(boolean_series):
            if not boolean_series.any():
                return 0
            runs = boolean_series.ne(boolean_series.shift()).cumsum()
            return boolean_series.groupby(runs).cumsum().max()

        summer_weather = synth_weather[
            synth_weather['date'].dt.month.isin([6, 7, 8])].copy()
        is_heatwave_day = summer_weather['tmax'] > 30
        consecutive_hot_days = get_max_consecutive_run(is_heatwave_day)

        is_dry_day = summer_weather['precip'] < 1
        consecutive_dry_days = get_max_consecutive_run(is_dry_day)

        drought_stress_index = (yield_pp - yield_wlp) / \
            yield_pp if yield_pp > 0 else 0.0
        days_to_anthesis = np.nan
        if 'DOA' in output_wlp.columns and (output_wlp['DOA'] is not None):
            first_anthesis_day = output_wlp[
                output_wlp['DOA'].notna()].index.min()
            if pd.notna(first_anthesis_day):
                days_to_anthesis = (first_anthesis_day - crop_start).days

        max_lai_achieved = output_wlp['LAI'].max(
        ) if 'LAI' in output_wlp.columns else 0.0
        cumulative_water_stress = (1 - output_wlp['TRA']).sum(
        ) if 'TRA' in output_wlp.columns else np.nan

        return {
            'member': member_row.get('seas5_member', 'N/A'),
            'yield_water_limited': yield_wlp,
            'yield_potential': yield_pp,
            'consecutive_tmax_gt_30c': consecutive_hot_days,
            'consecutive_dry_days': consecutive_dry_days,
            'drought_stress_index': drought_stress_index,
            'simulation_failed': False,
            'days_to_anthesis': days_to_anthesis,
            'max_lai_achieved': max_lai_achieved,
            'cumulative_water_stress': cumulative_water_stress
        }

    except Exception as e:
        logging.warning(
            f"[FORECAST_WORKER] Sim failed for dist {district_no}, "
            f"member {member_row.get('seas5_member', 'N/A')}: {e}")
        return {
            'member': member_row.get('seas5_member', 'N/A'),
            'yield_water_limited': 0.0,
            'yield_potential': 0.0,
            'consecutive_tmax_gt_30c': np.nan,
            'consecutive_dry_days': np.nan,
            'drought_stress_index': 1.0,
            'simulation_failed': True,
            'days_to_anthesis': np.nan,
            'max_lai_achieved': 0.0,
            'cumulative_water_stress': np.nan
        }


def run_forecast_simulation(df_static_year, df_seas5_year, district_wgs,
                            cropdata, year, cfg, expert_districts):
    full_ensemble_results = []
    # Temporarily disable INFO logging from the main thread for cleaner progress
    # bar. Re-enable after this loop if needed, but forecast workers will log
    # their own issues
    logging.disable(logging.INFO)
    district_params = {
        row['district_no']: _create_district_specific_parameters(row, cropdata)
        for _, row in df_static_year.iterrows()
    }
    logging.disable(logging.NOTSET)  # Re-enable logging

    for district_no, group in tqdm(
            df_seas5_year.groupby('district_no'),
            desc=f"Forecast Sim {year}"):
        if district_no not in district_params:
            continue
        parameters, site_data = district_params[district_no]
        wg = district_wgs.get(district_no, WeatherGenerator())
        apply_anomalies = district_no not in expert_districts

        tasks = [
            delayed(_run_single_forecast_member)(
                member_row, district_no, year, wg, parameters,
                site_data, cfg, apply_anomalies)
            for _, member_row in group.iterrows()
        ]
        ensemble_outputs = Parallel(n_jobs=-1, backend='loky')(tasks)
        for result in ensemble_outputs:
            if result is not None:
                full_ensemble_results.append({
                    'year': year,
                    'district_no': district_no,
                    'member': result['member'],
                    'yield_water_limited_dry_kgha':
                        result['yield_water_limited'],
                    'yield_potential_dry_kgha': result['yield_potential'],
                    'consecutive_tmax_gt_30c':
                        result['consecutive_tmax_gt_30c'],
                    'consecutive_dry_days': result['consecutive_dry_days'],
                    'drought_stress_index': result['drought_stress_index'],
                    'simulation_failed': result['simulation_failed'],
                    'days_to_anthesis': result['days_to_anthesis'],
                    'max_lai_achieved': result['max_lai_achieved'],
                    'cumulative_water_stress':
                        result['cumulative_water_stress']
                })
    return pd.DataFrame(full_ensemble_results)
