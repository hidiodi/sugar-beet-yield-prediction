# File: run_wofost_pipeline.py
# Refactored to use central configuration and a modular, package-based structure.

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
import geopandas as gpd
from tqdm import tqdm


from src import config
from src.02_models.Wofost7.1.wofost_modeling.analysis import (
    analyze_v2_model_inputs,
    analyze_and_plot_ensemble_results,
    aggregate_and_save_extreme_weather_metrics
)
from src.02_models.Wofost7.1.wofost_modeling.parameters import ParameterDict
from src.02_models.Wofost7.1.wofost_modeling.sowing import DynamicSowingManager
from src.02_models.Wofost7.1.wofost_modeling.weather import WeatherGenerator
from src.02_models.Wofost7.1.wofost_modeling.simulation import (
    run_historical_simulation,
    run_forecast_simulation
)

# Use the WOFOST_CONFIG dictionary from the central config file
CONFIG = config.WOFOST_CONFIG

# ==============================================================================
# === SCRIPT STARTS HERE ===
# ==============================================================================
# Clear existing handlers to prevent duplicate logging
logging.getLogger().handlers = []
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    # --- 1. SETUP LOGGING & PATHS ---
    os.makedirs(CONFIG['FILE_PATHS']['OUTPUT_DIR'], exist_ok=True)
    sowing_manager = DynamicSowingManager()

    # --- 2. CONSOLIDATED DATA LOADING ---
    logging.info("=" * 70 + "\nLoading and merging ALL data sources (V3 WORKFLOW)...\n" + "=" * 70)
    try:
        df_yield = pd.read_csv(CONFIG['FILE_PATHS']['YIELD_DATA'], dtype={'district_no': str})
        df_yield.rename(columns={'yield': 'kreisYield'}, inplace=True)

        df_static_physics = pd.read_csv(CONFIG['FILE_PATHS']['STATIC_SOIL_FEATURES'], dtype={'district_no': str})

        gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH)
        gdf_districts['latitude'] = gdf_districts.geometry.centroid.y
        gdf_districts['longitude'] = gdf_districts.geometry.centroid.x
        gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)

        df_wav = pd.read_csv(CONFIG['FILE_PATHS']['INITIAL_CONDITIONS'], dtype={'district_no': str})

        df_static_base = pd.merge(df_static_physics, gdf_districts[['district_no', 'latitude', 'longitude']],
                                  on='district_no')
        df_static_all = pd.merge(df_yield, df_static_base, on='district_no',
                                 how='inner')
        df_static_all = pd.merge(df_static_all, df_wav,
                                 on=['year', 'district_no'], how='left')

        missing_wav = df_static_all['WAV'].isna()
        if missing_wav.any():
            logging.warning(
                f"{missing_wav.sum()} rows have missing WAV values. "
                "Check year ranges. Filling with default 10.0.")
            df_static_all.loc[missing_wav, 'WAV'] = 10.0

        df_seas5_all = pd.read_csv(
            CONFIG['FILE_PATHS']['SEAS5_MEMBER_FEATURES'],
            dtype={'district_no': str})
        valid_combinations = df_static_all[
            ['year', 'district_no']].drop_duplicates()
        df_seas5_all = pd.merge(df_seas5_all, valid_combinations,
                                on=['year', 'district_no'], how='inner')
        logging.info(
            f"Proceeding with {len(df_static_all)} valid district-year records.")

    except FileNotFoundError as e:
        logging.error(
            f"FATAL: A required data file was not found. Error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: Error during data loading: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. LOAD WEATHER & CROP DATA ---
    logging.info("Loading auxiliary data (weather, crop)...")
    full_hist_weather_df = pd.concat([
        pd.read_csv(f, parse_dates=['date'], dtype={'district_no': str}) for f in
        CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'].glob("*.csv")
    ], ignore_index=True)
    with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
        crop_params_all = yaml.safe_load(f)['CropParameters']
    final_params_dict = {**crop_params_all.get('GenericC3', {}), **crop_params_all['EcoTypes']['sugarbeet'],
                         **crop_params_all['Varieties']['Sugarbeet_601']}
    cropdata = ParameterDict()
    for key, val in final_params_dict.items():
        if key not in ['Metadata', '<<'] and isinstance(val, list) and len(val) > 0:
            cropdata.add_variable(key, val[0])

    # --- 4. RUN V2 INPUT VALIDATION ---
    if not analyze_v2_model_inputs(df_static_all, cropdata):
        logging.error("Input data analysis failed. Aborting pipeline.")
        sys.exit(1)

    # --- 5. MAIN SIMULATION LOOP ---
    all_hist_results = []
    all_fcst_results = []
    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        logging.info("=" * 70 + f"\nPROCESSING YEAR: {year}\n" + "=" * 70)
        df_static_year = df_static_all[df_static_all['year'] == year].copy()
        df_seas5_year = df_seas5_all[df_seas5_all['year'] == year].copy()

        hist_weather_path = (
            CONFIG['FILE_PATHS']['HISTORICAL_DAILY_WEATHER_DIR'] /
            f"historical_daily_weather_era5_{year}.csv")
        try:
            df_daily_hist_year = pd.read_csv(
                hist_weather_path, parse_dates=['date'],
                dtype={'district_no': str})
        except FileNotFoundError:
            logging.warning(f"Weather for {year} not found. Skipping year.")
            continue

        if CONFIG['DISTRICT_LIMIT'] is not None:
            limited_districts = df_static_year['district_no'].unique()[
                :CONFIG['DISTRICT_LIMIT']]
            df_static_year = df_static_year[
                df_static_year['district_no'].isin(limited_districts)]
            df_seas5_year = df_seas5_year[
                df_seas5_year['district_no'].isin(limited_districts)]
        if df_static_year.empty:
            logging.warning(f"Missing static data for {year}. Skipping.")
            continue

        past_weather_df = full_hist_weather_df[
            full_hist_weather_df['year'] < year].copy()
        district_weather_generators = {}
        expert_districts = set()
        for district_no in tqdm(
                df_static_year['district_no'].unique(),
                desc=f"Fitting Expert WGs for {year}"):
            district_past_weather = past_weather_df[
                past_weather_df['district_no'] == district_no]
            min_years = CONFIG['ANALOG_YEAR_CONFIG']['MIN_YEARS_FOR_FIT']
            if len(district_past_weather['year'].unique()) < min_years:
                wg_expert = WeatherGenerator()
                wg_expert.fit(district_past_weather)
                district_weather_generators[district_no] = wg_expert
            else:
                expert_districts.add(district_no)
                climatology = district_past_weather.groupby('month')[
                    ['tmin', 'tmax', 'precip']].mean()
                climatology['temp'] = (
                    climatology['tmin'] + climatology['tmax']) / 2
                target_forecast = df_seas5_year[
                    df_seas5_year['district_no'] == district_no]
                target_anomalies = {
                    m: {
                        'temp': target_forecast.get(
                            f'spring_temp_anomaly_forecast'
                            if m in [3, 4, 5]
                            else f'summer_temp_anomaly_forecast',
                            pd.Series(0.0)
                        ).mean(),
                        'precip': target_forecast.get(
                            f'spring_precip_anomaly_forecast'
                            if m in [3, 4, 5]
                            else f'summer_precip_anomaly_forecast',
                            pd.Series(0.0)
                        ).mean()
                    } for m in range(3, 11)
                }
                target_weather = climatology.copy()
                for month in range(3, 11):
                    if month in target_weather.index:
                        target_weather.loc[month, 'temp'] += \
                            target_anomalies[month]['temp']
                        target_weather.loc[month, 'precip'] += \
                            target_anomalies[month]['precip']
                yearly_avg = district_past_weather.groupby(
                    ['year', 'month'])[['precip', 'tmin', 'tmax']].mean().reset_index()
                yearly_avg['temp'] = (
                    yearly_avg['tmin'] + yearly_avg['tmax']) / 2
                hist_pivot = yearly_avg.pivot_table(
                    index='year', columns='month', values=['temp', 'precip'])
                hist_pivot.columns = [
                    f'{val}_{month}' for val, month in hist_pivot.columns]
                target_series = {
                    f'{val}_{m}': target_weather.loc[m, val]
                    for m in range(3, 11)
                    if m in target_weather.index
                    for val in ['temp', 'precip']
                }
                common_cols = hist_pivot.columns.intersection(
                    target_series.keys())
                if not common_cols.any():
                    wg_expert = WeatherGenerator()
                    wg_expert.fit(district_past_weather)
                    district_weather_generators[district_no] = wg_expert
                    continue
                aligned_target = pd.Series(target_series)[common_cols]
                distances = np.sqrt(np.sum(
                    (hist_pivot[common_cols].dropna() - aligned_target)**2,
                    axis=1)).sort_values()
                num_analogs = CONFIG['ANALOG_YEAR_CONFIG']['NUM_ANALOGS']
                analog_years = distances.head(num_analogs).index.tolist()
                analog_weather_data = district_past_weather[
                    district_past_weather['year'].isin(analog_years)]
                wg_expert = WeatherGenerator()
                wg_expert.fit(analog_weather_data)
                district_weather_generators[district_no] = wg_expert

        dynamic_sowing_dates = {}
        for district_no in df_static_year['district_no'].unique():
            district_weather_for_sowing = df_daily_hist_year[df_daily_hist_year['district_no'] == district_no]
            if not district_weather_for_sowing.empty:
                sowing_date = sowing_manager.find_sowing_date(district_weather_for_sowing)
                dynamic_sowing_dates[district_no] = sowing_date

        df_hist = run_historical_simulation(df_static_year, df_daily_hist_year, cropdata, year, CONFIG,
                                            dynamic_sowing_dates)
        if not df_hist.empty: all_hist_results.append(df_hist)

        df_fcst = run_forecast_simulation(
            df_static_year, df_seas5_year, district_weather_generators,
            cropdata, year, CONFIG, expert_districts)
        if not df_fcst.empty:
            all_fcst_results.append(df_fcst)

    # --- 6. FINAL ANALYSIS ---
    if all_hist_results:
        final_hist_df = pd.concat(all_hist_results, ignore_index=True)
        final_hist_df.dropna(inplace=True)

        if len(final_hist_df) > 1:
            mae = mean_absolute_error(
                final_hist_df['actual_yield'],
                final_hist_df['lintul_yield_perfect_weather'])
            r2 = r2_score(
                final_hist_df['actual_yield'],
                final_hist_df['lintul_yield_perfect_weather'])
            logging.info(
                "\n" + "=" * 50 +
                f"\nFINAL HISTORICAL PERFORMANCE (DYNAMIC SOWING)\n"
                f"MAE: {mae:.2f}\nR2: {r2:.3f}\n" + "=" * 50)
        else:
            logging.warning(
                "Not enough simulation results to calculate final "
                "performance metrics.")

    else:
        logging.error("No simulation results were generated across all years.")

    logging.info("\n" + "=" * 70 + "\n✓ PIPELINE COMPLETED.\n" + "=" * 70)
