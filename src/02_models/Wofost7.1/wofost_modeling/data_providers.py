import logging
import pandas as pd
from pcse.base import WeatherDataProvider
from pcse.util import penman_monteith
from .parameters import ParameterDict
from src import config

CONFIG = config.WOFOST_CONFIG

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

                et0_mm = penman_monteith(
                    day, self.latitude, self.elevation, tmin, tmax,
                    irrad_kj_m2_day, vap_hpa, wind)
                et0_cm = et0_mm / 10.0

                self.store[(day, 0)] = ParameterDict({
                    'DAY': day, 'LAT': self.latitude, 'TMIN': tmin,
                    'TMAX': tmax, 'RAIN': precip_cm,
                    'IRRAD': irrad_j_m2_day, 'VAP': vap_hpa,
                    'WIND': wind, 'E0': et0_cm, 'ES0': et0_cm,
                    'ET0': et0_cm, 'SNOWDEPTH': 0.0
                })
            except Exception as e:
                logging.error(
                    f"CRITICAL: Failed processing weather row for date: "
                    f"{row.get('date')}. Error: {e}")
                raise e
