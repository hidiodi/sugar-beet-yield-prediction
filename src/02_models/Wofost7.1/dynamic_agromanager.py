# File: src/02_models/Wofost7.1/dynamic_agromanager.py
# Description: Implements a rule-based dynamic sowing date manager for WOFOST.

import datetime
import pandas as pd
import logging

class DynamicSowingManager:
    """
    Determines a dynamic sowing date based on weather conditions.
    """
    def __init__(self, sowing_window_start_month=3, sowing_window_start_day=15,
                 sowing_window_end_month=4, sowing_window_end_day=30,
                 temp_threshold_c=7.0, temp_avg_period_days=7):
        """
        Initializes the rules for finding the sowing date.

        :param sowing_window_start_month: Earliest month for sowing.
        :param sowing_window_start_day: Earliest day for sowing.
        :param sowing_window_end_month: Latest month for sowing.
        :param sowing_window_end_day: Latest day for sowing.
        :param temp_threshold_c: The moving average temperature that must be exceeded.
        :param temp_avg_period_days: The number of days for the temperature moving average.
        """
        self.start_month = sowing_window_start_month
        self.start_day = sowing_window_start_day
        self.end_month = sowing_window_end_month
        self.end_day = sowing_window_end_day
        self.threshold = temp_threshold_c
        self.period = temp_avg_period_days

    def find_sowing_date(self, weather_df_year: pd.DataFrame) -> datetime.date:
        """
        Finds the first suitable sowing date within the year's weather data.

        :param weather_df_year: DataFrame with daily weather for the entire year.
                                Must contain 'date', 'tmin', 'tmax'.
        :return: The calculated sowing date as a datetime.date object.
        """
        df = weather_df_year.copy()
        df['mean_temp'] = (df['tmin'] + df['tmax']) / 2
        df['temp_ma'] = df['mean_temp'].rolling(window=self.period, min_periods=self.period).mean()

        try:
            year = df['date'].dt.year.iloc[0]
            window_start = datetime.date(year, self.start_month, self.start_day)
            window_end = datetime.date(year, self.end_month, self.end_day)
        except IndexError:
            logging.error("DynamicSowingManager: weather_df_year is empty. Cannot determine year.")
            return datetime.date(2000, self.start_month, self.start_day) # Return a fallback

        # Filter for days within the window that meet the temperature criteria
        sow_mask = (
            (df['date'].dt.date >= window_start) &
            (df['date'].dt.date <= window_end) &
            (df['temp_ma'] >= self.threshold)
        )
        potential_sow_days = df[sow_mask]

        if not potential_sow_days.empty:
            sowing_date = potential_sow_days['date'].dt.date.iloc[0]
            logging.debug(f"Dynamic sowing date found: {sowing_date} (Threshold {self.threshold}C met).")
            return sowing_date
        else:
            # Fallback: if no day meets the criteria, sow on the last day of the window.
            logging.debug(f"No suitable day found. Sowing on fallback date: {window_end}.")
            return window_end