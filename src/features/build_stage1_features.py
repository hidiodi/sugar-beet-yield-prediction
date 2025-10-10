import pandas as pd
import xarray as xr
import logging
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def lag_economic_features(df, lag_period=1):
    """
    Creates lagged versions of economic features.
    """
    logging.info(f"Lagging economic features by {lag_period} year(s)...")
    eco_cols = ['pflanzliche_erzeugung', 'zuckerrben', 'dngemittel', 'pflanzenschutzmittel', 'saat_und_pflanzgut',
                'energie_und_schmierstoffe']
    cols_to_lag = [col for col in eco_cols if col in df.columns]
    if not cols_to_lag:
        logging.warning("Economic columns for lagging not found in the dataset. Skipping this step.")
        return df
    df_sorted = df.sort_values(by=['district_no', 'year'])
    for col in cols_to_lag:
        df_sorted[f'{col}_lag{lag_period}'] = df_sorted.groupby('district_no')[col].shift(lag_period)
    return df_sorted


def engineer_antecedent_weather(df, agera5_path):
    """
    Calculates and merges antecedent winter weather anomalies.
    """
    logging.info("Starting antecedent winter weather feature engineering...")
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logging.error("'latitude' or 'longitude' not found in the data. Cannot calculate weather features.")
        return df

    try:
        agera5_ds = xr.open_dataset(agera5_path)
        temp_var_name = 'Temperature_Air_2m_Mean_24h'
        precip_var_name = 'Precipitation_Flux'
    except Exception as e:
        logging.error(f"Could not load AgERA5 dataset. Error: {e}")
        return df

    logging.info("Calculating long-term winter climate normals...")
    start_time = time.time()
    years = pd.to_datetime(agera5_ds.time.values).year.unique()
    seasonal_temp_means, seasonal_precip_sums = [], []
    for year in tqdm(years, desc="Calculating Seasonal Normals"):
        start_date, end_date = f"{year - 1}-10-01", f"{year}-03-31"
        try:
            winter_period = agera5_ds.sel(time=slice(start_date, end_date))
            if winter_period.time.size > 0:
                seasonal_temp_means.append(winter_period[temp_var_name].mean(dim='time'))
                seasonal_precip_sums.append(winter_period[precip_var_name].sum(dim='time'))
        except (KeyError, ValueError):
            continue
    if not seasonal_temp_means:
        logging.error("Could not calculate seasonal statistics. Aborting weather feature engineering.")
        return df
    winter_temp_normal = xr.concat(seasonal_temp_means, dim='season').mean(dim='season')
    winter_precip_normal = xr.concat(seasonal_precip_sums, dim='season').mean(dim='season')
    winter_temp_normal.load()
    winter_precip_normal.load()
    logging.info(f"Climate normals calculated in {time.time() - start_time:.2f} seconds.")

    unique_locations = df[['district_no', 'latitude', 'longitude']].dropna().drop_duplicates().set_index('district_no')
    location_coords = xr.Dataset.from_dataframe(unique_locations)
    temp_normals_at_locs = winter_temp_normal.interp(lon=location_coords.longitude, lat=location_coords.latitude, method="linear")
    precip_normals_at_locs = winter_precip_normal.interp(lon=location_coords.longitude, lat=location_coords.latitude, method="linear")
    unique_locations['temp_normal'] = temp_normals_at_locs.to_pandas()
    unique_locations['precip_normal'] = precip_normals_at_locs.to_pandas()

    all_years_features = []
    unique_district_years = df[['year', 'district_no']].drop_duplicates()
    for _, row in tqdm(unique_district_years.iterrows(), total=len(unique_district_years), desc="Yearly Anomalies"):
        year, district_no = row['year'], row['district_no']
        if district_no not in unique_locations.index:
            continue
        start_date, end_date = f"{year - 1}-10-01", f"{year}-03-31"
        try:
            winter_period_ds = agera5_ds.sel(time=slice(start_date, end_date))
            if winter_period_ds.time.size == 0:
                continue
        except KeyError:
            continue
        lat = unique_locations.loc[district_no, 'latitude']
        lon = unique_locations.loc[district_no, 'longitude']
        actual_winter_temp = winter_period_ds[temp_var_name].interp(lat=lat, lon=lon, method="linear").mean(dim='time').compute().item()
        actual_winter_precip = winter_period_ds[precip_var_name].interp(lat=lat, lon=lon, method="linear").sum(dim='time').compute().item()
        temp_normal_val = unique_locations.loc[district_no, 'temp_normal']
        precip_normal_val = unique_locations.loc[district_no, 'precip_normal']
        temp_anomaly = actual_winter_temp - temp_normal_val
        precip_anomaly = (actual_winter_precip / (precip_normal_val + 1e-6)) - 1.0
        all_years_features.append({
            'year': year,
            'district_no': district_no,
            'winter_temp_anomaly': temp_anomaly,
            'winter_precip_anomaly': precip_anomaly
        })

    if not all_years_features:
        logging.warning("No antecedent weather features could be calculated.")
        return df

    df_weather_features = pd.DataFrame(all_years_features)
    df_weather_features['district_no'] = df_weather_features['district_no'].astype(str).str.zfill(5)
    return pd.merge(df, df_weather_features, on=['year', 'district_no'], how='left')


def main():
    logging.info("--- Starting Final Data Pipeline ---")

    master_path = Path('data/04_master/master_dataset.csv')
    merged_final_path = Path('data/04_master/merged_final_dataset.csv')
    agera5_file = Path('data/02_intermediate/agera5_germany_2017_2024_merged.nc')
    output_path = Path('data/05_model_input/')
    output_file = output_path / 'final_imputed_dataset.csv'
    output_path.mkdir(exist_ok=True)

    try:
        df1 = pd.read_csv(master_path)
        df1['district_no'] = df1['district_no'].astype(str).str.zfill(5)
        df2 = pd.read_csv(merged_final_path)
        df2['district_no'] = df2['district_no'].astype(str).str.zfill(5)
    except FileNotFoundError as e:
        logging.error(f"FATAL: Could not load input file. Error: {e}")
        return

    df_combined = pd.merge(df1, df2, on=['district_no', 'year'], how='outer', suffixes=('_A', '_B'))
    for col in df_combined.columns:
        if col.endswith('_B'):
            base_col = col[:-2]
            col_A = base_col + '_A'
            if col_A in df_combined.columns:
                df_combined[base_col] = df_combined[col].combine_first(df_combined[col_A])
                df_combined.drop(columns=[col_A, col], inplace=True)
    logging.info(f"Successfully merged data. Shape of combined data: {df_combined.shape}")

    # --- Feature Engineering ---
    df_featured = lag_economic_features(df_combined)
    df_featured = engineer_antecedent_weather(df_featured, agera5_file)
    logging.info(f"Feature engineering complete. Shape is now: {df_featured.shape}")

    # --- Derived / Normalized Targets ---
    if {'kreisYield', 'kreisField_ha'}.issubset(df_featured.columns):
        df_featured['yield_density'] = df_featured['kreisYield'] / (df_featured['kreisField_ha'] + 1e-6)
        logging.info("Added yield_density (kreisYield normalized by kreisField_ha).")

    # --- Create Composite Lag Indices to Reduce Multicollinearity ---
    lag_vars = ['dngemittel_lag1', 'saat_und_pflanzgut_lag1', 'energie_und_schmierstoffe_lag1']
    if all(v in df_featured.columns for v in lag_vars):
        df_featured['input_price_index'] = df_featured[lag_vars].mean(axis=1)
        logging.info("Created composite input_price_index feature.")
        df_featured.drop(columns=lag_vars, inplace=True)

    # --- Remove redundant campaign columns ---
    drop_campaign_cols = ['national_campaign_start_day_of_year', 'national_campaign_end_day_of_year']
    df_featured.drop(columns=drop_campaign_cols, inplace=True, errors='ignore')

    # --- Drop unnecessary or target-leaking columns ---
    cols_to_remove = [
        'yield', 'Yield_dt/ha', 'Field_ha', 'Harvested_t', 'state_name', 'latitude', 'longitude',
        'pflanzliche_erzeugung', 'zuckerrben', 'dngemittel', 'energie_und_schmierstoffe',
        'pflanzenschutzmittel', 'saat_und_pflanzgut'
    ]
    df_featured.drop(columns=cols_to_remove, inplace=True, errors='ignore')
    logging.info(f"Removed intermediate columns. Kept {len(df_featured.columns)} columns.")

    # --- Log-transform skewed features ---
    skewed_vars = ['pflanzenschutzmittel_lag1', 'pflanzliche_erzeugung_lag1']
    for var in skewed_vars:
        if var in df_featured.columns:
            df_featured[f'{var}_log'] = np.log1p(df_featured[var])
            logging.info(f"Log-transformed {var}.")

    # --- Imputation ---
    numeric_cols = df_featured.select_dtypes(include=np.number).columns.tolist()
    df_imputed = df_featured.copy()
    df_imputed[numeric_cols] = df_imputed.groupby('district_no')[numeric_cols].transform(lambda x: x.fillna(x.mean()))
    df_imputed.fillna(df_imputed.mean(numeric_only=True), inplace=True)
    missing_after_impute = df_imputed.isnull().sum().sum()
    if missing_after_impute > 0:
        logging.warning(f"Warning: {missing_after_impute} missing values still remain after imputation.")
    else:
        logging.info("Successfully imputed all missing numeric values.")

    # --- Save Final Dataset ---
    df_imputed.to_csv(output_file, index=False)
    logging.info(f"\n--- SUCCESS: Final, imputed dataset created! ---")
    logging.info(f"Dataset saved to '{output_file}' with {df_imputed.shape[0]} rows.")
    logging.info(f"Final columns ({len(df_imputed.columns)} total): {df_imputed.columns.tolist()}")


if __name__ == '__main__':
    main()
