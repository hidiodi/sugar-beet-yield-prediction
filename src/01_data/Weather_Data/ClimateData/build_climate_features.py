import pandas as pd
from pathlib import Path
import logging
import warnings

# ==============================================================================
# === CONFIGURATION ===
# ==============================================================================
# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths to your downloaded files
base_path = Path.cwd()
RAW_DATA_DIR = base_path / 'data/01_raw/climateIndices'
OUTPUT_DIR = base_path / 'data/02_intermediate/climateIndices'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- USES YOUR LOCAL FILES. QBO HAS BEEN REMOVED. ---
FILES = {
    'nao': RAW_DATA_DIR / 'norm.nao.monthly.b5001.current.ascii.table',
    'sca': RAW_DATA_DIR / 'scand.csv',
    'enso': RAW_DATA_DIR / 'meiv2.csv'
}


# ==============================================================================
# === PROCESSING FUNCTIONS (TAILORED TO YOUR FILES) ===
# ==============================================================================

def process_nao(file_path):
    """
    Parses your specific wide-format NOAA NAO monthly data file.
    """
    logging.info(f"Processing NAO data from: {file_path}")
    df = pd.read_csv(
        file_path, sep=r'\s+', header=None, skiprows=1,  # Skip the month name header
        names=['year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    )
    df_long = df.melt(id_vars=['year'], var_name='month', value_name='nao')
    df_long['month'] = pd.to_numeric(df_long['month'])
    return df_long


def process_simple_timeseries(file_path, value_col_name):
    """
    Parses your simple two-column CSVs (scand.csv and meiv2.csv).
    Format: Date, Value with a header row to skip.
    """
    logging.info(f"Processing {value_col_name} data from: {file_path}")
    df = pd.read_csv(file_path, skiprows=1, header=None, names=['date_str', value_col_name])
    df['date'] = pd.to_datetime(df['date_str'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df[['year', 'month', value_col_name]]


def calculate_winter_average(df, value_col):
    """
    Calculates the winter (Dec, Jan, Feb) average for a given index.
    """
    if df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=['year', f'{value_col}_winter_avg'])

    df = df.copy()
    df['forecast_year'] = df.apply(lambda row: row['year'] + 1 if row['month'] == 12 else row['year'], axis=1)
    winter_df = df[df['month'].isin([12, 1, 2])]
    winter_avg = winter_df.groupby('forecast_year')[value_col].mean().reset_index()
    winter_avg = winter_avg.rename(columns={'forecast_year': 'year', value_col: f'{value_col}_winter_avg'})
    return winter_avg


# ==============================================================================
# === MAIN EXECUTION ===
# ==============================================================================

def main():
    """
    Main function to load, process, and merge the three reliable climate indices.
    """
    # Step 1: Process each raw file into a standardized format
    df_nao = process_nao(FILES['nao'])
    df_sca = process_simple_timeseries(FILES['sca'], 'sca')
    df_enso = process_simple_timeseries(FILES['enso'], 'enso_mei')

    # Step 2: Calculate the winter average for each index
    nao_winter = calculate_winter_average(df_nao, 'nao')
    sca_winter = calculate_winter_average(df_sca, 'sca')
    enso_winter = calculate_winter_average(df_enso, 'enso_mei')

    # Step 3: Merge all features into a single DataFrame
    logging.info("Merging all indices into a single feature table.")
    years = pd.DataFrame({'year': range(1980, pd.Timestamp.now().year + 1)})

    final_df = pd.merge(years, nao_winter, on='year', how='left')
    final_df = pd.merge(final_df, sca_winter, on='year', how='left')
    final_df = pd.merge(final_df, enso_winter, on='year', how='left')

    # Sort and fill any minor gaps safely
    final_df = final_df.sort_values('year').reset_index(drop=True)
    if final_df.isnull().values.any():
        logging.warning("Missing values detected. Imputing with 0.0 as a neutral value.")
        final_df = final_df.fillna(0.0)

    # Step 4: Save the final, clean CSV file
    output_path = OUTPUT_DIR / 'long_range_climate_features.csv'
    final_df.to_csv(output_path, index=False)

    logging.info(f"✓ Successfully created climate features file at: {output_path}")
    print("\n--- Final Climate Features (Sample) ---")
    print(final_df.tail())
    print("\n")


if __name__ == "__main__":
    main()