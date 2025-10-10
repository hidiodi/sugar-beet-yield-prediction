import pandas as pd
import json
import sys


def process_district_file(filepath):
    """
    Processes the district JSON file to create a mapping from district ID to state name.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            features = data.get('features', [])
            if not features:
                f.seek(0)
                features = [json.loads(line) for line in f if line.strip()]
            return {int(f['id']): f['properties']['state'] for f in features}
    except Exception as e:
        print(f"Error: Could not parse '{filepath}'. Details: {e}", file=sys.stderr)
        sys.exit(1)


def load_and_process_economic_data(producer_price_file, input_price_file):
    """
    Loads, processes, and merges the separate economic data files into a single,
    analysis-ready DataFrame with annual data.
    """
    print("\nStep 2a: Loading and processing external economic data sources...")
    try:
        # --- Producer Prices (Annual Data) ---
        df_prod_raw = pd.read_csv(producer_price_file)
        # Filter for sugar beets, melt to long format
        df_prod = df_prod_raw[df_prod_raw['ID'] == 'LWPR-132'].melt(
            id_vars=['ID', 'Description'], var_name='year', value_name='producer_price_index'
        )
        df_prod['year'] = pd.to_numeric(df_prod['year'])
        df_prod = df_prod[['year', 'producer_price_index']]
        print(" -> Producer price data processed.")

        # --- Input Prices (Quarterly Data) ---
        df_input_raw = pd.read_csv(input_price_file)
        # Filter for fertilizer and energy
        df_input = df_input_raw[df_input_raw['ID'].isin(['LWBM-13', 'LWBM-12'])]
        # Melt to long format
        df_input_melted = df_input.melt(
            id_vars=['ID', 'Description'], var_name='period', value_name='price_index'
        )
        # Extract year and convert to numeric
        df_input_melted['year'] = pd.to_numeric(df_input_melted['period'].str.split('/').str[1], errors='coerce')
        df_input_melted.dropna(subset=['year'], inplace=True)  # Drop non-year columns
        df_input_melted['year'] = df_input_melted['year'].astype(int)

        # Calculate the annual average for each index
        df_annual_avg = df_input_melted.groupby(['year', 'ID'])['price_index'].mean().reset_index()

        # Pivot to get separate columns for each index
        df_input_final = df_annual_avg.pivot(index='year', columns='ID', values='price_index').reset_index()
        df_input_final.rename(columns={
            'LWBM-13': 'fertilizer_price_index',
            'LWBM-12': 'energy_price_index'
        }, inplace=True)
        print(" -> Input price data (fertilizer, energy) processed.")

        # --- Merge economic datasets ---
        df_economic = pd.merge(df_prod, df_input_final, on='year', how='outer')
        print(" -> Economic datasets successfully merged.")
        return df_economic

    except Exception as e:
        print(f"Error: Failed to load or process economic data files. Details: {e}", file=sys.stderr)
        return None


def analyze_missing_data(df, stage=""):
    """Reusable function to analyze and report missing data."""
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        print(f"\nAnalysis of missing values {stage}:")
        missing_df = pd.DataFrame({
            'Missing Count': missing_values,
            'Missing Percentage (%)': (missing_values / len(df) * 100).round(2)
        })
        print(missing_df.to_string())
    else:
        print(f"\nNo missing values found in the dataset {stage}.")


def impute_campaign_data(df):
    """
    Imputes missing national campaign data using the mean of the respective columns.
    """
    print("\nStep 6a: Imputing missing national campaign data...")
    campaign_cols = [
        'national_campaign_start_day_of_year',
        'national_campaign_end_day_of_year',
        'national_campaign_duration'
    ]

    imputed_something = False
    for col in campaign_cols:
        if col in df.columns and df[col].isnull().any():
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
            print(f" -> Filled missing values in '{col}' with the mean ({mean_value:.2f}).")
            imputed_something = True

    if not imputed_something:
        print(" -> No missing campaign data found to impute.")

    return df


def main():
    """
    Main function to execute the data merging and transformation workflow.
    """
    # --- Configuration with corrected file paths ---
    master_file = 'data/04_master/master_dataset.csv'
    state_level_file = 'data/01_raw/final_data2.csv'
    districts_file = 'data/01_raw/districts_official.geojson'
    producer_price_file = 'data/01_raw/61211-0002_de/61211-0001_de.csv'
    input_price_file = 'data/01_raw/61211-0002_de/61221-0003_de.csv'
    output_file = 'data/04_master/merged_final_dataset.csv'

    # --- Step 1: Load base CSV datasets ---
    print("Step 1: Loading base CSV datasets...")
    try:
        master_df = pd.read_csv(master_file)
        state_df = pd.read_csv(state_level_file, na_values=['-', '/', '.'])
        print(f"'{master_file}' and '{state_level_file}' loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Load external economic data ---
    df_economic = load_and_process_economic_data(producer_price_file, input_price_file)
    if df_economic is None:
        sys.exit(1)

    # --- Step 3: Map districts to states ---
    print("\nStep 3: Processing districts file to map districts to states...")
    district_to_state_map = process_district_file(districts_file)
    master_df['state'] = master_df['district_no'].map(district_to_state_map)
    print("District-to-state mapping completed.")

    # --- Step 4: Merge all datasets ---
    print("\nStep 4: Merging all data sources...")
    master_df['year'] = master_df['year'].astype(int)
    state_df['Year'] = state_df['Year'].astype(int)

    # Merge master with state-level data
    merged_df = pd.merge(
        master_df, state_df,
        left_on=['year', 'state'], right_on=['Year', 'Land'],
        how='left'
    )

    # Drop original economic columns to avoid conflicts, then merge new economic data
    cols_to_drop = ['producer_price_index', 'fertilizer_price_index', 'energy_price_index']
    merged_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    merged_df = pd.merge(merged_df, df_economic, on='year', how='left')
    print("All data sources merged successfully.")

    # --- Step 5: Transform data and finalize features ---
    print("\nStep 5: Transforming data and calculating new features...")
    merged_df.rename(columns={'yield': 'kreisYield'}, inplace=True)

    cols_to_convert = ['kreisYield', 'Yield_dt/ha', 'Field_ha']
    for col in cols_to_convert:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    initial_rows = len(merged_df)
    merged_df.dropna(subset=['kreisYield'], inplace=True)
    rows_dropped = initial_rows - len(merged_df)
    if rows_dropped > 0:
        print(f" -> Dropped {rows_dropped} rows where 'kreisYield' was not provided.")

    epsilon = 1e-6
    merged_df['kreisField_ha'] = (merged_df['kreisYield'] / (merged_df['Yield_dt/ha'] + epsilon)) * merged_df[
        'Field_ha']
    print(" -> Calculated 'kreisField_ha' feature.")

    # --- Step 6: Analyze data BEFORE any potential imputation ---
    analyze_missing_data(merged_df, stage="BEFORE imputation")

    # --- Step 6a: Impute missing campaign data ---
    merged_df = impute_campaign_data(merged_df)

    # --- Step 7: Select and order the final columns ---
    print("\nStep 7: Finalizing columns for the output file...")
    final_columns = [
        'district_no', 'year', 'Field_ha', 'Harvested_t', 'Yield_dt/ha', 'kreisYield', 'kreisField_ha',
        'precip_total_peak_growth', 'temp_mean_peak_growth', 'heat_stress_days_peak_growth',
        'temp_mean_early_growth', 'solar_rad_peak_growth', 'avg_elevation', 'avg_soil_pawc',
        'national_campaign_start_day_of_year', 'national_campaign_end_day_of_year',
        'national_campaign_duration', 'producer_price_index', 'fertilizer_price_index',
        'energy_price_index'
    ]

    final_df = merged_df[final_columns]
    print("Columns selected and ordered.")

    # --- Step 8: Save the result ---
    print(f"\nStep 8: Saving the transformed data to '{output_file}'...")
    final_df.to_csv(output_file, index=False)

    # --- Step 9: Final Analysis ---
    analyze_missing_data(final_df, stage="FINAL")

    print("\n----------------------------------------")
    print("Script finished successfully!")
    print(f"Output saved to: {output_file}")
    print("----------------------------------------")


if __name__ == '__main__':
    main()

