import pandas as pd
import os


def process_campaign_features(campaign_file_path):
    """
    Calculates national-level harvest campaign features for each year.
    """
    print("Processing campaign history data...")
    campaign_history = pd.read_csv(campaign_file_path)

    campaign_history['campaign_start_date'] = pd.to_datetime(campaign_history['campaign_start_date'], errors='coerce')
    campaign_history['campaign_end_date'] = pd.to_datetime(campaign_history['campaign_end_date'], errors='coerce')

    national_averages = campaign_history.groupby('year').agg(
        avg_start_date=('campaign_start_date', 'mean'),
        avg_end_date=('campaign_end_date', 'mean')
    ).reset_index()

    national_averages['national_campaign_start_day_of_year'] = national_averages['avg_start_date'].dt.dayofyear
    national_averages['national_campaign_end_day_of_year'] = national_averages['avg_end_date'].dt.dayofyear
    national_averages['national_campaign_duration'] = (
                national_averages['avg_end_date'] - national_averages['avg_start_date']).dt.days

    print("  -> Campaign features created successfully.")
    return national_averages[['year', 'national_campaign_start_day_of_year', 'national_campaign_end_day_of_year',
                              'national_campaign_duration']]


def process_producer_prices(producer_price_file):
    """
    Loads and transforms the producer price index for sugar beets.
    This function is specifically designed to handle the comma-delimited format
    where numbers like '91,4' are split into two columns.
    """
    print("Processing producer price data...")
    # Read the raw CSV content without headers
    df = pd.read_csv(producer_price_file, header=None, delimiter=',')

    # The first row is the header. Let's grab it.
    header = list(df.iloc[0])

    # The actual data is in the following rows
    data = df.iloc[1:]

    # Find the row containing "Zuckerrüben"
    # Column 1 contains the description
    sugar_beet_raw_row = data[data[1].str.contains("Zuckerrüben")]

    if sugar_beet_raw_row.empty:
        raise ValueError("Could not find 'Zuckerrüben' in the producer price file.")

    # Prepare data for the new, clean DataFrame
    clean_data = {'year': [], 'producer_price_index': []}

    # Iterate through the YEAR columns. They start at column index 2 and jump by 2.
    # e.g., index 2 is '2015', index 4 is '2016', etc.
    for i in range(2, len(header), 2):
        year = int(header[i])

        # The integer part is at index i, the decimal part is at i+1
        integer_part = pd.to_numeric(sugar_beet_raw_row.iloc[0, i])
        decimal_part = pd.to_numeric(sugar_beet_raw_row.iloc[0, i + 1])

        full_value = integer_part + (decimal_part / 10.0)

        clean_data['year'].append(year)
        clean_data['producer_price_index'].append(full_value)

    final_df = pd.DataFrame(clean_data)
    print("  -> Producer prices processed successfully.")
    return final_df


def process_input_prices(input_price_file):
    """
    Loads and transforms key input price indices (Energy, Fertilizer).
    This function also handles the comma-delimited format where numbers are split.
    """
    print("Processing input price data...")
    df = pd.read_csv(input_price_file, header=None, delimiter=',')

    # The first row is the header
    header = list(df.iloc[0])

    # The rest is data
    data = df.iloc[1:]

    # Find the rows we need
    key_inputs_pattern = "Energie und Schmierstoffe|Düngemittel"
    input_prices_raw = data[data[1].str.contains(key_inputs_pattern)]

    if input_prices_raw.empty:
        raise ValueError("Could not find required descriptions in the input price file.")

    # We will build a clean "long" format table first
    long_format_data = []

    # Iterate over each required row (Energy, Fertilizer)
    for index, row in input_prices_raw.iterrows():
        description = row[1]  # The name of the category
        # Iterate through the PERIOD columns, which start at index 2 and jump by 2
        for i in range(2, len(header), 2):
            period = header[i]  # e.g., '01/2018'

            integer_part = pd.to_numeric(row.iloc[i])
            decimal_part = pd.to_numeric(row.iloc[i + 1])

            price_index = integer_part + (decimal_part / 10.0)
            year = int(period.split('/')[1])

            long_format_data.append([year, description, price_index])

    # Convert to a DataFrame for easy processing
    melted_df = pd.DataFrame(long_format_data, columns=['year', 'Description', 'price_index'])

    # Calculate the annual average from the quarterly data
    annual_avg_prices = melted_df.groupby(['year', 'Description'])['price_index'].mean().reset_index()

    # Pivot the table to get one row per year
    final_input_prices = annual_avg_prices.pivot(index='year', columns='Description',
                                                 values='price_index').reset_index()
    final_input_prices.rename(
        columns={'Energie und Schmierstoffe': 'energy_price_index', 'Düngemittel': 'fertilizer_price_index'},
        inplace=True)

    print("  -> Input prices processed successfully.")
    return final_input_prices


def main():
    """Main function to orchestrate the data merging and saving."""

    # --- Define File Paths ---
    climate_yield_file = 'data/03_processed/final_dataset_with_advanced_features.csv'
    static_features_file = 'data/03_processed/static_features_districts_advanced.csv'
    campaign_file = 'data/01_raw/HarvestData/campaign_history.csv'
    producer_price_file = 'data/01_raw/61211-0002_de/61211-0001_de.csv'
    input_price_file = 'data/01_raw/61211-0002_de/61221-0003_de.csv'

    output_path = 'data/04_master/'
    output_file = os.path.join(output_path, 'master_dataset.csv')
    os.makedirs(output_path, exist_ok=True)

    # --- Load and Process Data ---
    print("Loading base data...")
    climate_yield_data = pd.read_csv(climate_yield_file)
    static_features = pd.read_csv(static_features_file)

    campaign_features = process_campaign_features(campaign_file)
    producer_prices = process_producer_prices(producer_price_file)
    input_prices = process_input_prices(input_price_file)

    # --- Merge Datasets into Master Table ---
    print("\nMerging all datasets into a master table...")
    master_df = climate_yield_data.copy()

    master_df = pd.merge(master_df, static_features, on='district_no', how='left')
    master_df = pd.merge(master_df, campaign_features, on='year', how='left')
    master_df = pd.merge(master_df, producer_prices, on='year', how='left')
    master_df = pd.merge(master_df, input_prices, on='year', how='left')

    print("  -> Merging complete.")

    # --- Save Final Dataset ---
    master_df.to_csv(output_file, index=False)
    print(f"\nMaster dataset successfully created and saved to: {output_file}")
    print(f"Master dataset has {master_df.shape[0]} rows and {master_df.shape[1]} columns.")


if __name__ == '__main__':
    main()