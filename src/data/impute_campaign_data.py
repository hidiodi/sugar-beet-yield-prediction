import pandas as pd
import os

# --- Define File Paths ---
raw_data_path = 'data/01_raw/HarvestData/'
campaign_history_file = os.path.join(raw_data_path, 'campaign_history.csv')
factory_mapping_file = os.path.join(raw_data_path, 'factory_mapping.csv')

intermediate_data_path = 'data/02_intermediate/'
output_file = os.path.join(intermediate_data_path, 'campaign_history_imputed.csv')

os.makedirs(intermediate_data_path, exist_ok=True)

# --- Step 1: Load the Data ---
try:
    factory_mapping = pd.read_csv(factory_mapping_file)
    campaign_history = pd.read_csv(campaign_history_file)
    print("Successfully loaded data files.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("\nPlease ensure you are running this script from the project's root directory.")
    exit()

# --- Step 2: Add Company Information ---
campaign_history_merged = pd.merge(campaign_history, factory_mapping, on='factory_id')
print("Merged factory mapping information.")

# Convert date columns to datetime objects for calculations
campaign_history_merged['campaign_start_date'] = pd.to_datetime(campaign_history_merged['campaign_start_date'], errors='coerce')
campaign_history_merged['campaign_end_date'] = pd.to_datetime(campaign_history_merged['campaign_end_date'], errors='coerce')

# --- Step 3: Calculate Company Averages ---
print("Calculating company averages by year...")
merged_copy = campaign_history_merged.copy()
merged_copy['start_dayofyear'] = merged_copy['campaign_start_date'].dt.dayofyear
merged_copy['end_dayofyear'] = merged_copy['campaign_end_date'].dt.dayofyear

company_averages = merged_copy.groupby(['parent_company', 'year']).agg(
    avg_start_day=('start_dayofyear', 'mean'),
    avg_end_day=('end_dayofyear', 'mean')
).reset_index()

# --- Step 4: Fill the Gaps ---
print("Imputing missing start and end dates...")
campaign_history_with_avg = pd.merge(
    campaign_history_merged,
    company_averages,
    on=['parent_company', 'year'],
    how='left'
)

def day_of_year_to_date(day, year):
    if pd.isna(day) or pd.isna(year):
        return pd.NaT
    return pd.to_datetime(f'{int(year)}-01-01') + pd.to_timedelta(round(day) - 1, unit='D')

# --- FIX for FutureWarning ---
# Only attempt to fill values if there are any missing dates to begin with.
missing_start_mask = campaign_history_with_avg['campaign_start_date'].isnull()
if missing_start_mask.any():
    print("Found and imputing missing start dates...")
    campaign_history_with_avg.loc[missing_start_mask, 'campaign_start_date'] = \
        campaign_history_with_avg[missing_start_mask].apply(
            lambda row: day_of_year_to_date(row['avg_start_day'], row['year']), axis=1
        )

missing_end_mask = campaign_history_with_avg['campaign_end_date'].isnull()
if missing_end_mask.any():
    print("Found and imputing missing end dates...")
    campaign_history_with_avg.loc[missing_end_mask, 'campaign_end_date'] = \
        campaign_history_with_avg[missing_end_mask].apply(
            lambda row: day_of_year_to_date(row['avg_end_day'], row['year']), axis=1
        )

# --- Step 5: Save the Result ---
# Clean up by dropping the temporary average columns before saving
campaign_history_imputed = campaign_history_with_avg.drop(columns=['avg_start_day', 'avg_end_day'], errors='ignore')

campaign_history_imputed.to_csv(output_file, index=False, date_format='%Y-%m-%d')

print(f"\nProcess complete. Imputed data saved to: {output_file}")