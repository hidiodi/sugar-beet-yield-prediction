import pandas as pd

# Define common German statistical null/missing value placeholders
MISSING_VALUES = ['/', '-', '...', ':', '.', '']

# File paths (adjust as needed)
INPUT_FILE = 'data/06_test/41241-01-03-4-B.csv'
OUTPUT_FILE = 'data/06_test/testYield_clean.csv'

# --- 1. Read and Clean Data ---
try:
    # Read the data, specifying the delimiter, encoding, and the list of NA values
    df = pd.read_csv(
        INPUT_FILE,
        sep=';',
        encoding='latin-1',  # Use 'latin-1' to handle special characters
        na_values=MISSING_VALUES # Treat these strings as NaN
    )
except UnicodeDecodeError:
    # Fallback to cp1252 if latin-1 fails
    df = pd.read_csv(
        INPUT_FILE,
        sep=';',
        encoding='cp1252',
        na_values=MISSING_VALUES
    )

# --- 2. Select and Filter Columns ---
# Select only the required columns
df_transformed = df[['Jahr', 'ID', 'Zuckerrben']]

# --- 3. Drop Null Values ---
# Use .dropna() to remove any row where 'Zuckerrüben' is NaN (which now includes '/', '-')
# Since we are only interested in these three columns, using .dropna() on the whole
# df_transformed is sufficient.
df_cleaned = df_transformed.dropna()

# --- 4. Output Results ---

# Print the result to the console for verification
print("--- Cleaned Data Head ---")
print(df_cleaned.head())
print("\n--- Cleaned Data (CSV Output) ---")
print(df_cleaned.to_csv(index=False, sep=';'))

# Save the final cleaned DataFrame to a new CSV file
df_cleaned.to_csv(OUTPUT_FILE, index=False, sep=';')

print(f"\nSuccessfully cleaned and saved data to {OUTPUT_FILE}")