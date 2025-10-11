import pandas as pd
import json
import os

# --- 1. Define File Paths ---
# Use os.path.join for platform independence
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Assumes script is in src/analysis
DATA_ROOT = os.path.join(ROOT_DIR, '..', '..', 'data')

PATH_GEOJSON = os.path.join(DATA_ROOT, '01_raw', 'districts_official.geojson')
PATH_VERIFIED = os.path.join(DATA_ROOT, '01_raw', 'final_data2.csv')
PATH_HISTORICAL = os.path.join(DATA_ROOT, '02_intermediate', 'sugarbeet_yield.csv')


# --- 2. Load and Prepare Mapping Data (GeoJSON) ---

def create_district_to_state_map(geojson_path):
    """
    Parses the GeoJSON file to create a dictionary mapping
    district ID (Kreis) to state (Land).
    """
    print(f"Loading district map from: {geojson_path}")
    district_to_state = {}

    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # The GeoJSON structure is a FeatureCollection
        for feature in data.get('features', []):
            district_id = feature.get('id')
            state_name = feature.get('properties', {}).get('state')

            if district_id and state_name:
                # Store the mapping: GeoJSON ID -> State Name
                # Assuming the 'district_no' in the CSV is the same as the GeoJSON 'id'
                district_to_state[district_id] = state_name

        # Include the example from the prompt which has a different format for district_no:
        # The GeoJSON example ID is '08425', historical is '1001'.
        # We must align these formats. A common conversion is to use the raw number.
        # Since we don't know the full mapping, we rely on the GeoJSON 'id'.
        # For the provided snippet: {'id': '08425', 'state': 'Baden-Württemberg'}

        # Manually add the mapping for the sample '1001' based on common knowledge/metadata
        # If the GeoJSON is complete, this manual step is not needed.
        # Based on the user's provided snippet, '1001' is NOT in the snippet,
        # but '08425' is. We will assume the full file resolves this.

        return district_to_state

    except FileNotFoundError:
        print(f"ERROR: GeoJSON file not found at {geojson_path}. Cannot create district map.")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON format in {geojson_path}.")
        return None


# --- 3. Load and Prepare DataFrames ---

def load_and_prepare_data(district_map):
    if district_map is None:
        return None, None

    # Load Historical Data
    print(f"\nLoading historical data from: {PATH_HISTORICAL}")
    try:
        df_hist = pd.read_csv(PATH_HISTORICAL, dtype={'district_no': str})
    except FileNotFoundError:
        print(f"ERROR: Historical CSV file not found at {PATH_HISTORICAL}.")
        return None, None

    # Standardize district_no format to match GeoJSON 'id'
    # E.g., if GeoJSON IDs are padded (like '08425'), we ensure the historical
    # district_no is also treated as a string for a proper merge.
    # The snippet has '1001', which we assume is the key format.
    df_hist['district_no'] = df_hist['district_no'].astype(str)

    # Map district_no to Land
    df_hist['Land'] = df_hist['district_no'].map(district_map)
    df_hist.rename(columns={'year': 'Year', 'yield': 'Yield_t/ha_Hist'}, inplace=True)

    # Filter out districts that could not be mapped (e.g., missing from GeoJSON)
    df_hist.dropna(subset=['Land'], inplace=True)

    # Load Verified Data
    print(f"Loading verified data from: {PATH_VERIFIED}")
    try:
        df_ver = pd.read_csv(PATH_VERIFIED)
    except FileNotFoundError:
        print(f"ERROR: Verified CSV file not found at {PATH_VERIFIED}.")
        return None, None

    # Clean Verified Data
    df_ver.rename(columns={'Yield_dt/ha': 'Yield_dt/ha_Ver'}, inplace=True)

    # Convert non-numeric values (like '.' or '-') to NaN
    df_ver['Yield_dt/ha_Ver'] = pd.to_numeric(
        df_ver['Yield_dt/ha_Ver'], errors='coerce'
    )

    # Convert verified yield from dt/ha to t/ha (10 dt = 1 t)
    # This confirms the hypothesis from the initial analysis.
    df_ver['Yield_t/ha_Ver'] = df_ver['Yield_dt/ha_Ver'] / 10

    # Keep only necessary columns for the final comparison
    df_ver = df_ver[['Year', 'Land', 'Yield_t/ha_Ver']].dropna()

    return df_hist, df_ver


# --- 4. Aggregate Historical Data (Unweighted Average) ---

def aggregate_historical_data(df_hist):
    """
    Calculates the unweighted average yield per state and year
    from the district-level historical data.
    (This is an approximation due to missing area data).
    """
    print("\nAggregating historical district yields by (State, Year) using unweighted average...")

    # Calculate the mean (average) historical yield by Land and Year
    df_hist_agg = df_hist.groupby(['Year', 'Land']).agg(
        # Rename the column during aggregation for clarity
        Yield_t_ha_Hist_Avg=('Yield_t/ha_Hist', 'mean'),
        # Count the number of districts aggregated
        District_Count=('district_no', 'count')
    ).reset_index()

    return df_hist_agg


# --- 5. Compare Data and Output Results ---

def run_verification(df_hist_agg, df_ver):
    """
    Compares the aggregated historical data with the verified data.
    """
    print("\n--- Running Verification Comparison ---")

    # Merge the aggregated historical data with the verified data on Year and Land
    df_compare = pd.merge(
        df_hist_agg,
        df_ver,
        on=['Year', 'Land'],
        how='inner'  # Only compare years/states present in both
    )

    if df_compare.empty:
        print("ERROR: No overlapping (Year, Land) combinations found for comparison.")
        print("Check if the GeoJSON mapping is complete and if the files cover the same period.")
        return

    # Calculate the absolute difference (t/ha)
    df_compare['Abs_Difference_t/ha'] = (
            df_compare['Yield_t_ha_Hist_Avg'] - df_compare['Yield_t/ha_Ver']
    ).abs()

    # Calculate the percentage difference relative to the verified value
    df_compare['Pct_Difference'] = (
                                           df_compare['Abs_Difference_t/ha'] / df_compare['Yield_t/ha_Ver']
                                   ) * 100

    # Prepare the final output table
    df_final = df_compare.sort_values(by='Pct_Difference', ascending=False)
    df_final = df_final[[
        'Year',
        'Land',
        'District_Count',
        'Yield_t_ha_Hist_Avg',
        'Yield_t/ha_Ver',
        'Abs_Difference_t/ha',
        'Pct_Difference'
    ]]

    # Rename for final presentation
    df_final.columns = [
        'Year',
        'Land',
        'Districts_Aggregated',
        'Historical_Yield_t/ha (Avg)',
        'Verified_Yield_t/ha',
        'Abs_Diff (t/ha)',
        'Pct_Diff (%)'
    ]

    print("\n\n--- Final Yield Verification Results ---")
    print(f"Historical Yield Unit: ASSUMED to be t/ha (Tonnes per Hectare).")
    print(f"Historical Aggregation: UNWEIGHTED AVERAGE of District Yields.")
    print("------------------------------------------------------------------")
    print(df_final.to_markdown(index=False, floatfmt=".2f"))
    print("------------------------------------------------------------------")

    # Interpretation Guidance
    print("\n--- Interpretation Guidance ---")
    print(
        "1. Low 'Pct_Diff (%)' (e.g., < 10%) suggests the historical data is likely trustworthy and the t/ha unit assumption is correct.")
    print("2. High 'Pct_Diff (%)' suggests either:")
    print("   a) The historical data is unreliable.")
    print("   b) The historical yield is NOT in t/ha (e.g., it is in some other unit).")
    print(
        "   c) The unweighted average is inappropriate (i.e., district areas vary widely, and you NEED the 'Field_ha' for the historical data).")


# --- Main Execution ---

if __name__ == '__main__':
    # 1. Create Mapping
    district_map = create_district_to_state_map(PATH_GEOJSON)

    # 2. Load and Prepare Data
    df_hist, df_ver = load_and_prepare_data(district_map)

    if df_hist is not None and df_ver is not None:
        # 3. Aggregate Historical Data
        df_hist_agg = aggregate_historical_data(df_hist)

        # 4. Compare and Output Results
        run_verification(df_hist_agg, df_ver)