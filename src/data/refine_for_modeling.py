import pandas as pd
import os


def refine_dataset_for_modeling():
    """
    Loads the master dataset, removes unnecessary columns and rows with no target value,
    and saves the result as a model-ready input file.
    """
    # --- Define File Paths ---
    # Input
    master_file_path = 'data/04_master/master_dataset.csv'

    # Output
    output_path = 'data/05_model_input/'
    output_file = os.path.join(output_path, 'model_input.csv')
    os.makedirs(output_path, exist_ok=True)

    # --- Step 1: Load the Master Dataset ---
    try:
        df = pd.read_csv(master_file_path)
        print(f"Successfully loaded master dataset. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Master file not found at '{master_file_path}'.")
        print("Please run the 'create_master_dataset.py' script first.")
        return

    # --- Step 2: Remove Unnecessary Columns ---
    columns_to_drop = ['district', 'nuts_id']
    df_refined = df.drop(columns=columns_to_drop)
    print(f"Dropped columns: {columns_to_drop}. Shape is now: {df_refined.shape}")

    # --- Step 3: Handle Missing Target ('yield') Values ---
    # Check how many rows have a missing yield
    missing_yield_count = df_refined['yield'].isnull().sum()
    if missing_yield_count > 0:
        print(f"Found {missing_yield_count} rows with missing yield. These will be removed.")
        # Remove rows where 'yield' is NaN
        df_refined.dropna(subset=['yield'], inplace=True)
        print(f"Removed rows with missing yield. Shape is now: {df_refined.shape}")
    else:
        print("No rows with missing yield found. No rows removed.")

    # As a final check, let's see if any other columns have missing values
    remaining_nans = df_refined.isnull().sum()
    if remaining_nans.sum() > 0:
        print("\nWarning: Missing values still exist in the following columns:")
        print(remaining_nans[remaining_nans > 0])
        print("You may need to handle these before training (e.g., by filling with the mean).")
    else:
        print("\nNo other missing values found in the dataset.")

    # --- Step 4: Save the Final, Cleaned Dataset ---
    df_refined.to_csv(output_file, index=False)
    print(f"\nRefinement complete. Model-ready data saved to: {output_file}")
    print(f"Final dataset has {df_refined.shape[0]} rows and {df_refined.shape[1]} columns.")


if __name__ == '__main__':
    refine_dataset_for_modeling()