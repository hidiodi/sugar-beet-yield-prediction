# inspect_final_dataset.py (ADVANCED VERSION)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Define the path to the ADVANCED dataset ---
# --- CHANGE 1: Point to the new file ---
filepath = Path("data/03_processed/final_dataset_with_advanced_features.csv")

# --- 1. Load the Data ---
try:
    df = pd.read_csv(filepath)
except FileNotFoundError:
    print(f"ERROR: The file was not found at '{filepath}'")
    exit()

print(f"--- Inspecting Advanced Dataset: {filepath.name} ---")
print(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
print("\n" + "="*50 + "\n")

# --- 2. Check Data Structure and Types ---
print("--- 1. Data Structure and Types ---")
print(df.info())
print("\n" + "="*50 + "\n")

# --- 3. Check for Missing Values ---
print("--- 2. Missing Values Check ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("SUCCESS: No missing values found in the dataset.")
else:
    print("WARNING: Missing values detected. Review the columns listed above.")
print("\n" + "="*50 + "\n")

# --- 4. Check Descriptive Statistics ---
print("--- 3. Descriptive Statistics for Numerical Columns ---")
# The data is already in Celsius, so no conversion is needed.
# Let's select the yield and all our new feature columns.
# --- CHANGE 2: Select the new feature columns for the summary ---
feature_cols = [col for col in df.columns if '_growth' in col or '_days' in col]
print(df[['yield'] + feature_cols].describe())
print("\n" + "="*50 + "\n")

# --- 5. Correlation Analysis and Heatmap ---
print("--- 4. Correlation Analysis (Advanced Features vs. Yield) ---")

# --- CHANGE 3: Select the new feature columns for the correlation ---
correlation_matrix = df[['yield'] + feature_cols].corr()

# Focus on the correlations with the 'yield' column
yield_correlations = correlation_matrix[['yield']].sort_values(by='yield', ascending=False)
print("Correlations of ADVANCED weather features with crop yield:")
print(yield_correlations)

# Create a heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(yield_correlations, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation of Advanced Weather Features with Sugar Beet Yield', fontsize=16)

# Save the plot to a file
plot_filename = "yield_advanced_features_correlation.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight') # Use bbox_inches='tight' for better layout
print(f"\nSUCCESS: ADVANCED correlation heatmap saved to '{plot_filename}'.")
print("This plot will show the much more insightful relationships.")