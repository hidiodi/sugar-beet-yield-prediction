import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Configuration ---
output_dir = 'reports/stage2_feature_diagnostics/figures'
tables_dir = 'reports/stage2_feature_diagnostics/tables'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

file_path = 'data/05_model_input/stage1_preseason_features.csv'
target = 'kreisYield'  # Explicitly define target

# --- Load Dataset ---
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    raise SystemExit(f"❌ File not found: {file_path}")

print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ==============================================================================
# === NEW: DATA QUALITY & ANOMALY CHECKS ===
# ==============================================================================

# --- 1. NEW: Missing Value (NaN) Check ---
print("\n--- 1. Missing Value (NaN) Check ---")
missing_vals = df.isnull().sum()
missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=False)
if missing_vals.empty:
    print("✅ No missing (NaN) values found.")
else:
    print(f"😱 Found {len(missing_vals)} columns with missing values:")
    print(missing_vals)
    print(f"Total missing cells: {missing_vals.sum()}")
    missing_vals.to_csv(os.path.join(tables_dir, 'missing_values_report.csv'))

# --- 2. NEW: Zero Value Check ---
print("\n--- 2. Zero Value Check ---")
# Exclude non-numeric columns from this check
numeric_df = df.select_dtypes(include=[np.number])
zero_vals = (numeric_df == 0).sum()
zero_vals = zero_vals[zero_vals > 0].sort_values(ascending=False)
if zero_vals.empty:
    print("✅ No zero values found in numeric columns.")
else:
    print(f"Found {len(zero_vals)} numeric columns containing zero values:")
    print(zero_vals.head(20))  # Show top 20

    # Pay special attention to features that should *not* be zero
    suspect_zeros = [
        'wofost_forecast_yield_fresh_dt', 'stat_trend_forecast',
        'lat', 'lon', 'avg_elevation'
    ]
    suspect_zero_counts = zero_vals[zero_vals.index.isin(suspect_zeros)]
    if not suspect_zero_counts.empty:
        print("\n🚨 WARNING: Zeros found in columns that are often non-zero:")
        print(suspect_zero_counts)

# --- 3. NEW: Low Variance / Constant Feature Check ---
print("\n--- 3. Low Variance / Constant Feature Check ---")
unique_counts = df.nunique()
constant_cols = unique_counts[unique_counts == 1].index.tolist()
if constant_cols:
    print(f"🚨 WARNING: Constant columns (zero variance) found: {constant_cols}")
    print("   These columns should be DROPPED before training.")
else:
    print("✅ No constant columns found.")

quasi_constant_cols = unique_counts[(unique_counts > 1) & (unique_counts < 5)].index.tolist()
if quasi_constant_cols:
    print(f"⚠️ INFO: Quasi-constant columns (2-4 unique values): {quasi_constant_cols}")

# ==============================================================================
# === ORIGINAL SCRIPT (STATISTICAL ANALYSIS) ===
# ==============================================================================

# --- Inspect Target ---
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset")
# Check for NaNs in target
target_nans = df[target].isnull().sum()
if target_nans > 0:
    print(f"\n🚨🚨🚨 CRITICAL: {target_nans} missing values found in target column '{target}'.")
    print("   Dropping rows with missing target for analysis.")
    df = df.dropna(subset=[target])
else:
    print(f"\n✅ Target column '{target}' found with no missing values.")

# --- Descriptive Stats ---
print("\n📊 Descriptive Statistics:")
desc_stats = df.describe().T
print(desc_stats)
desc_stats.to_csv(os.path.join(tables_dir, 'descriptive_statistics.csv'))

# --- Target Distribution ---
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(df[target], kde=True, bins=30, color='skyblue')
plt.title(f"Distribution of {target}")
plt.subplot(1, 2, 2)
sns.boxplot(y=df[target], color='lightcoral')
plt.title(f"Boxplot of {target}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{target}_distribution.png'))
plt.close()

# --- Correlation Matrix ---
# Use only a subset for the heatmap, as 124x124 is unreadable
print("\nGenerating Correlation Heatmap (for top 50 correlated features)...")
corr = df.corr(numeric_only=True)
corr_target_all = corr.get(target, pd.Series(dtype=float))
if corr_target_all.empty:
    print("Warning: Could not get target correlations. Skipping heatmap.")
else:
    top_50_features = corr_target_all.abs().sort_values(ascending=False).head(50).index
    corr_subset = df[top_50_features].corr(numeric_only=True)

    plt.figure(figsize=(20, 16))  # Increased size
    sns.heatmap(corr_subset, cmap='RdBu_r', center=0, annot=False)
    plt.title("Correlation Matrix of Top 50 Features (by corr. with target)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix_top50.png'))
    plt.close()

# --- Correlation Ranking ---
corr_target = corr_target_all.drop(target, errors='ignore').sort_values(ascending=False)
top_pos = corr_target.head(15)
top_neg = corr_target.tail(15)
print("\n🔥 Top 15 Positive Correlations with Target:")
print(top_pos)
print("\n❄️ Top 15 Negative Correlations with Target:")
print(top_neg)

# --- Multicollinearity (VIF) ---
print("\n🔍 Variance Inflation Factor (VIF) Check:")
# VIF is computationally expensive. Let's check a subset, e.g., our top 50.
# VIF on all 120 features will be slow and likely crash.
vif_features = df[top_50_features].drop(columns=[target], errors='ignore')
# Drop any rows with NaNs *just for VIF calculation*
vif_features_clean = vif_features.dropna()

print(f"Calculating VIF for {len(vif_features_clean.columns)} features...")
try:
    vif_data = [variance_inflation_factor(vif_features_clean.values, i) for i in range(vif_features_clean.shape[1])]
    vif_df = pd.DataFrame({'feature': vif_features_clean.columns, 'VIF': vif_data})
    print(vif_df.sort_values('VIF', ascending=False).head(20))  # Show top 20
except Exception as e:
    print(f"Could not calculate VIF: {e}")
    vif_df = pd.DataFrame()  # Create empty df for export

# --- 4. NEW: Outlier Check (Z-Score > 3) ---
print("\n--- 4. Outlier Check (Z-Score > 3) ---")
numeric_features = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
# This can be slow, let's just check the top 50 features
z_check_features = numeric_features.get(top_50_features.drop(target, errors='ignore'), numeric_features)

try:
    z_scores = np.abs(z_check_features.apply(lambda x: (x - x.mean()) / x.std()))
    outliers_count = (z_scores > 3).sum().sort_values(ascending=False)
    outliers_count = outliers_count[outliers_count > 0]

    if outliers_count.empty:
        print("✅ No significant outliers (Z-score > 3) found in top 50 features.")
    else:
        print(f"Found {len(outliers_count)} columns (in top 50) with potential outliers:")
        print(outliers_count.head(20))
except Exception as e:
    print(f"Could not calculate Z-scores: {e}")

# --- Feature Distributions + Skewness ---
skew_df = numeric_features.skew().sort_values(ascending=False)
print("\n📈 Features with High Skewness (>|2|):")
print(skew_df[abs(skew_df) > 2])  # Upped threshold to 2

# --- Scatter Plots: Top Correlated Features ---
top_predictors = list(top_pos.head(6).index)
if top_predictors:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, feat in enumerate(top_predictors):
        sns.regplot(x=df[feat], y=df[target], ax=axes[i], scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
        axes[i].set_title(f"{target} vs {feat}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_predictor_scatterplots.png'))
    plt.close()
else:
    print("Skipping scatter plots, no top predictors found.")

# --- Time Trend of Target ---
if 'year' in df.columns:
    yearly = df.groupby('year')[target].mean()  # Changed to mean
    plt.figure(figsize=(12, 6))
    plt.plot(yearly.index, yearly.values, marker='o')
    plt.title(f"Average {target} Over Time")
    plt.xlabel("Year")
    plt.ylabel(f"Average {target}")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{target}_time_series.png'))
    plt.close()

# --- Yield by District (Mean Yield) ---
if 'district_no' in df.columns:
    district_summary = df.groupby('district_no')[target].mean().sort_values(ascending=False)
    plt.figure(figsize=(16, 6))
    # Don't plot all 400+ districts, just top/bottom
    top_districts = district_summary.head(30)
    bottom_districts = district_summary.tail(30)

    plt.subplot(1, 2, 1)
    sns.barplot(x=top_districts.index, y=top_districts.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.title(f"Top 30 Districts by Avg. {target}")
    plt.ylabel(f"Average {target}")

    plt.subplot(1, 2, 2)
    sns.barplot(x=bottom_districts.index, y=bottom_districts.values, palette='plasma')
    plt.xticks(rotation=90)
    plt.title(f"Bottom 30 Districts by Avg. {target}")
    plt.ylabel(f"Average {target}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yield_by_district.png'))
    plt.close()

# --- Summary Export ---
vif_df.to_csv(os.path.join(tables_dir, 'vif_report.csv'), index=False)
corr_target.to_csv(os.path.join(tables_dir, f'{target}_correlations.csv'))
skew_df.to_csv(os.path.join(tables_dir, 'skewness_report.csv'))

print("\nAnalysis Complete. Outputs saved to:")
print(f"Figures: {output_dir}")
print(f"Tables:  {tables_dir}")