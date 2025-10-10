import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Configuration ---
output_dir = 'reports/stage2_feature_diagnostics/figures'
os.makedirs(output_dir, exist_ok=True)

file_path = 'data/05_model_input/final_imputed_dataset.csv'
target = 'kreisYield'  # Explicitly define target

# --- Load Dataset ---
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    raise SystemExit(f"❌ File not found: {file_path}")

print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")

# --- Inspect Target ---
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset")

# --- Descriptive Stats ---
print("📊 Descriptive Statistics:")
print(df.describe().T)

# --- Target Distribution ---
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.histplot(df[target], kde=True, bins=30, color='skyblue')
plt.title(f"Distribution of {target}")
plt.subplot(1,2,2)
sns.boxplot(y=df[target], color='lightcoral')
plt.title(f"Boxplot of {target}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{target}_distribution.png'))
plt.close()

# --- Correlation Matrix ---
corr = df.corr(numeric_only=True)
plt.figure(figsize=(18,14))
sns.heatmap(corr, cmap='RdBu_r', center=0, annot=False)
plt.title("Correlation Matrix of All Variables")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

# --- Correlation Ranking ---
corr_target = corr[target].drop(target).sort_values(ascending=False)
top_pos = corr_target.head(15)
top_neg = corr_target.tail(15)
print("\n🔥 Top 15 Positive Correlations with Target:")
print(top_pos)
print("\n❄️ Top 15 Negative Correlations with Target:")
print(top_neg)

# --- Multicollinearity (VIF) ---
print("\n🔍 Variance Inflation Factor (VIF) Check:")
numeric_features = df.select_dtypes(include=[np.number]).drop(columns=[target])
vif_df = pd.DataFrame({
    'feature': numeric_features.columns,
    'VIF': [variance_inflation_factor(numeric_features.values, i)
            for i in range(numeric_features.shape[1])]
})
print(vif_df.sort_values('VIF', ascending=False).head(10))

# --- Feature Distributions + Skewness ---
skew_df = df[numeric_features.columns].skew().sort_values(ascending=False)
print("\n📈 Features with High Skewness (>|1|):")
print(skew_df[abs(skew_df) > 1])

# --- Scatter Plots: Top Correlated Features ---
top_predictors = list(top_pos.head(6).index)
fig, axes = plt.subplots(2, 3, figsize=(18,10))
axes = axes.flatten()
for i, feat in enumerate(top_predictors):
    sns.regplot(x=df[feat], y=df[target], ax=axes[i], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[i].set_title(f"{target} vs {feat}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_predictor_scatterplots.png'))
plt.close()

# --- Time Trend of Target ---
if 'year' in df.columns:
    yearly = df.groupby('year')[target].sum()
    plt.figure(figsize=(12,6))
    plt.plot(yearly.index, yearly.values, marker='o')
    plt.title(f"{target} Over Time")
    plt.xlabel("Year")
    plt.ylabel(target)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{target}_time_series.png'))
    plt.close()

# --- Yield by District ---
if 'district_no' in df.columns:
    district_summary = df.groupby('district_no')[target].sum().sort_values(ascending=False)
    plt.figure(figsize=(16,6))
    sns.barplot(x=district_summary.index, y=district_summary.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.title(f"Total {target} by District")
    plt.ylabel(f"Total {target}")
    plt.xlabel("District")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yield_by_district.png'))
    plt.close()

# --- Summary Export ---
summary_dir = os.path.join(output_dir, '../tables')
os.makedirs(summary_dir, exist_ok=True)
vif_df.to_csv(os.path.join(summary_dir, 'vif_report.csv'), index=False)
corr_target.to_csv(os.path.join(summary_dir, f'{target}_correlations.csv'))

print("\n✅ Analysis Complete. Outputs saved to:")
print(output_dir)
