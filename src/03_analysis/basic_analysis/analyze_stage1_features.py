# File: src/03_analysis/basic_analysis/analyze_stage1_features.py
# Description: Detailed diagnostics of the Stage 1 Feature Set.
#              Includes specific validation for Hybrid Logic (Trend vs Physics).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

# --- Configuration ---
# We use the paths from the config to ensure consistency
INPUT_FILE = config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']['OUTPUT_FILE']
OUTPUT_DIR = Path('reports/stage2_feature_diagnostics/figures')
TABLES_DIR = Path('reports/stage2_feature_diagnostics/tables')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'kreisYield'


def main():
    print(f"--- Starting Analysis of: {INPUT_FILE} ---")

    # --- Load Dataset ---
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ File not found: {INPUT_FILE}")
        return

    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # ==============================================================================
    # === 1. HYBRID LOGIC VALIDATION (THE CRITICAL CHECKS) ===
    # ==============================================================================
    print("\n" + "=" * 50)
    print("=== 1. HYBRID LOGIC VALIDATION ===")
    print("=" * 50)

    # A. UNIT SANITY CHECK
    # Are WOFOST yields and Actual yields in the same universe?
    # They don't need to match perfectly (bias is expected), but they should be same order of magnitude.
    print("\n--- A. Unit Sanity Check (dt/ha) ---")
    act_mean = df[TARGET].mean()
    wof_mean = df['wofost_esp_mean'].mean()
    trend_mean = df['stat_trend_forecast'].mean()

    print(f"Actual Yield Mean : {act_mean:.2f}")
    print(f"Trend Model Mean  : {trend_mean:.2f}")
    print(f"WOFOST Sim Mean   : {wof_mean:.2f}")

    if abs(act_mean - wof_mean) > 300:
        print("🚨 WARNING: Massive unit mismatch (>300 dt/ha). Did conversion fail?")
    else:
        print("✅ Units look compatible (Fresh Weight dt/ha).")

    # B. THE GAP HYPOTHESIS
    # Does the 'Gap' (WOFOST - Trend) correlate with the 'Residual' (Actual - Trend)?
    # If yes, WOFOST is successfully correcting the Trend.
    print("\n--- B. The Gap Hypothesis ---")
    df['trend_residual'] = df[TARGET] - df['stat_trend_forecast']

    gap_corr = df[['trend_vs_phys_gap', 'trend_residual']].corr().iloc[0, 1]
    print(f"Correlation between (WOFOST - Trend) and (Actual - Trend): {gap_corr:.4f}")

    plt.figure(figsize=(8, 6))
    sns.regplot(x='trend_vs_phys_gap', y='trend_residual', data=df,
                scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'})
    plt.title(f"Hybrid Logic Validation: Gap vs Residual (Corr: {gap_corr:.2f})")
    plt.xlabel("Physical Signal (WOFOST - Trend)")
    plt.ylabel("Trend Error (Actual - Trend)")
    plt.grid(True, linestyle=':')
    plt.savefig(OUTPUT_DIR / 'hybrid_logic_gap_validation.png')
    plt.close()

    # C. RISK SENSITIVITY (2018 Check)
    # Did the ensemble spread (std) spike in 2018?
    print("\n--- C. Risk Sensitivity (Uncertainty over Time) ---")
    yearly_risk = df.groupby('year')['wofost_esp_std'].mean()

    plt.figure(figsize=(12, 5))
    yearly_risk.plot(marker='o', color='purple')
    plt.title("WOFOST Ensemble Uncertainty (Std Dev) Over Time")
    plt.ylabel("Ensemble Spread (dt/ha)")
    plt.grid(True)
    # Highlight 2018
    if 2018 in yearly_risk.index:
        plt.axvline(x=2018, color='red', linestyle='--', alpha=0.5, label='2018 (Drought)')
        plt.legend()
    plt.savefig(OUTPUT_DIR / 'hybrid_risk_time_series.png')
    plt.close()

    # D. MECHANISM CORRELATIONS
    # Do the new specific features correlate with the *Residual*?
    print("\n--- D. Mechanism Features vs Trend Residual ---")
    mech_cols = ['toxic_carryover_index', 'vector_pressure_local', 'nitrogen_leaching_index',
                 'winter_pest_kill_days', 'sowing_potential_days']

    valid_mech = [c for c in mech_cols if c in df.columns]
    if valid_mech:
        mech_corr = df[valid_mech + ['trend_residual']].corr()['trend_residual'].drop('trend_residual')
        print(mech_corr.sort_values(ascending=False))
    else:
        print("No mechanism features found.")

    # ==============================================================================
    # === 2. STANDARD DATA QUALITY CHECKS ===
    # ==============================================================================
    print("\n" + "=" * 50)
    print("=== 2. DATA QUALITY CHECKS ===")
    print("=" * 50)

    # Missing Values
    missing_vals = df.isnull().sum()
    missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=False)
    if not missing_vals.empty:
        print(f"Found {len(missing_vals)} columns with missing values:")
        print(missing_vals.head())
        missing_vals.to_csv(TABLES_DIR / 'missing_values_report.csv')
    else:
        print("✅ No missing values.")

    # Zero Values
    numeric_df = df.select_dtypes(include=[np.number])
    zero_vals = (numeric_df == 0).sum()
    zero_vals = zero_vals[zero_vals > 0].sort_values(ascending=False)
    if not zero_vals.empty:
        print(f"\nFound {len(zero_vals)} numeric columns with zeros (Top 5):")
        print(zero_vals.head(5))

    # Low Variance
    unique_counts = df.nunique()
    constant_cols = unique_counts[unique_counts == 1].index.tolist()
    if constant_cols:
        print(f"\n🚨 WARNING: Constant columns found: {constant_cols}")

    # ==============================================================================
    # === 3. CORRELATION & VIF ===
    # ==============================================================================
    print("\n" + "=" * 50)
    print("=== 3. CORRELATION & VIF ===")
    print("=" * 50)

    # Target Correlations
    corr = df.corr(numeric_only=True)
    if TARGET in corr.columns:
        target_corr = corr[TARGET].sort_values(ascending=False)
        print("\nTop 10 Positively Correlated:")
        print(target_corr.head(10))
        print("\nTop 10 Negatively Correlated:")
        print(target_corr.tail(10))
        target_corr.to_csv(TABLES_DIR / f'{TARGET}_correlations.csv')

        # Heatmap (Top 30)
        top_feats = target_corr.abs().sort_values(ascending=False).head(30).index
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[top_feats].corr(), cmap='RdBu_r', center=0, annot=False)
        plt.title("Correlation Matrix (Top 30 Features)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'correlation_matrix_top30.png')
        plt.close()

    # VIF (Top 20 Features only to save time)
    print("\nCalculating VIF for Top 20 Features...")
    if TARGET in corr.columns:
        top_20_feats = target_corr.abs().sort_values(ascending=False).head(21).index.drop(TARGET)
        X_vif = df[top_20_feats].dropna()

        try:
            vif_data = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
            vif_df = pd.DataFrame({'feature': X_vif.columns, 'VIF': vif_data})
            print(vif_df.sort_values('VIF', ascending=False))
            vif_df.to_csv(TABLES_DIR / 'vif_report.csv', index=False)
        except Exception as e:
            print(f"VIF Calculation failed: {e}")

    # ==============================================================================
    # === 4. VISUALIZATION ===
    # ==============================================================================
    print("\n" + "=" * 50)
    print("=== 4. VISUALIZATION ===")
    print("=" * 50)

    # Target Distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[TARGET], kde=True, bins=30)
    plt.title(f"{TARGET} Distribution")
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[TARGET])
    plt.title(f"{TARGET} Boxplot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_distribution.png')
    plt.close()

    # Yield by District (Top/Bottom 20)
    if 'district_no' in df.columns:
        dist_avg = df.groupby('district_no')[TARGET].mean().sort_values()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.barplot(x=dist_avg.tail(20).index, y=dist_avg.tail(20).values, palette='viridis')
        plt.xticks(rotation=90)
        plt.title("Top 20 Districts")

        plt.subplot(1, 2, 2)
        sns.barplot(x=dist_avg.head(20).index, y=dist_avg.head(20).values, palette='plasma')
        plt.xticks(rotation=90)
        plt.title("Bottom 20 Districts")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'district_yield_rankings.png')
        plt.close()

    print(f"\nAnalysis Complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()