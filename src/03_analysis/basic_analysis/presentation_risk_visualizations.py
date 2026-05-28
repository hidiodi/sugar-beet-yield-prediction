import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / 'reports/figures/presentation_updates'
DATA_FILE = PROJECT_ROOT / 'reports/figures/final_model_comparison/super_ensemble_training_data.csv'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Unify plotting style
sns.set_context("talk")
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 20,
    'font.family': 'sans-serif'
})


def load_forecast_data():
    """Helper function to load the required forecast data robustly."""
    if not DATA_FILE.exists():
        print(f"Error: Forecast file not found at {DATA_FILE}")
        return None

    df = pd.read_csv(DATA_FILE)

    # Standardize column names to match the requested logic
    if 'Statistical_Trend_pred' in df.columns:
        df = df.rename(columns={'Statistical_Trend_pred': 'trend_pred'})
    if 'Hybrid_XGB_pred' in df.columns:
        df = df.rename(columns={'Hybrid_XGB_pred': 'xgb_pred'})

    # Ensure district format
    if 'district_no' in df.columns:
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)

    # Restrict to years 2000+ where predictions are valid
    if 'year' in df.columns:
        df = df[df['year'] >= 2000].copy()

    # Ensure required columns exist
    if not {'trend_pred', 'xgb_pred', 'kreisYield'}.issubset(df.columns):
        print("Missing required columns in forecast data.")
        return None

    return df


def plot_error_butterfly(df):
    """
    Graph 1: The "Error Butterfly" (Absolute Error by Yield Regime)
    """
    print("Generating Graph 1: Error Butterfly...")

    # Calculate absolute errors
    df['trend_ae'] = (df['kreisYield'] - df['trend_pred']).abs()
    df['xgb_ae'] = (df['kreisYield'] - df['xgb_pred']).abs()

    # Define actual yield bins. (Note: Data appears to be in dt/ha and ranges roughly 300-1000)
    # We will use bins: <500, 500-600, 600-700, 700-800, >800
    bins = [0, 500, 600, 700, 800, 2000]
    labels = ['<500', '500-600', '600-700', '700-800', '>800']

    df['yield_bin'] = pd.cut(df['kreisYield'], bins=bins, labels=labels)

    # Aggregate Mean Absolute Error (MAE) per bin
    summary = df.groupby('yield_bin')[['trend_ae', 'xgb_ae']].mean().reset_index()

    # Log Data Table
    print("\n[DATA TABLE] Graph 1: Error Butterfly (Absolute Error by Yield Regime)")
    print(summary.rename(columns={'yield_bin': 'Actual Yield Bin (dt/ha)', 'trend_ae': 'Baseline Trend MAE',
                                  'xgb_ae': 'Hybrid Model MAE'}).to_markdown(index=False, floatfmt=".2f"))
    print("\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(summary['yield_bin'], summary['trend_ae'], marker='o', linestyle='-', color='#95a5a6', linewidth=2.5,
             markersize=8, label='Statistical Trend (Baseline)')
    plt.plot(summary['yield_bin'], summary['xgb_ae'], marker='s', linestyle='-', color='#e74c3c', linewidth=2.5,
             markersize=8, label='Hybrid Risk Model (Ours)')

    plt.title('The "Error Butterfly": Model Performance by Yield Regime', pad=15)
    plt.xlabel('Actual Yield Outcome (dt/ha)', labelpad=10)
    plt.ylabel('Mean Absolute Error (dt/ha)', labelpad=10)
    plt.legend(title='Model Structure')
    plt.tight_layout()

    out_path = OUTPUT_DIR / 'Graph_1_Error_Butterfly.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_f1_score_ablation(df):
    """
    Graph 2: F1-Score Ablation Study (Justifying the 4% Threshold)
    """
    print("Generating Graph 2: F1-Score Ablation Study...")

    # Define a "True Regional Crash"
    # Actual yield is 10% or more below the trend
    df['true_crash'] = (df['kreisYield'] - df['trend_pred']) / df['trend_pred'] <= -0.10

    thresholds = np.linspace(0.01, 0.10, 10)
    f1_scores = []

    for t in thresholds:
        # Predicted Crash = Model predicts 100% downside when deviation is below threshold
        predicted_crash = (df['xgb_pred'] - df['trend_pred']) / df['trend_pred'] <= -t

        # Calculate F1 Score
        f1 = f1_score(df['true_crash'], predicted_crash, zero_division=0)
        f1_scores.append(f1)

    # Log Data Table
    ablation_df = pd.DataFrame({
        'Threshold (%)': np.round(thresholds * 100, 1),
        'F1-Score': np.round(f1_scores, 3)
    })
    print("\n[DATA TABLE] Graph 2: F1-Score Ablation Study for Crash Detection")
    print(ablation_df.to_markdown(index=False))
    print("\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds * 100, f1_scores, marker='o', linestyle='-', color='#8e44ad', linewidth=2.5, markersize=8)
    plt.axvline(x=4, color='#2c3e50', linestyle='--', label='Optimal 4% Detection Threshold')

    plt.title('Optimizing for Anomaly Detection: Threshold vs F1-Score', pad=15)
    plt.xlabel('Gating Threshold (%)', labelpad=10)
    plt.ylabel('F1-Score (Detecting True Crashes)', labelpad=10)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / 'Graph_2_F1_Ablation_Study.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_shock_localization(df, threshold=0.04):
    """
    Graph 3: Shock Localization Precision & Recall
    Compares the Trend Model vs. the Gated Hybrid Model's ability to classify regional crashes.
    """
    print("Generating Graph 3: Shock Localization Metrics...")

    # Define "True Regional Crash"
    df['true_crash'] = (df['kreisYield'] - df['trend_pred']) / df['trend_pred'] <= -0.10

    # Baseline Trend Prediction: Never predicts a crash relative to itself
    trend_predicted_crash = np.zeros(len(df), dtype=bool)

    # Hybrid Model Prediction (using 4% threshold)
    hybrid_predicted_crash = (df['xgb_pred'] - df['trend_pred']) / df['trend_pred'] <= -threshold

    # Calculate Metrics
    metrics = {
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Statistical Trend (Baseline)': [
            precision_score(df['true_crash'], trend_predicted_crash, zero_division=0),
            recall_score(df['true_crash'], trend_predicted_crash, zero_division=0),
            f1_score(df['true_crash'], trend_predicted_crash, zero_division=0)
        ],
        'Gated Hybrid Model (Ours)': [
            precision_score(df['true_crash'], hybrid_predicted_crash, zero_division=0),
            recall_score(df['true_crash'], hybrid_predicted_crash, zero_division=0),
            f1_score(df['true_crash'], hybrid_predicted_crash, zero_division=0)
        ]
    }

    metrics_df = pd.DataFrame(metrics)

    # Log Data Table
    print("\n[DATA TABLE] Graph 3: Shock Localization Performance")
    print(metrics_df.to_markdown(index=False, floatfmt=".3f"))
    print("\n")

    # Melt for plotting
    melted = pd.melt(metrics_df, id_vars='Metric',
                     value_vars=['Statistical Trend (Baseline)', 'Gated Hybrid Model (Ours)'],
                     var_name='Model', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette=['#bdc3c7', '#27ae60'])

    plt.title('Shock Localization: Anomaly Detection Performance', pad=15)
    plt.xlabel('Evaluation Metric', labelpad=10)
    plt.ylabel('Score (0.0 to 1.0)', labelpad=10)
    plt.ylim(0, 1.05)
    plt.legend(title='System Framework', loc='upper right')
    plt.tight_layout()

    out_path = OUTPUT_DIR / 'Graph_3_Shock_Localization.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def log_yearly_recall_deepdive(df, threshold=0.04):
    """
    Analyzes the classification performance on a year-by-year basis,
    specifically contrasting the 2018 success with the 2022 failure.
    """
    print("\n--- DEEP DIVE: Yearly Recall Analysis (2018 vs 2022) ---")

    # Define "True Regional Crash" globally
    df['true_crash'] = (df['kreisYield'] - df['trend_pred']) / df['trend_pred'] <= -0.10
    df['hybrid_predicted_crash'] = (df['xgb_pred'] - df['trend_pred']) / df['trend_pred'] <= -threshold

    results = []

    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year]

        actual_crashes = df_year['true_crash'].sum()
        if actual_crashes == 0:
            continue  # Skip years without severe regional shocks

        caught_crashes = (df_year['true_crash'] & df_year['hybrid_predicted_crash']).sum()
        false_alarms = (~df_year['true_crash'] & df_year['hybrid_predicted_crash']).sum()

        recall = caught_crashes / actual_crashes
        precision = caught_crashes / (caught_crashes + false_alarms) if (caught_crashes + false_alarms) > 0 else 0

        results.append({
            'Year': year,
            'Total Districts Crashed': actual_crashes,
            'Model Caught': caught_crashes,
            'Recall (%)': recall * 100,
            'Precision (%)': precision * 100
        })

    results_df = pd.DataFrame(results)

    print("\n[DATA TABLE] Yearly Shock Localization")
    print(results_df.to_markdown(index=False, floatfmt=".1f"))
    print("\n")

    # Specific logging for 2018 and 2022
    if 2018 in results_df['Year'].values and 2022 in results_df['Year'].values:
        r2018 = results_df[results_df['Year'] == 2018].iloc[0]
        r2022 = results_df[results_df['Year'] == 2022].iloc[0]

        print(
            f"** 2018 Drought (Success): Out of {r2018['Total Districts Crashed']} severe district crashes, the model caught {r2018['Model Caught']} (Recall: {r2018['Recall (%)']:.1f}%).")
        print(
            f"** 2022 Drought (Failure): Out of {r2022['Total Districts Crashed']} severe district crashes, the model caught {r2022['Model Caught']} (Recall: {r2022['Recall (%)']:.1f}%).")


def analyze_feature_discrepancy_2018_vs_2022(forecast_df):
    """
    Investigates why 2018/2003 had high recall and 2022 had low recall
    by comparing the pre-season feature signals (Z-scores) for crashed districts.
    """
    print("\n--- DEEP DIVE: Why did 2022 fail? (Feature Signal Analysis) ---")

    features_file = PROJECT_ROOT / 'data/05_model_input/stage1_preseason_features.csv'
    if not features_file.exists():
        print(f"Error: Features file not found at {features_file}")
        return

    features_df = pd.read_csv(features_file)
    if 'district_no' in features_df.columns:
        features_df['district_no'] = features_df['district_no'].astype(str).str.zfill(5)

    # We only care about districts that actually crashed
    forecast_df['true_crash'] = (forecast_df['kreisYield'] - forecast_df['trend_pred']) / forecast_df[
        'trend_pred'] <= -0.10

    # Merge forecast actuals with the features the model saw that year
    merged = pd.merge(forecast_df[['year', 'district_no', 'true_crash']],
                      features_df, on=['year', 'district_no'], how='inner')

    crashes = merged[merged['true_crash'] == True].copy()

    if crashes.empty:
        print("No crash data available after merge.")
        return

    # Get numeric feature columns only
    exclude = ['year', 'kreisYield', 'trend_pred', 'xgb_pred']
    numeric_cols = crashes.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    # Calculate global mean and standard deviation across ALL years/districts to establish a baseline
    global_mean = merged[feature_cols].mean()
    global_std = merged[feature_cols].std()

    # Calculate the average Z-score for the crashed districts in our target years
    z_scores_list = []
    for target_year in [2003, 2018, 2022]:
        year_crashes = crashes[crashes['year'] == target_year]
        if year_crashes.empty:
            continue

        # Mean raw feature values for the crashed districts in this specific year
        year_mean = year_crashes[feature_cols].mean()

        # Convert to Z-score relative to global baseline
        year_z = (year_mean - global_mean) / (global_std + 1e-9)
        year_z.name = f'Z-Score_{target_year}'
        z_scores_list.append(year_z)

    if len(z_scores_list) < 3:
        print("Missing data for one of the target years (2003, 2018, 2022) in the features file.")
        return

    z_df = pd.concat(z_scores_list, axis=1)

    # Calculate the discrepancy: How different was the signal in 2022 compared to 2018?
    z_df['Abs_Difference_18_22'] = (z_df['Z-Score_2018'] - z_df['Z-Score_2022']).abs()

    # Sort by the biggest discrepancy to find the "missing signals"
    z_df_sorted = z_df.sort_values(by='Abs_Difference_18_22', ascending=False).head(15)

    print("\n[DATA TABLE] Top 15 Missing Pre-Season Signals (2018 vs 2022)")
    print("Values represent Z-scores (How many Standard Deviations away from normal the features were).")
    print(z_df_sorted.to_markdown(floatfmt=".2f"))
    print("\n")


def evaluate_feature_ablation_all_years(forecast_df):
    """
    Performs LOYO (Leave-One-Year-Out) cross-validation across all years (2000+)
    testing 6 different feature subsets to see the true impact of each data stream.
    Logs global metrics and pivots specific interesting years.
    """
    print("\n--- DEEP DIVE: Multi-Year Feature Ablation (LOYO CV) ---")
    print("Training 150 XGBoost models... This may take a minute or two.\n")

    import xgboost as xgb

    features_file = PROJECT_ROOT / 'data/05_model_input/stage1_preseason_features.csv'
    if not features_file.exists():
        print(f"Error: Features file not found at {features_file}")
        return

    features_df = pd.read_csv(features_file)
    if 'district_no' in features_df.columns:
        features_df['district_no'] = features_df['district_no'].astype(str).str.zfill(5)

    # SAFETY FIX: Drop overlapping columns to avoid merge conflicts
    overlap_cols = [c for c in ['kreisYield', 'trend_pred', 'xgb_pred', 'state_name'] if c in features_df.columns]
    features_df = features_df.drop(columns=overlap_cols)

    # Merge features with actual yields and trend predictions
    merged = pd.merge(forecast_df[['year', 'district_no', 'kreisYield', 'trend_pred']],
                      features_df, on=['year', 'district_no'], how='inner')

    target = 'kreisYield'
    exclude = ['year', 'district_no', 'kreisYield', 'trend_pred', 'xgb_pred', 'state_name']
    all_features = [c for c in merged.columns if c not in exclude and np.issubdtype(merged[c].dtype, np.number)]

    # Define Subsets
    econ_terms = ['dngemittel', 'energie', 'schmierstoffe', 'zuckerrben', 'pflanzenschutz', 'saat_und_pflanzgut',
                  'pachtentgelte']
    wofost_terms = ['wofost', 'twso', 'lai', 'tagp', 'index_']
    eo_terms = ['ndvi', 'evi', 'b0', 'sat', 'capture']

    feature_sets = {
        "1. All Features (Baseline)": all_features,
        "2. No Economic Noise": [f for f in all_features if not any(x in f.lower() for x in econ_terms)],
        "3. No Seasonal Forecasts": [f for f in all_features if 'forecast' not in f.lower()],
        "4. No WOFOST (Biophysical)": [f for f in all_features if not any(x in f.lower() for x in wofost_terms)],
        "5. No Earth Observation (Sat)": [f for f in all_features if not any(x in f.lower() for x in eo_terms)],
        "6. Pure Weather & Soil": [f for f in all_features if 'forecast' not in f.lower() and not any(
            x in f.lower() for x in econ_terms + wofost_terms + eo_terms)]
    }

    years = sorted(merged['year'].unique())

    # Master dataframe to store all predictions for all years
    all_preds_df = merged[['year', 'district_no', 'kreisYield', 'trend_pred']].copy()
    all_preds_df['true_crash'] = (all_preds_df['kreisYield'] - all_preds_df['trend_pred']) / all_preds_df[
        'trend_pred'] <= -0.10

    for set_name in feature_sets.keys():
        all_preds_df[f'pred_{set_name}'] = np.nan

    for year in years:
        train_df = merged[merged['year'] != year]
        test_df = merged[merged['year'] == year]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        for set_name, features in feature_sets.items():
            if not features:
                continue

            model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
            model.fit(train_df[features], train_df[target])

            # Predict and store in the master dataframe
            preds = model.predict(test_df[features])
            all_preds_df.loc[all_preds_df['year'] == year, f'pred_{set_name}'] = preds

    # --- Metrics Calculation ---
    results_global = []

    # Track specific years in pivoted format
    target_years = [2003, 2014, 2018, 2021, 2022, 2024]
    recall_pivot = []
    rmse_pivot = []

    for set_name, features in feature_sets.items():
        pred_col = f'pred_{set_name}'

        # Helper to calculate metrics for a given subset of data
        def calc_metrics(df_subset):
            # Apply 4% Gating Logic
            pred_crash = (df_subset[pred_col] - df_subset['trend_pred']) / df_subset['trend_pred'] <= -0.04
            caught = (df_subset['true_crash'] & pred_crash).sum()
            actual_crashes = df_subset['true_crash'].sum()
            false_alarms = (~df_subset['true_crash'] & pred_crash).sum()

            recall = caught / actual_crashes if actual_crashes > 0 else 0
            precision = caught / (caught + false_alarms) if (caught + false_alarms) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            rmse = np.sqrt(np.mean((df_subset[pred_col] - df_subset['kreisYield']) ** 2))
            return rmse, recall, precision, f1

        # 1. Global Metrics (All Years)
        g_rmse, g_rec, g_prec, g_f1 = calc_metrics(all_preds_df.dropna(subset=[pred_col]))
        results_global.append({
            'Feature Set': set_name,
            'Features Count': len(features),
            'Global RMSE': g_rmse,
            'Global Recall (%)': g_rec * 100,
            'Global F1-Score': g_f1
        })

        # 2. Specific Years Pivot
        row_recall = {'Feature Set': set_name}
        row_rmse = {'Feature Set': set_name}

        for y in target_years:
            if y in years:
                df_y = all_preds_df[all_preds_df['year'] == y].dropna(subset=[pred_col])
                if not df_y.empty:
                    y_rmse, y_rec, y_prec, y_f1 = calc_metrics(df_y)
                    row_recall[str(y)] = y_rec * 100
                    row_rmse[str(y)] = y_rmse
                else:
                    row_recall[str(y)] = np.nan
                    row_rmse[str(y)] = np.nan
            else:
                row_recall[str(y)] = np.nan
                row_rmse[str(y)] = np.nan

        recall_pivot.append(row_recall)
        rmse_pivot.append(row_rmse)

    print("\n[DATA TABLE] Global Feature Ablation (Years 2000-2024)")
    print(pd.DataFrame(results_global).to_markdown(index=False, floatfmt=".2f"))

    print("\n[DATA TABLE] TARGET YEARS: Crash Recall (%) by Feature Set")
    print(pd.DataFrame(recall_pivot).to_markdown(index=False, floatfmt=".1f"))

    print("\n[DATA TABLE] TARGET YEARS: RMSE by Feature Set")
    print(pd.DataFrame(rmse_pivot).to_markdown(index=False, floatfmt=".1f"))
    print("\n")

def analyze_forecast_quality_impact(forecast_df):
    """
    Evaluates whether the ML model's success (Recall) is strictly tied
    to the quality/severity of the Meteorological Seasonal Forecast.
    """
    print("\n--- DEEP DIVE: Seasonal Forecast Quality vs. Model Recall ---")

    features_file = PROJECT_ROOT / 'data/05_model_input/stage1_preseason_features.csv'
    if not features_file.exists():
        print(f"Error: Features file not found at {features_file}")
        return

    features_df = pd.read_csv(features_file)
    if 'district_no' in features_df.columns:
        features_df['district_no'] = features_df['district_no'].astype(str).str.zfill(5)

    # Drop overlapping columns before merge
    overlap_cols = [c for c in ['kreisYield', 'trend_pred', 'xgb_pred', 'state_name'] if c in features_df.columns]
    features_df = features_df.drop(columns=overlap_cols)

    # Merge
    merged = pd.merge(forecast_df[['year', 'district_no', 'kreisYield', 'trend_pred', 'xgb_pred']],
                      features_df, on=['year', 'district_no'], how='inner')

    # Calculate Actual Shock (%) and Predicted Crash (using our 4% threshold)
    merged['Actual_Shock_Pct'] = (merged['kreisYield'] - merged['trend_pred']) / merged['trend_pred']
    merged['true_crash'] = merged['Actual_Shock_Pct'] <= -0.10
    merged['pred_crash'] = (merged['xgb_pred'] - merged['trend_pred']) / merged['trend_pred'] <= -0.04

    # Target specific extreme years
    target_years = [2003, 2015, 2018, 2019, 2022]

    results = []

    for year in target_years:
        df_year = merged[merged['year'] == year]
        if df_year.empty:
            continue

        # 1. How bad was the ACTUAL year? (Average yield drop across all districts)
        avg_actual_drop = df_year['Actual_Shock_Pct'].mean() * 100

        # 2. How did the ML model perform? (Recall)
        actual_crashes = df_year['true_crash'].sum()
        caught = (df_year['true_crash'] & df_year['pred_crash']).sum()
        recall = (caught / actual_crashes * 100) if actual_crashes > 0 else np.nan

        # 3. What did the METEOROLOGICAL SEASONAL FORECAST say in March?
        # We look at the average predicted summer temperature anomaly and spring precipitation
        # (Assuming these feature names exist based on previous logs)
        forecast_temp = df_year[
            'summer_temp_anomaly_forecast'].mean() if 'summer_temp_anomaly_forecast' in df_year.columns else np.nan
        forecast_precip = df_year[
            'spring_precip_anomaly_forecast'].mean() if 'spring_precip_anomaly_forecast' in df_year.columns else np.nan

        results.append({
            'Year': year,
            'Actual Yield Drop (%)': avg_actual_drop,
            'Forecasted Summer Heat Anomaly': forecast_temp,
            'Forecasted Spring Rain Anomaly': forecast_precip,
            'Model Crash Recall (%)': recall
        })

    results_df = pd.DataFrame(results)

    print("\n[DATA TABLE] Meteorological Forecast Quality vs ML Model Success")
    print("Does the ML model only succeed when the meteorological forecast correctly predicts extreme weather?")
    print(results_df.to_markdown(index=False, floatfmt=".2f"))
    print("\n")

def analyze_comprehensive_physical_drivers(forecast_df):
    """
    Extracts the full spectrum of pre-season signals (Antecedent, Climate, Forecast)
    to show exactly what physical drivers the model saw on March 1st for extreme years.
    """
    print("\n--- DEEP DIVE: Comprehensive March 1st Physical Drivers ---")

    features_file = PROJECT_ROOT / 'data/05_model_input/stage1_preseason_features.csv'
    if not features_file.exists():
        print(f"Error: Features file not found at {features_file}")
        return

    features_df = pd.read_csv(features_file)
    if 'district_no' in features_df.columns:
        features_df['district_no'] = features_df['district_no'].astype(str).str.zfill(5)

    overlap_cols = [c for c in ['kreisYield', 'trend_pred', 'xgb_pred', 'state_name'] if c in features_df.columns]
    features_df = features_df.drop(columns=overlap_cols)

    merged = pd.merge(forecast_df[['year', 'district_no', 'kreisYield', 'trend_pred', 'xgb_pred']],
                      features_df, on=['year', 'district_no'], how='inner')

    merged['Actual_Shock_Pct'] = (merged['kreisYield'] - merged['trend_pred']) / merged['trend_pred']
    merged['true_crash'] = merged['Actual_Shock_Pct'] <= -0.10
    merged['pred_crash'] = (merged['xgb_pred'] - merged['trend_pred']) / merged['trend_pred'] <= -0.04

    target_years = [2003, 2015, 2018, 2019, 2022]
    results = []

    for year in target_years:
        df_year = merged[merged['year'] == year]
        if df_year.empty:
            continue

        actual_crashes = df_year['true_crash'].sum()
        caught = (df_year['true_crash'] & df_year['pred_crash']).sum()
        recall = (caught / actual_crashes * 100) if actual_crashes > 0 else np.nan

        # Helper to safely extract column means
        def get_mean(col_name):
            return df_year[col_name].mean() if col_name in df_year.columns else np.nan

        results.append({
            'Year': year,
            'Recall (%)': recall,

            # 1. ANTECEDENT (ACTUAL WINTER WEATHER from AgERA5)
            'Winter GDD Anom': get_mean('antecedent_gdd_sum_anomaly'),
            'Winter Frost Days Anom': get_mean('antecedent_frost_days_anomaly'),

            # 2. CLIMATE INDICES (Macro Teleconnections)
            'NAO Winter Avg': get_mean('nao_winter_avg'),
            'ENSO Winter Avg': get_mean('enso_mei_winter_avg'),

            # 3. SEAS51 METEOROLOGICAL FORECAST
            'Spring Precip Forecast': get_mean('spring_precip_anomaly_forecast'),
            'Summer Heat Forecast': get_mean('summer_temp_anomaly_forecast'),
            'Summer Solar Forecast': get_mean('summer_solar_rad_anomaly_forecast')
        })

    results_df = pd.DataFrame(results)

    print("\n[DATA TABLE] The March 1st Physical Dashboard (What the ML saw pre-season)")
    print(results_df.to_markdown(index=False, floatfmt=".2f"))
    print("\n")


def test_2022_extrapolation_ablation(forecast_df):
    """
    Tests if the 2022 failure was caused by XGBoost's inability to extrapolate
    out-of-distribution (OOD) extremes, by comparing it to a Linear Model.
    """
    print("\n--- DEEP DIVE: Algorithmic Extrapolation (XGBoost vs Linear) for 2022 ---")

    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    features_file = PROJECT_ROOT / 'data/05_model_input/stage1_preseason_features.csv'
    features_df = pd.read_csv(features_file)
    if 'district_no' in features_df.columns:
        features_df['district_no'] = features_df['district_no'].astype(str).str.zfill(5)

    overlap_cols = [c for c in ['kreisYield', 'trend_pred', 'xgb_pred', 'state_name'] if c in features_df.columns]
    features_df = features_df.drop(columns=overlap_cols)

    merged = pd.merge(forecast_df[['year', 'district_no', 'kreisYield', 'trend_pred']],
                      features_df, on=['year', 'district_no'], how='inner')

    # Filter to pure weather/forecasts to eliminate economic noise
    target = 'kreisYield'
    exclude = ['year', 'district_no', 'kreisYield', 'trend_pred', 'state_name']
    all_numeric = [c for c in merged.columns if c not in exclude and np.issubdtype(merged[c].dtype, np.number)]
    pure_weather_features = [f for f in all_numeric if
                             'forecast' in f.lower() or 'antecedent' in f.lower() or 'nao' in f.lower() or 'enso' in f.lower() or 'sca' in f.lower()]

    train_df = merged[merged['year'] < 2022].copy()
    test_df = merged[merged['year'] == 2022].copy()

    if test_df.empty:
        return

    test_df['true_crash'] = (test_df['kreisYield'] - test_df['trend_pred']) / test_df['trend_pred'] <= -0.10
    actual_2022_crashes = test_df['true_crash'].sum()

    # 1. Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_model.fit(train_df[pure_weather_features], train_df[target])
    test_df['pred_xgb'] = xgb_model.predict(test_df[pure_weather_features])

    # 2. Train Linear Model (Ridge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[pure_weather_features])
    X_test_scaled = scaler.transform(test_df[pure_weather_features])

    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_train_scaled, train_df[target])
    test_df['pred_linear'] = ridge_model.predict(X_test_scaled)

    results = []

    for model_name, col in [('XGBoost (Tree-Based)', 'pred_xgb'), ('Ridge (Linear)', 'pred_linear')]:
        pred_crash = (test_df[col] - test_df['trend_pred']) / test_df['trend_pred'] <= -0.04
        caught = (test_df['true_crash'] & pred_crash).sum()
        recall = (caught / actual_2022_crashes * 100) if actual_2022_crashes > 0 else 0

        # Calculate how "extreme" the predictions were allowed to go
        max_downside_prediction = ((test_df[col] - test_df['trend_pred']) / test_df['trend_pred']).min() * 100

        results.append({
            'Algorithm': model_name,
            '2022 Crash Recall (%)': recall,
            'Max Predicted Yield Drop (%)': max_downside_prediction
        })

    print(pd.DataFrame(results).to_markdown(index=False, floatfmt=".1f"))
    print("\n")


def evaluate_architectural_fixes_walkforward(forecast_df):
    """
    100% LEAK-FREE WALK-FORWARD VALIDATION.
    Evaluates architectural fixes across the 2014-2024 test window,
    ensuring no future data is ever used to predict a past year.
    """
    print("\n--- DEEP DIVE: Strict Walk-Forward Evaluation (2014-2024) ---")
    print("Training models sequentially without future data leaks...\n")

    import xgboost as xgb

    features_file = PROJECT_ROOT / 'data/05_model_input/stage1_preseason_features.csv'
    features_df = pd.read_csv(features_file)
    if 'district_no' in features_df.columns:
        features_df['district_no'] = features_df['district_no'].astype(str).str.zfill(5)

    overlap_cols = [c for c in ['kreisYield', 'trend_pred', 'xgb_pred', 'state_name'] if c in features_df.columns]
    features_df = features_df.drop(columns=overlap_cols)

    merged = pd.merge(forecast_df[['year', 'district_no', 'kreisYield', 'trend_pred']],
                      features_df, on=['year', 'district_no'], how='inner')

    target = 'kreisYield'
    exclude = ['year', 'district_no', 'kreisYield', 'trend_pred', 'state_name']
    all_numeric = [c for c in merged.columns if c not in exclude and np.issubdtype(merged[c].dtype, np.number)]

    # Define Strict Features (Fix A)
    econ_terms = ['dngemittel', 'energie', 'schmierstoffe', 'zuckerrben', 'pflanzenschutz', 'saat_und_pflanzgut',
                  'pachtentgelte']
    strict_features = [f for f in all_numeric if (
                'summer' in f.lower() or 'antecedent' in f.lower() or 'wofost' in f.lower() or 'twso' in f.lower()) and not any(
        x in f.lower() for x in econ_terms)]

    # We only evaluate on the operational test window (2014-2024)
    test_years = [y for y in sorted(merged['year'].unique()) if y >= 2014]

    all_preds_df = merged[merged['year'] >= 2014][['year', 'district_no', 'kreisYield', 'trend_pred']].copy()
    all_preds_df['true_crash'] = (all_preds_df['kreisYield'] - all_preds_df['trend_pred']) / all_preds_df[
        'trend_pred'] <= -0.10

    for model_name in ['Base', 'Fix_A', 'Fix_B', 'Fix_C']:
        all_preds_df[f'pred_{model_name}'] = np.nan

    for year in test_years:
        # STRICT WALK-FORWARD: Only train on data strictly BEFORE the target year
        train_df = merged[merged['year'] < year].copy()
        test_df = merged[merged['year'] == year].copy()

        if train_df.empty or test_df.empty: continue

        # --- BASELINE ---
        model_base = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
        model_base.fit(train_df[all_numeric], train_df[target])
        pred_base = model_base.predict(test_df[all_numeric])
        all_preds_df.loc[all_preds_df['year'] == year, 'pred_Base'] = pred_base

        # --- FIX A: Strict Features ---
        model_strict = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
        model_strict.fit(train_df[strict_features], train_df[target])
        pred_strict = model_strict.predict(test_df[strict_features])
        all_preds_df.loc[all_preds_df['year'] == year, 'pred_Fix_A'] = pred_strict

        # --- FIX B: Crisis-Weighted ---
        train_df['true_crash_train'] = (train_df['kreisYield'] - train_df['trend_pred']) / train_df[
            'trend_pred'] <= -0.10
        weights = np.where(train_df['true_crash_train'], 5.0, 1.0)
        model_weighted = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
        model_weighted.fit(train_df[all_numeric], train_df[target], sample_weight=weights)
        pred_weighted = model_weighted.predict(test_df[all_numeric])
        all_preds_df.loc[all_preds_df['year'] == year, 'pred_Fix_B'] = pred_weighted

        # --- FIX C: Physics Override ---
        ml_crash = (pred_base - test_df['trend_pred']) / test_df['trend_pred'] <= -0.04
        heat_col = 'summer_temp_anomaly_forecast'
        if heat_col in test_df.columns:
            physics_trigger = (test_df[heat_col] > 1.0).values
        else:
            physics_trigger = np.zeros(len(test_df), dtype=bool)

        pred_override = np.where(physics_trigger & (~ml_crash), test_df['trend_pred'] * 0.90, pred_base)
        all_preds_df.loc[all_preds_df['year'] == year, 'pred_Fix_C'] = pred_override

    # --- Metrics Calculation ---
    results_global = []

    for model_name, display_name in [
        ('Base', '1. Baseline (All Features)'),
        ('Fix_A', '2. Fix A (Strict Biophyiscal Core)'),
        ('Fix_B', '3. Fix B (5x Crisis-Weighted Loss)'),
        ('Fix_C', '4. Fix C (Physics Override Gate)')
    ]:
        pred_col = f'pred_{model_name}'
        df_valid = all_preds_df.dropna(subset=[pred_col])

        pred_crash = (df_valid[pred_col] - df_valid['trend_pred']) / df_valid['trend_pred'] <= -0.04
        caught = (df_valid['true_crash'] & pred_crash).sum()
        actual_crashes = df_valid['true_crash'].sum()
        false_alarms = (~df_valid['true_crash'] & pred_crash).sum()

        recall = (caught / actual_crashes * 100) if actual_crashes > 0 else 0
        precision = (caught / (caught + false_alarms) * 100) if (caught + false_alarms) > 0 else 0
        rmse = np.sqrt(np.mean((df_valid[pred_col] - df_valid['kreisYield']) ** 2))

        results_global.append({
            'Architecture': display_name,
            '2014-2024 RMSE': rmse,
            '2014-2024 Recall (%)': recall,
            '2014-2024 Precision (%)': precision
        })

    print("\n[DATA TABLE] Leak-Free Operational Performance Trade-offs (Test Window: 2014-2024)")
    print(pd.DataFrame(results_global).to_markdown(index=False, floatfmt=".2f"))

    # 2022 Specific Check
    df_2022 = all_preds_df[all_preds_df['year'] == 2022]
    print(
        f"\n2022 Recall for Fix C (Physics Override): {((df_2022['true_crash'] & ((df_2022['pred_Fix_C'] - df_2022['trend_pred']) / df_2022['trend_pred'] <= -0.04)).sum() / df_2022['true_crash'].sum()) * 100:.1f}%")
    print("\n")


def plot_national_yield_variance(df):
    """Graph 0: National Yield Volatility for the Motivation Slide"""
    print("Generating Graph 0: National Yield Variance...")
    nat_yield = df.groupby('year')[['kreisYield', 'trend_pred']].mean().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(nat_yield['year'], nat_yield['kreisYield'], marker='o', color='#1a1a2e', label='Actual Yield', linewidth=2)
    plt.plot(nat_yield['year'], nat_yield['trend_pred'], linestyle='--', color='#718096', label='Statistical Trend',
             linewidth=2)

    # Highlight crashes
    for y in [2003, 2018, 2022]:
        if y in nat_yield['year'].values:
            val = nat_yield[nat_yield['year'] == y]['kreisYield'].values[0]
            plt.scatter([y], [val], color='#c0392b', s=100, zorder=5)
            plt.annotate(f"{y}", (y, val - 15), ha='center', color='#c0392b', fontweight='bold', fontsize=10)

    plt.title('National Average Sugarbeet Yield (Germany)', pad=15)
    plt.ylabel('Yield (dt/ha)', labelpad=10)
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'Graph_0_National_Yield.png'
    plt.savefig(out_path, dpi=300)
    plt.close()


def evaluate_gate_ablation():
    """
    Ablation study to test different gate weighting strategies vs the ML Meta-Learner.
    Loads the walkforward predictions file directly to ensure all models are present.
    """
    print("\n" + "=" * 80)
    print("      GATE ABLATION: Deterministic Weights vs ML Meta-Learner")
    print("=" * 80)

    file_path = PROJECT_ROOT / 'reports/figures/final_model_comparison/super_ensemble_walkforward_predictions.csv'
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Check actual column names in the CSV
    trend_col = 'Statistical_Trend_pred' if 'Statistical_Trend_pred' in df.columns else 'Statistical Trend_pred'
    xgb_col = 'Hybrid_XGB_pred' if 'Hybrid_XGB_pred' in df.columns else 'Standalone XGBoost_pred'
    meta_col = 'Super_Ensemble_pred' if 'Super_Ensemble_pred' in df.columns else 'Super Ensemble_pred'

    # Define gate logic
    def apply_gate(row, down_weight, up_weight, threshold=0.04):
        trend = row[trend_col]
        xgb = row[xgb_col]

        if pd.isna(trend) or pd.isna(xgb) or trend == 0:
            return np.nan

        dev = (xgb - trend) / trend
        if dev < -threshold:
            return trend + ((xgb - trend) * down_weight)
        elif dev > threshold:
            return trend + ((xgb - trend) * up_weight)
        else:
            return trend

    # Create the ablation scenarios
    df['Gate_100_Down_50_Up'] = df.apply(lambda r: apply_gate(r, 1.0, 0.5), axis=1)
    df['Gate_50_Down_50_Up'] = df.apply(lambda r: apply_gate(r, 0.5, 0.5), axis=1)
    df['Gate_100_Down_100_Up'] = df.apply(lambda r: apply_gate(r, 1.0, 1.0), axis=1)

    anomalies = [2003, 2014, 2018, 2022]

    models_to_compare = {
        'Statistical Trend': trend_col,
        'ML Meta-Learner (Super Ens)': meta_col,
        'Gate (100% Down, 50% Up) [Ours]': 'Gate_100_Down_50_Up',
        'Gate (50% Down,  50% Up) [Bad]': 'Gate_50_Down_50_Up',
        'Gate (100% Down, 100% Up)': 'Gate_100_Down_100_Up'
    }

    for year in anomalies:
        if year not in df['year'].values:
            continue

        subset = df[df['year'] == year].copy()

        # Calculate actual national mean
        if 'harvested_area' in subset.columns and subset['harvested_area'].sum() > 0:
            actual = np.average(subset['kreisYield'], weights=subset['harvested_area'])
        else:
            actual = subset['kreisYield'].mean()

        print(f"\nYEAR {year} (Actual National Yield: {actual:.1f} dt/ha)")

        for label, col in models_to_compare.items():
            if col in subset.columns:
                if 'harvested_area' in subset.columns and subset['harvested_area'].sum() > 0:
                    pred_mean = np.average(subset[col], weights=subset['harvested_area'])
                    mae = np.average((subset[col] - subset['kreisYield']).abs(), weights=subset['harvested_area'])
                else:
                    pred_mean = subset[col].mean()
                    mae = (subset[col] - subset['kreisYield']).abs().mean()

                # Highlight our chosen presentation gate
                marker = "->" if "Ours" in label else "  "
                print(f"{marker} {label:<35}: Pred {pred_mean:.1f} (Avg MAE: {mae:.1f})")
        print("-" * 40)

if __name__ == "__main__":
    """
    Main execution block:
    Generates presentation-ready scientific visualizations to defend the "Anomaly Detection" framing.
    """
    print("--- Starting Scientific Presentation Visualizations ---")
    df = load_forecast_data()
    if df is not None:
        #plot_error_butterfly(df)
        #plot_f1_score_ablation(df)
        #plot_shock_localization(df)
        #log_yearly_recall_deepdive(df, threshold=0.04)
        #analyze_feature_discrepancy_2018_vs_2022(df)
        #evaluate_feature_ablation_all_years(df)
        #analyze_forecast_quality_impact(df)
        #analyze_comprehensive_physical_drivers(df)
        #evaluate_architectural_fixes_walkforward(df)
        #plot_national_yield_variance(df)
        evaluate_gate_ablation()
    print("--- Complete! ---")
