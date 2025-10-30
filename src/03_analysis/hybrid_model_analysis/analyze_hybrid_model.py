# Description: A comprehensive script to generate a multi-faceted
#              explainability and diagnostics report for the final champion model.
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
import joblib
from pathlib import Path
import logging
import warnings
from sklearn.inspection import permutation_importance

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Input Files ---
FEATURES_FILE = Path('data/05_model_input/stage1_preseason_features.csv')
MODEL_FILE = Path('src/models/final_quantile_model_median.joblib')
GEOJSON_FILE = Path('data/01_raw/districts_official.geojson')
BACKTEST_FILE = Path('reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv')

# --- Output Directory ---
OUTPUT_DIR = Path('reports/figures/model_explainability_dashboard')


# --- Helper Function for Consistent Plotting ---
def setup_plot_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (14, 8), 'axes.titlesize': 20, 'axes.labelsize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 14
    })


# ==============================================================================
# === All analysis parts (part_0 to part_7) remain unchanged. ===
# === The error was in how the data was passed to them.         ===
# ==============================================================================
def part_0_data_exploration(df, gdf, top_features, output_dir):
    logging.info("--- 0. Starting Foundational Data Exploration ---")
    plt.figure(figsize=(16, 12));
    sns.heatmap(df[top_features].corr(), cmap='coolwarm', annot=False);
    plt.title("Correlation Matrix of Top 20 Features");
    plt.savefig(output_dir / '0a_correlation_heatmap.png', dpi=300, bbox_inches='tight');
    plt.close()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10));
    for i, feature in enumerate(top_features[:8]): sns.histplot(df[feature], kde=True, ax=axes[i // 4, i % 4]); axes[
        i // 4, i % 4].set_title(f'Distribution of {feature}', fontsize=14)
    plt.suptitle("Distributions of Top 8 Most Important Features", fontsize=22);
    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    plt.savefig(output_dir / '0b_top_features_distribution.png', dpi=300);
    plt.close()
    df_static = df.groupby('district_no')[['avg_sand_0_30cm', 'avg_clay_0_30cm', 'avg_elevation']].mean().reset_index();
    gdf_merged = gdf.merge(df_static, left_on='id', right_on='district_no')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8));
    gdf_merged.plot(column='avg_sand_0_30cm', ax=axes[0], legend=True, cmap='YlOrBr', legend_kwds={'label': "Sand %"});
    axes[0].set_title("Sand Content (0-30cm)");
    gdf_merged.plot(column='avg_clay_0_30cm', ax=axes[1], legend=True, cmap='Blues', legend_kwds={'label': "Clay %"});
    axes[1].set_title("Clay Content (0-30cm)");
    gdf_merged.plot(column='avg_elevation', ax=axes[2], legend=True, cmap='Greens',
                    legend_kwds={'label': "Elevation (m)"});
    axes[2].set_title("Average Elevation")
    for ax in axes: ax.set_axis_off();
    plt.suptitle("Geospatial Distribution of Key Static Features", fontsize=24);
    plt.savefig(output_dir / '0c_geospatial_feature_maps.png', dpi=300, bbox_inches='tight');
    plt.close()
    logging.info("✓ Part 0 complete.")


def part_1_global_importance(model, X, y, output_dir):
    logging.info("--- 1. Generating Global Feature Importance Plots ---")
    explainer = shap.TreeExplainer(model);
    shap_values = explainer(X.sample(1000, random_state=42));
    plt.figure(figsize=(12, 10));
    shap.summary_plot(shap_values, X.sample(1000, random_state=42), show=False, plot_size=None);
    plt.title("Global Importance: SHAP Summary Plot");
    plt.tight_layout();
    plt.savefig(output_dir / '1a_shap_summary_plot.png', dpi=300);
    plt.close()
    perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1);
    sorted_idx = perm_importance.importances_mean.argsort()[-20:];
    plt.figure(figsize=(12, 10));
    plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx]);
    plt.xlabel("Permutation Importance (Impact on Model Error)");
    plt.title("Global Importance: Permutation Importance (Top 20)");
    plt.tight_layout();
    plt.savefig(output_dir / '1b_permutation_importance.png', dpi=300);
    plt.close()
    logging.info("✓ Part 1 complete.")
    return X.columns[sorted_idx][::-1]


def part_5_synergistic_effects(model, X, output_dir):
    logging.info("--- 5. Discovering and Visualizing Feature Interactions ---")
    explainer = shap.TreeExplainer(model);
    X_sample = X.sample(500, random_state=42);
    shap_interaction_values = explainer.shap_interaction_values(X_sample);
    mean_abs_interactions = np.abs(shap_interaction_values).mean(0);
    np.fill_diagonal(mean_abs_interactions, 0);
    top_interactions = [];
    for i in range(mean_abs_interactions.shape[0]):
        for j in range(i + 1, mean_abs_interactions.shape[0]): top_interactions.append(
            ((X.columns[i], X.columns[j]), mean_abs_interactions[i, j]))
    top_interactions.sort(key=lambda x: x[1], reverse=True);
    logging.info("Top 5 discovered interactions (Feature Pair, SHAP Interaction Value):");
    for (f1, f2), val in top_interactions[:5]: print(f"  - ({f1}, {f2}): {val:.4f}")
    top_interaction_pair = top_interactions[0][0];
    logging.info(f"Visualizing strongest interaction: {top_interaction_pair}");
    shap_values = explainer(X);
    plt.figure();
    shap.dependence_plot(top_interaction_pair[0], shap_values.values, X, interaction_index=top_interaction_pair[1]);
    plt.title(f"Strongest Discovered Interaction: {top_interaction_pair[0]} & {top_interaction_pair[1]}");
    plt.tight_layout();
    plt.savefig(output_dir / '5a_strongest_interaction_plot.png', dpi=300);
    plt.close()
    logging.info("✓ Part 5 complete.")


def part_6_diagnosing_weaknesses(df_backtest, gdf, output_dir):
    logging.info("--- 6. Diagnosing Model Blind Spots ---")
    df_backtest['final_error'] = df_backtest['kreisYield'] - df_backtest['predicted_yield_median'];
    df_bias = df_backtest.groupby('district_no')['final_error'].mean().reset_index();
    gdf_bias = gdf.merge(df_bias, left_on='id', right_on='district_no');
    fig, ax = plt.subplots(1, 1, figsize=(10, 10));
    gdf_bias.plot(column='final_error', ax=ax, legend=True, cmap='RdBu_r',
                  legend_kwds={'label': "Average Prediction Error (dt/ha)"});
    ax.set_title("Geospatial Bias: Where is the Model Consistently Wrong?");
    ax.set_axis_off();
    plt.savefig(output_dir / '6a_geospatial_error_bias_map.png', dpi=300, bbox_inches='tight');
    plt.close()
    plt.figure(figsize=(18, 8));
    sns.boxplot(data=df_backtest, x='year', y='final_error', palette='viridis');
    plt.axhline(0, color='r', linestyle='--');
    plt.title("Model Error Distribution by Year");
    plt.ylabel("Prediction Error (Actual - Predicted)");
    plt.xticks(rotation=45);
    plt.tight_layout();
    plt.savefig(output_dir / '6b_error_by_year_boxplot.png', dpi=300);
    plt.close()
    logging.info("✓ Part 6 complete.")


def part_7_local_explainability(model, df_backtest, X_full_for_lookup, output_dir):
    logging.info("--- 7. Performing a Local Explanation Case Study ---")
    df_backtest['final_error'] = df_backtest['kreisYield'] - df_backtest['predicted_yield_median'];
    df_backtest['abs_error'] = df_backtest['final_error'].abs();
    case_study_row = df_backtest.loc[df_backtest['abs_error'].idxmax()];
    district = case_study_row['district_no'];
    year = case_study_row['year'];
    logging.info(f"Case Study: Largest error occurred in District {district}, Year {year}")
    case_study_X = X_full_for_lookup[
        (X_full_for_lookup['district_no'] == district) & (X_full_for_lookup['year'] == year)].drop(
        columns=['district_no', 'year'])
    if case_study_X.empty: logging.error("Could not find case study features. Skipping local explanation."); return
    explainer = shap.TreeExplainer(model);
    shap_values = explainer(case_study_X);
    plt.figure();
    shap.waterfall_plot(shap_values[0], show=False);
    plt.title(f"Local Explanation (Waterfall Plot)\nDistrict: {district}, Year: {year}");
    plt.tight_layout();
    plt.savefig(output_dir / '7a_local_explanation_waterfall.png', dpi=300);
    plt.close()
    logging.info("✓ Part 7 complete.")


# ==============================================================================
# === MAIN ORCHESTRATION FUNCTION ===
# ==============================================================================
def main():
    """Main function to run the entire explainability and diagnostics pipeline."""
    logging.info("====== Starting Model Explainability and Diagnostics Dashboard Generation ======")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    # --- Load Data and Model ---
    try:
        df_full = pd.read_csv(FEATURES_FILE)
        model = joblib.load(MODEL_FILE)
        gdf = gpd.read_file(GEOJSON_FILE)
        df_backtest = pd.read_csv(BACKTEST_FILE)
        gdf['id'] = gdf['id'].astype(str).str.zfill(5)
        df_full['district_no'] = df_full['district_no'].astype(str).str.zfill(5)
        df_backtest['district_no'] = df_backtest['district_no'].astype(str).str.zfill(5)
    except FileNotFoundError as e:
        logging.error(f"❌ FATAL: A required file was not found. Details: {e}");
        return

    # --- CORRECTED DATA PREPARATION ---
    # 1. Prepare the data exactly as it was for training
    df_full.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df_full['forecast_residual'] = df_full['kreisYield'] - df_full['stage1_forecast']

    # 2. Define all columns needed for the analysis
    feature_cols = [f for f in model.feature_names_in_ if f in df_full.columns]
    all_needed_cols = ['district_no', 'year', 'kreisYield', 'forecast_residual'] + feature_cols

    # 3. Create a single, guaranteed-clean dataframe
    df_clean = df_full[all_needed_cols].dropna().copy()

    if df_clean.empty:
        logging.error("FATAL: Dataframe is empty after cleaning. No data to analyze.");
        return

    logging.info(f"Cleaned data for analysis contains {len(df_clean)} rows.")

    # 4. Select final, aligned X and y for modeling functions
    X = df_clean[feature_cols]
    y = df_clean['forecast_residual']

    # --- Run Analysis for Each Part ---
    top_features = part_1_global_importance(model, X, y, OUTPUT_DIR)
    part_0_data_exploration(df_clean, gdf, top_features, OUTPUT_DIR)
    part_5_synergistic_effects(model, X, OUTPUT_DIR)
    part_6_diagnosing_weaknesses(df_backtest, gdf, OUTPUT_DIR)
    # The 'df_clean' dataframe is now passed for the case study lookup
    part_7_local_explainability(model, df_backtest, df_clean, OUTPUT_DIR)

    logging.info("\n====== ✅ Explainability Dashboard Generation Complete! ======")
    logging.info(f"All plots and analyses saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
