import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
import sys

# --- Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config as global_config
import importlib
analysis_config = importlib.import_module("src.03_analysis.config")
models_config = importlib.import_module("src.02_models.config")

# --- Config ---
CONFIG = analysis_config.MODEL_COMPARISON_CONFIG
OUTPUT_DIR = global_config.DATA_DIR / '07_paper_figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input Files
FORECAST_FILE = Path(CONFIG['OUTPUT_DIR']) / 'super_ensemble_final_forecast_TSCV.csv'
TRAINING_FILE = Path(CONFIG['OUTPUT_DIR']) / 'super_ensemble_training_data.csv'
ABLATION_FILE = Path(CONFIG['OUTPUT_DIR']) / 'ablation_results.csv'
GEO_FILE = models_config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']['GEOJSON_DISTRICTS']

sns.set_theme(style="whitegrid", font_scale=1.2)


def load_data():
    if not FORECAST_FILE.exists():
        print(f"Error: Forecast file not found at {FORECAST_FILE}")
        return None, None

    df = pd.read_csv(FORECAST_FILE)
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)

    # Merge with Training Data to get Statistical Trend
    if TRAINING_FILE.exists():
        df_train = pd.read_csv(TRAINING_FILE)
        df_train['district_no'] = df_train['district_no'].astype(str).str.zfill(5)
        if 'Statistical_Trend_pred' in df_train.columns:
            # Merge
            df = pd.merge(df, df_train[['year', 'district_no', 'Statistical_Trend_pred']],
                          on=['year', 'district_no'], how='left')

    # Rename for consistency with old script or plotting logic
    if 'kreisYield' in df.columns:
        df.rename(columns={'kreisYield': 'Actual'}, inplace=True)
    if 'Super_Ensemble_pred' in df.columns:
        df.rename(columns={'Super_Ensemble_pred': 'Final_Pred'}, inplace=True)
    if 'Statistical_Trend_pred' in df.columns:
         df.rename(columns={'Statistical_Trend_pred': 'Trend_Pred'}, inplace=True)

    # Load Geo
    if not GEO_FILE.exists():
        print("Warning: GeoJSON not found. Map plots will be skipped.")
        return df, None

    gdf = gpd.read_file(GEO_FILE)
    # Flexible ID finding
    id_col = None
    for c in ['AGS', 'RS', 'id', 'district_no']:
        if c in gdf.columns: id_col = c; break

    if id_col:
        gdf['district_no'] = gdf[id_col].astype(str).str.zfill(5)
        merged_gdf = pd.merge(gdf, df, on='district_no', how='inner')
        return df, merged_gdf
    else:
        return df, None


def plot_regime_map_categorical(gdf, year):
    """Plot the Discrete Strategy Mode for a specific year."""
    if gdf is None: return

    subset = gdf[gdf['year'] == year].copy()
    if subset.empty: return

    # Map 'Predicted_Best_Model' to colors
    # Models: Statistical Trend, Hybrid XGB, Native Ensemble, V31 Solar Gated, Robust Linear

    # We want to highlight deviations from Trend
    # Normal = Statistical Trend
    # Crisis/Opportunity = Others

    unique_models = subset['Predicted_Best_Model'].unique()
    print(f"Models in {year}: {unique_models}")

    # Palette
    # Trend = Blue (Normal)
    # V31 Solar = Green (Opportunity?)
    # Native/Hybrid = Red/Orange (Correction?)

    model_colors = {
        'Statistical_Trend': '#3498db', # Blue
        'V31_Solar_Gated': '#2ecc71',   # Green
        'Native_Ensemble': '#e74c3c',   # Red
        'Hybrid_XGB': '#e67e22',        # Orange
        'Robust_Linear': '#9b59b6'      # Purple
    }

    # Fallback for unknown models
    subset['color'] = subset['Predicted_Best_Model'].map(model_colors).fillna('gray')

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    subset.plot(
        color=subset['color'],
        linewidth=0.1,
        edgecolor='0.8',
        ax=ax
    )

    ax.set_title(f"Model Selection Map: {year}", fontsize=16)
    ax.axis('off')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for model, color in model_colors.items():
        if model in unique_models:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=model, markerfacecolor=color, markersize=15))

    ax.legend(handles=legend_elements, loc='lower left')

    plt.savefig(OUTPUT_DIR / f'fig1_regime_map_{year}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Regime Map for {year}")


def plot_error_butterfly(df):
    """Compare Error Distributions."""
    # We need a baseline to compare against. Assuming 'Trend_Pred' is in df if loaded correctly.
    # If not present in final forecast, we might need to rely on what we have.
    # 'super_ensemble_final_forecast_TSCV.csv' might not have 'Statistical_Trend_pred' unless we kept it.
    # Check 'generate_forecast' in 'execute_ensemble_forecast.py': it saves cols starting with Prob_ but seemingly not other preds explicitly except Super_Ensemble_pred.
    # However, 'super_ensemble_training_data.csv' had them.
    # Let's hope the forecast file includes them or we can't plot the comparison easily without reloading training data.

    if 'Trend_Pred' not in df.columns:
        print("Skipping Butterfly Plot: 'Trend_Pred' (Statistical Trend) not found in data.")
        return

    df['Trend_Error'] = (df['Actual'] - df['Trend_Pred'])
    df['Ensemble_Error'] = (df['Actual'] - df['Final_Pred'])

    # Filter for interesting years or deviations
    # Let's just plot overall
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['Trend_Error'], fill=True, color='gray', label='Statistical Trend (Baseline)', alpha=0.3)
    sns.kdeplot(df['Ensemble_Error'], fill=True, color='purple', label='Super Ensemble (Ours)', alpha=0.3)

    plt.title("Error Distribution Comparison", fontsize=16)
    plt.xlabel("Prediction Error (dt/ha)")
    plt.xlim(-200, 200)
    plt.legend()

    plt.savefig(OUTPUT_DIR / 'fig2_error_butterfly.png', dpi=300)
    plt.close()
    print("Saved Butterfly Plot")


def plot_time_series_aggregated(df):
    """Figure 2: Time Series of Actual vs. Predicted Yields (2000–2024)."""
    # Aggregate by year (mean yield across districts)
    yearly = df.groupby('year')[['Actual', 'Final_Pred']].mean().reset_index()

    # If we have Trend, plot it too
    has_trend = 'Trend_Pred' in df.columns
    if has_trend:
        yearly_trend = df.groupby('year')['Trend_Pred'].mean().reset_index()
        yearly = pd.merge(yearly, yearly_trend, on='year')

    plt.figure(figsize=(12, 6))

    # Plot Lines
    plt.plot(yearly['year'], yearly['Actual'], 'o-', color='black', label='Actual Yield', linewidth=2)
    plt.plot(yearly['year'], yearly['Final_Pred'], 's--', color='#2ca02c', label='Super Ensemble', linewidth=2)

    if has_trend:
        plt.plot(yearly['year'], yearly['Trend_Pred'], ':', color='gray', label='Statistical Trend', linewidth=1.5)

    # Highlight 2003 and 2018
    for year in [2003, 2018]:
        if year in yearly['year'].values:
            plt.axvline(x=year, color='red', alpha=0.3, linestyle='-', linewidth=10)
            # plt.text(year, yearly['Actual'].max() + 20, str(year), ha='center', color='red')

    plt.title("National Average Yield: Forecast vs Reality (2000-2024)", fontsize=16)
    plt.ylabel("Yield (dt/ha)")
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'fig2_time_series.png', dpi=300)
    plt.close()
    print("Saved Time Series Plot")


def plot_ablation_results():
    """Figure 3: Ablation Results."""
    if not ABLATION_FILE.exists():
        print("Ablation results file not found. Skipping Figure 3.")
        return

    df_ab = pd.read_csv(ABLATION_FILE)

    # Setup Data for Plotting
    # We want to compare MAE_Overall, MAE_2003, MAE_2018
    # Melt the dataframe
    df_melt = df_ab.melt(id_vars=['Experiment'], value_vars=['MAE_Overall', 'MAE_2003', 'MAE_2018'],
                         var_name='Metric', value_name='MAE')

    # Clean Metric Names
    df_melt['Metric'] = df_melt['Metric'].replace({
        'MAE_Overall': 'Overall',
        'MAE_2003': '2003 (Drought)',
        'MAE_2018': '2018 (Drought)'
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x='Metric', y='MAE', hue='Experiment', palette='viridis')

    plt.title("Ablation Study: Impact on Forecast Error", fontsize=16)
    plt.ylabel("Mean Absolute Error (dt/ha)")
    plt.xlabel("Evaluation Context")
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_ablation_results.png', dpi=300)
    plt.close()
    print("Saved Ablation Plot")


def plot_parity_by_regime(df):
    """Actual vs Predicted colored by Best Model."""
    plt.figure(figsize=(10, 10))

    model_colors = {
        'Statistical_Trend': '#3498db', # Blue
        'V31_Solar_Gated': '#2ecc71',   # Green
        'Native_Ensemble': '#e74c3c',   # Red
        'Hybrid_XGB': '#e67e22',        # Orange
        'Robust_Linear': '#9b59b6'      # Purple
    }

    # Fallback palette if model names don't match exactly
    sns.scatterplot(data=df, x='Actual', y='Final_Pred', hue='Predicted_Best_Model', palette=model_colors, alpha=0.6, s=40)

    # Perfect fit line
    min_val, max_val = 300, 1000
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)

    plt.title("Prediction Accuracy by Selected Model", fontsize=16)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_parity_plot.png', dpi=300)
    plt.close()
    print("Saved Parity Plot")


def main():
    print("Generating Paper Figures...")
    df, gdf = load_data()

    if df is not None:
        # 1. Maps
        plot_regime_map_categorical(gdf, 2018)  # The Crisis
        plot_regime_map_categorical(gdf, 2014)  # The Bumper

        # 2. Statistics & Time Series
        plot_error_butterfly(df)
        plot_time_series_aggregated(df)
        plot_parity_by_regime(df)

    # 3. Ablation
    plot_ablation_results()

    print(f"Done. Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
