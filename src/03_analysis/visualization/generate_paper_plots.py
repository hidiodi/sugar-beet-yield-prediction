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
from src import config

# --- Config ---
OUTPUT_DIR = config.DATA_DIR / '07_paper_figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = config.DATA_DIR / '06_model_output/final_switched_forecast.csv'
GEO_FILE = config.FEATURE_ENGINEERING_CONFIG['FILE_PATHS']['GEOJSON_DISTRICTS']

sns.set_theme(style="whitegrid", font_scale=1.2)


def load_data():
    df = pd.read_csv(DATA_FILE)
    df['district_no'] = df['district_no'].astype(str).str.zfill(5)

    # Load Geo
    gdf = gpd.read_file(GEO_FILE)
    gdf['district_no'] = gdf['id']

    return pd.merge(gdf, df, on='district_no', how='inner')


def plot_regime_map_categorical(gdf, year):
    """Plot the Discrete Strategy Mode for a specific year."""
    subset = gdf[gdf['year'] == year].copy()

    # Create a color map for the 3 strategies
    # Crisis (Red), Opportunity (Green), Normal (Blue/Grey)
    condition_colors = {
        'Crisis (V2)': '#e74c3c',  # Red
        'Opportunity (V8)': '#2ecc71',  # Green
        'Normal (Ensemble)': '#3498db'  # Blue
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # Plot data
    subset.plot(
        column='Strategy_Mode',
        categorical=True,
        legend=True,
        color=subset['Strategy_Mode'].map(condition_colors),
        linewidth=0.1,
        edgecolor='0.8',
        ax=ax
    )

    ax.set_title(f"Regime Activation Map: {year}\n(Scenario Analysis)", fontsize=16)
    ax.axis('off')

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Opportunity (V8)', markerfacecolor='#2ecc71', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Normal (Ensemble)', markerfacecolor='#3498db', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Crisis (V2)', markerfacecolor='#e74c3c', markersize=15)
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    plt.savefig(OUTPUT_DIR / f'fig1_regime_map_{year}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Regime Map for {year}")


def plot_error_butterfly(df):
    """Compare Error Distributions (The value add)."""
    df['Ensemble_Error'] = (df['Actual'] - df['Native_Ensemble'])
    df['Strategy_Error'] = (df['Actual'] - df['Final_Pred'])

    # Filter for years where the strategy actually DID something (Not Normal)
    active = df[df['Strategy_Mode'] != 'Normal (Ensemble)']

    plt.figure(figsize=(10, 6))
    sns.kdeplot(active['Ensemble_Error'], fill=True, color='gray', label='Standard Ensemble', alpha=0.3)
    sns.kdeplot(active['Strategy_Error'], fill=True, color='purple', label='Regime Switching (Ours)', alpha=0.3)

    plt.title("Error Mitigation in Extreme Years (Crisis & Opportunity Modes)", fontsize=16)
    plt.xlabel("Prediction Error (dt/ha)")
    plt.xlim(-200, 200)
    plt.legend()

    plt.savefig(OUTPUT_DIR / 'fig2_error_butterfly.png', dpi=300)
    plt.close()
    print("Saved Butterfly Plot")


def plot_parity_by_regime(df):
    """Actual vs Predicted colored by Regime."""
    plt.figure(figsize=(10, 10))

    colors = {
        'Crisis (V2)': '#e74c3c',
        'Opportunity (V8)': '#2ecc71',
        'Normal (Ensemble)': '#3498db'
    }

    sns.scatterplot(data=df, x='Actual', y='Final_Pred', hue='Strategy_Mode', palette=colors, alpha=0.6, s=40)

    # Perfect fit line
    min_val, max_val = 300, 1000
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)

    plt.title("Prediction Accuracy by Regime", fontsize=16)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.savefig(OUTPUT_DIR / 'fig3_parity_plot.png', dpi=300)
    plt.close()
    print("Saved Parity Plot")


def main():
    print("Generating Paper Figures...")
    gdf = load_data()

    # 1. Maps
    plot_regime_map_categorical(gdf, 2018)  # The Crisis
    plot_regime_map_categorical(gdf, 2014)  # The Bumper

    # 2. Statistics
    plot_error_butterfly(gdf)
    plot_parity_by_regime(gdf)

    print(f"Done. Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()