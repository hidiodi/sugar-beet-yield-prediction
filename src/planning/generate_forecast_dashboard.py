# File: src/planning/generate_interactive_value_story.py
# Description: Creates a single, interactive HTML file to definitively explain the model's value
# using a historical case study (e.g., the 2018 drought year).

import pandas as pd
import geopandas as gpd
import joblib
import os
import folium
import json
from branca.element import MacroElement
from jinja2 import Template

# --- Configuration ---
# You can change this to any year to create a different case study
CASE_STUDY_YEAR = 2018

MODEL_PATH = os.path.join('src/models', 'final_xgb_model_random_split.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
OUTPUT_DIR = os.path.join('reports', 'value_story')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIER_LABELS = [
    '5: High Vulnerability', '4: Caution Advised', '3: Standard Expectation',
    '2: Favorable Outlook', '1: Prime Opportunity'
]


def create_value_story():
    """Generates the single, self-contained HTML file with the interactive comparison map."""
    print(f"--- Generating Interactive Value Story for the {CASE_STUDY_YEAR} Season ---")

    # --- Step 1: Prepare the Hindsight Data ---
    model = joblib.load(MODEL_PATH)
    full_data = pd.read_csv(DATA_PATH)

    historical_df = full_data[full_data['year'] < CASE_STUDY_YEAR]
    historical_baseline = historical_df.groupby('district_no')['kreisYield'].mean().reset_index()
    historical_baseline.rename(columns={'kreisYield': 'historical_avg_yield'}, inplace=True)

    model_input_df = full_data[full_data['year'] == CASE_STUDY_YEAR].copy()
    feature_cols = model.get_booster().feature_names
    model_input_df['model_forecast'] = model.predict(model_input_df[feature_cols])

    comparison_df = pd.merge(historical_baseline, model_input_df, on='district_no', how='inner')
    comparison_df.rename(columns={'kreisYield': 'actual_yield'}, inplace=True)

    # --- Step 2: Calculate Deviations and Assign Tiers ---
    comparison_df['model_deviation_pct'] = ((comparison_df['model_forecast'] - comparison_df['historical_avg_yield']) /
                                            comparison_df['historical_avg_yield']) * 100
    comparison_df['actual_deviation_pct'] = ((comparison_df['actual_yield'] - comparison_df['historical_avg_yield']) /
                                             comparison_df['historical_avg_yield']) * 100

    # Use qcut to assign the actual outcome to one of our 5 tiers
    comparison_df['actual_tier'] = pd.qcut(comparison_df['actual_deviation_pct'], q=5, labels=TIER_LABELS,
                                           duplicates='drop')

    # Prepare strings for the tooltip
    for col in ['historical_avg_yield', 'model_forecast', 'actual_yield']:
        comparison_df[f'{col}_str'] = comparison_df[col].round(1).astype(str) + ' dt/ha'
    comparison_df['model_deviation_str'] = comparison_df['model_deviation_pct'].apply(lambda x: f"{x:+.1f}%")

    # --- Step 3: Create the Interactive Map with Folium ---
    print("Creating interactive comparison map...")
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf['district_no'] = gdf['id'].astype(int)
    merged_gdf = gdf.merge(comparison_df, on='district_no', how='left')

    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, tiles='CartoDB positron')

    # --- MODIFIED: Robust coloring method ---
    tier_colors = {
        '1: Prime Opportunity': '#1a9850', '2: Favorable Outlook': '#91cf60',
        '3: Standard Expectation': '#ffffbf', '4: Caution Advised': '#fc8d59',
        '5: High Vulnerability': '#d73027', 'nan': '#cccccc'
    }

    # Create a style function that colors each district based on its 'actual_tier'
    def style_function(feature):
        tier = feature['properties']['actual_tier']
        return {
            'fillColor': tier_colors.get(str(tier), '#cccccc'),  # Use grey for missing tiers
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
        }

    # Add the GeoJson layer with our style function and the powerful tooltip
    tooltip = folium.GeoJsonTooltip(
        fields=['name', 'historical_avg_yield_str', 'model_forecast_str', 'actual_yield_str', 'model_deviation_str'],
        aliases=['District:', 'Historical Expectation:', "<b>Model's March Forecast:</b>",
                 '<b>Actual Harvest Result:</b>', "Model Deviation:"],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
        sticky=True
    )

    folium.GeoJson(
        merged_gdf,
        style_function=style_function,
        tooltip=tooltip
    ).add_to(m)

    # --- Add a custom legend ---
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 220px; height: 150px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: .85;
     ">&nbsp; <b>Actual Outcome Tiers ({})</b><br>
     &nbsp; <i class="fa fa-square" style="color:#1a9850"></i> &nbsp; 1: Prime Opportunity<br>
     &nbsp; <i class="fa fa-square" style="color:#91cf60"></i> &nbsp; 2: Favorable Outlook<br>
     &nbsp; <i class="fa fa-square" style="color:#ffffbf"></i> &nbsp; 3: Standard Expectation<br>
     &nbsp; <i class="fa fa-square" style="color:#fc8d59"></i> &nbsp; 4: Caution Advised<br>
     &nbsp; <i class="fa fa-square" style="color:#d73027"></i> &nbsp; 5: High Vulnerability<br>
     </div>
     '''.format(CASE_STUDY_YEAR)
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- Step 4: Save to a single HTML file ---
    output_path = os.path.join(OUTPUT_DIR, f'value_story_{CASE_STUDY_YEAR}.html')
    m.save(output_path)

    print(f"\n✅ Interactive Value Story saved to {output_path}")
    print("--- Open this file in a web browser to explore the model's advantage. ---")


if __name__ == "__main__":
    create_value_story()