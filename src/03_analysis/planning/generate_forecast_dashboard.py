# File: src/planning/generate_interactive_value_story.py
# Description: FINALIZED & CORRECTED. Fixes the "blank map" issue by robustly handling
# NaN values after the merge, ensuring a valid GeoJSON is always generated.

import pandas as pd
import geopandas as gpd
import joblib
import os
import folium
from folium.features import GeoJsonTooltip

# --- Configuration ---
CASE_STUDY_YEAR = 2018

MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion_final.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
OUTPUT_DIR = os.path.join('reports', 'value_story')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIER_LABELS = [
    '5: High Vulnerability (>10% loss)',
    '4: Caution Advised (2-10% loss)',
    '3: Standard Expectation (+/- 2%)',
    '2: Favorable Outlook (2-10% gain)',
    '1: Prime Opportunity (>10% gain)'
]
TIER_COLORS = ['#d73027', '#fc8d59', '#ffffbf', '#91cf60', '#1a9850']


def create_value_story():
    """Generates the single, self-contained HTML file with the interactive comparison map."""
    print(f"--- Generating Interactive Value Story for the {CASE_STUDY_YEAR} Season ---")

    # --- Step 1: Load data and make a re-trended prediction ---
    model = joblib.load(MODEL_PATH)
    full_data = pd.read_csv(DATA_PATH)

    full_data['yield_trend'] = full_data.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    ).fillna(method='ffill').fillna(method='bfill')

    model_input_df = full_data[full_data['year'] == CASE_STUDY_YEAR].copy()
    if model_input_df.empty:
        print(f"ERROR: No data found for the year {CASE_STUDY_YEAR}.")
        return

    feature_cols = model.feature_names_in_
    detrended_prediction = model.predict(model_input_df[feature_cols])
    model_input_df['model_forecast'] = detrended_prediction + model_input_df['yield_trend']

    comparison_df = model_input_df.rename(columns={'kreisYield': 'actual_yield', 'yield_trend': 'baseline_yield'})
    comparison_df['forecast_error'] = comparison_df['model_forecast'] - comparison_df['actual_yield']

    # --- Step 2: Calculate Deviations and Assign Tiers ---
    comparison_df['model_deviation_pct'] = ((comparison_df['model_forecast'] - comparison_df['baseline_yield']) /
                                            comparison_df['baseline_yield']) * 100
    comparison_df['actual_deviation_pct'] = ((comparison_df['actual_yield'] - comparison_df['baseline_yield']) /
                                             comparison_df['baseline_yield']) * 100

    bins = [-float('inf'), -10, -2, 2, 10, float('inf')]
    comparison_df['forecast_tier'] = pd.cut(comparison_df['model_deviation_pct'], bins=bins, labels=TIER_LABELS,
                                            right=False)
    comparison_df['actual_tier'] = pd.cut(comparison_df['actual_deviation_pct'], bins=bins, labels=TIER_LABELS,
                                          right=False)

    # --- Step 3: Create the Interactive Map with Folium ---
    print("Creating interactive map with Forecast vs. Reality layers...")
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf['district_no'] = gdf['id'].astype(str).str.zfill(5)
    comparison_df['district_no'] = comparison_df['district_no'].astype(str).str.zfill(5)
    merged_gdf = gdf.merge(comparison_df, on='district_no', how='left')

    # ============================ THE FIX ============================
    # ### CRITICAL FIX: Handle NaN values after the left merge ###
    # This prevents the creation of an invalid GeoJSON file which causes the blank map.

    # Fill categorical tier columns with a specific 'N/A' string
    merged_gdf['forecast_tier'] = merged_gdf['forecast_tier'].cat.add_categories(['N/A']).fillna('N/A')
    merged_gdf['actual_tier'] = merged_gdf['actual_tier'].cat.add_categories(['N/A']).fillna('N/A')

    # Prepare string columns for the tooltip, explicitly handling NaNs
    str_cols_to_create = {
        'baseline_yield': 'baseline_yield_str', 'model_forecast': 'model_forecast_str',
        'actual_yield': 'actual_yield_str', 'forecast_error': 'forecast_error_str'
    }
    for in_col, out_col in str_cols_to_create.items():
        merged_gdf[out_col] = merged_gdf[in_col].apply(
            lambda x: f"{x:+.1f} dt/ha" if pd.notna(x) else "Data not available")

    merged_gdf['model_deviation_str'] = merged_gdf['model_deviation_pct'].apply(
        lambda x: f"{x:+.1f}% vs baseline" if pd.notna(x) else "N/A")
    # ===============================================================

    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, tiles='CartoDB positron')
    color_map = dict(zip(TIER_LABELS, TIER_COLORS))
    color_map['N/A'] = '#cccccc'  # Add a color for the 'Not Available' category

    tooltip = GeoJsonTooltip(
        fields=['name', 'baseline_yield_str', 'model_forecast_str', 'actual_yield_str', 'model_deviation_str',
                'forecast_error_str'],
        aliases=['District:', 'Baseline Expectation:', "<b>Model's March Forecast:</b>",
                 '<b>Actual Harvest Result:</b>', 'Model Deviation:', '<b>Forecast Error:</b>'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
        sticky=True
    )

    # The rest of the plotting logic is now robust because NaNs are handled
    folium.GeoJson(
        merged_gdf,
        name=f"The Forecast (Model's Prediction in March {CASE_STUDY_YEAR})",
        style_function=lambda feature: {
            'fillColor': color_map.get(feature['properties']['forecast_tier'], '#cccccc'),
            'color': 'black', 'weight': 0.5, 'fillOpacity': 0.7,
        },
        tooltip=tooltip
    ).add_to(m)

    folium.GeoJson(
        merged_gdf,
        name=f"The Reality (Actual Outcome in {CASE_STUDY_YEAR})",
        style_function=lambda feature: {
            'fillColor': color_map.get(feature['properties']['actual_tier'], '#cccccc'),
            'color': 'black', 'weight': 0.5, 'fillOpacity': 0.7,
        },
        tooltip=tooltip,
        show=False
    ).add_to(m)

    folium.LayerControl().add_to(m)

    legend_html = f'''
     <div style="position: fixed; bottom: 30px; left: 30px; width: 240px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: .9;">
     <div style="padding: 5px; background-color: #f2f2f2;"><b>Yield Deviation ({CASE_STUDY_YEAR})</b></div>
     <div style="padding: 5px;">
     <i class="fa fa-square" style="color:{TIER_COLORS[4]}"></i> Prime (>+10%)<br>
     <i class="fa fa-square" style="color:{TIER_COLORS[3]}"></i> Favorable (+2% to +10%)<br>
     <i class="fa fa-square" style="color:{TIER_COLORS[2]}"></i> Standard (+/- 2%)<br>
     <i class="fa fa-square" style="color:{TIER_COLORS[1]}"></i> Caution (-2% to -10%)<br>
     <i class="fa fa-square" style="color:{TIER_COLORS[0]}"></i> Vulnerable (<-10%)<br>
     </div></div>'''
    m.get_root().html.add_child(folium.Element(legend_html))

    output_path = os.path.join(OUTPUT_DIR, f'value_story_{CASE_STUDY_YEAR}.html')
    m.save(output_path)

    print(f"\n✅ Interactive Value Story saved to {output_path}")
    print("--- Open this file in a web browser to explore the model's advantage. ---")


if __name__ == "__main__":
    create_value_story()
