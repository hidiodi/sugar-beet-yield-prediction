import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import sys

# --- Project Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- CONFIGURATION ---
FORECAST_FILE = Path(config.MODEL_COMPARISON_CONFIG['OUTPUT_DIR']) / 'super_ensemble_final_forecast_TSCV.csv'
SHAPEFILE_SEARCH_DIR = config.DATA_DIR / '01_raw'
OUTPUT_IMG = config.BASE_DIR / 'reports/figures/final_error_map_v1.png'
OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)


def find_geodata():
    """Attempts to find a German Districts geospatial file (.geojson or .shp)."""

    # Priority 1: User specified GeoJSON (Best Quality)
    p1 = SHAPEFILE_SEARCH_DIR / 'districts_official.geojson'
    if p1.exists(): return p1

    # Priority 2: Standard Shapefile paths
    p2 = SHAPEFILE_SEARCH_DIR / 'shapefiles/kreise_vg250.shp'
    if p2.exists(): return p2

    # Priority 3: Recursive search for any valid geodata
    extensions = ['*.geojson', '*.shp']
    candidates = []
    for ext in extensions:
        candidates.extend(list(SHAPEFILE_SEARCH_DIR.rglob(ext)))

    # Sort candidates (prefer files with 'official' or 'kreis' in name)
    for c in candidates:
        if 'official' in c.name.lower() or 'kreis' in c.name.lower():
            return c

    if candidates: return candidates[0]
    return None


def generate_map():
    if not FORECAST_FILE.exists():
        logging.error(f"Forecast file not found at: {FORECAST_FILE}")
        return

    logging.info("--- Generating Final Spatial Error Map ---")

    # 1. Load Forecast Data
    df = pd.read_csv(FORECAST_FILE)
    df['Error'] = (df['kreisYield'] - df['Super_Ensemble_pred']).abs()
    district_stats = df.groupby('district_no')['Error'].mean().reset_index()

    # Convert ID to string for merging (ensure 5 digits)
    district_stats['district_no'] = district_stats['district_no'].astype(int).astype(str).str.zfill(5)

    logging.info(f"Loaded forecasts for {len(district_stats)} districts.")
    logging.info(f"Average MAE: {district_stats['Error'].mean():.2f} dt/ha")

    # 2. Load GeoData
    geo_path = find_geodata()
    if not geo_path:
        logging.error("No .geojson or .shp file found in data/01_raw. Cannot plot map.")
        return

    logging.info(f"Loading Map Data: {geo_path.name}")
    gdf = gpd.read_file(geo_path)

    # 3. Match District IDs
    # Common German keys: 'AGS', 'RS', 'CC_2', 'ARS_0', 'id'
    join_col = None

    # Heuristic: Check columns for ID-like patterns
    possible_id_cols = ['AGS', 'RS', 'ARS', 'CC_2', 'id', 'district_no', 'ags']

    # First check explicit names
    for col in possible_id_cols:
        if col in gdf.columns:
            join_col = col
            break

    # Fallback: scan columns for matching content
    if not join_col:
        for col in gdf.columns:
            # Check if column content resembles our IDs (5 digits)
            if gdf[col].dtype == 'object' or pd.api.types.is_string_dtype(gdf[col]):
                sample = str(gdf[col].iloc[0])
                if sample.isdigit() and len(sample) >= 2:
                    join_col = col
                    break

    if not join_col:
        logging.error("Could not find matching District ID column in map file.")
        logging.info(f"Available columns: {gdf.columns.tolist()}")
        return

    logging.info(f"Joining on map column: {join_col}")
    gdf['district_no'] = gdf[join_col].astype(str).str.zfill(5)

    # 4. Merge
    merged = gdf.merge(district_stats, on='district_no', how='left')

    # 5. Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))

    merged.plot(
        column='Error',
        ax=ax,
        legend=True,
        cmap='RdYlGn_r',  # Red (Bad) to Green (Good)
        missing_kwds={'color': 'lightgrey', 'label': 'No Data'},
        legend_kwds={'label': "Mean Absolute Error (dt/ha)", 'orientation': "horizontal", 'shrink': 0.8}
    )

    ax.set_title('Super Ensemble Performance (2000-2024)\nMAE by District', fontsize=16, fontweight='bold')
    ax.set_axis_off()

    # Annotate worst districts
    if not district_stats.empty:
        top_bad = district_stats.sort_values('Error', ascending=False).head(3)
        logging.info("\nTop 3 Hardest Districts:")
        for _, row in top_bad.iterrows():
            logging.info(f"  District {row['district_no']}: MAE {row['Error']:.1f}")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    logging.info(f"\n✅ Map saved to: {OUTPUT_IMG}")


if __name__ == "__main__":
    generate_map()