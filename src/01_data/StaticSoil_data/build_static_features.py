# File: src/01_data/StaticSoil_data/build_static_features.py
# Description: V6 - NUMERICALLY STABLE & CORRECTED PHYSICS.
# - Implements the correct Saxton-Rawls (2006) equations.
# - The previous versions were numerically unstable and physically incorrect. This is the fix.

import ee
import geemap
import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
import time
import sys
import numpy as np

# --- Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEBUG_LOGGING_COUNT = 0
MAX_DEBUG_ROWS = 3


# ==============================================================================
# === V3 PEDOTRANSFER FUNCTION (STABLE & CORRECTED) ============================
# ==============================================================================
def _parameterize_wofost_soil_inputs_v3(row):
    """
    V3 of the PTF wrapper. Implements the correct, stable Saxton-Rawls (2006) equations.
    """
    global DEBUG_LOGGING_COUNT

    sand_pct = row['avg_sand_0_100cm']
    clay_pct = row['avg_clay_0_100cm']
    som_pct = row['avg_som_0_100cm']
    bdod = row['avg_bdod_0_100cm']

    if DEBUG_LOGGING_COUNT < MAX_DEBUG_ROWS:
        logging.info(f"--- PTF DEBUG START (Row {DEBUG_LOGGING_COUNT}) ---")
        logging.info(
            f"  INPUTS -> sand: {sand_pct:.2f}%, clay: {clay_pct:.2f}%, som: {som_pct:.2f}%, bdod: {bdod:.3f} g/cm3")

    params = {}

    # --- Corrected Saxton-Rawls (2006) Equations ---
    # These are the stable equations from the paper.

    # 1. Wilting Point (1500 kPa)
    theta_1500_t = -0.024 * (sand_pct / 100) + 0.487 * (clay_pct / 100) + 0.006 * som_pct + \
                   0.005 * (sand_pct / 100) * som_pct - 0.013 * (clay_pct / 100) * som_pct + \
                   0.068 * (sand_pct / 100) * (clay_pct / 100) + 0.031
    theta_1500_t_adj = theta_1500_t + (0.14 * theta_1500_t - 0.02)

    # 2. Field Capacity (33 kPa)
    theta_33_t = -0.251 * (sand_pct / 100) + 0.195 * (clay_pct / 100) + 0.011 * som_pct + \
                 0.006 * (sand_pct / 100) * som_pct - 0.027 * (clay_pct / 100) * som_pct + \
                 0.452 * (sand_pct / 100) * (clay_pct / 100) + 0.299
    theta_33_t_adj = theta_33_t + (1.283 * theta_33_t ** 2 - 0.374 * theta_33_t - 0.015)

    # 3. Porosity and Saturation
    porosity = 1.0 - (bdod / config.WOFOST_CONFIG['CONSTANTS']['SOIL_PARTICLE_DENSITY'])

    # Lamda (pore size distribution index)
    lamda = (np.log(1500) - np.log(33)) / (np.log(theta_33_t_adj) - np.log(theta_1500_t_adj))

    # Saturated water content (SM0)
    theta_s_adj = theta_33_t_adj + (porosity - theta_33_t_adj)
    sm0_frac_final = theta_s_adj

    # Apply bulk density correction
    smw_frac_final = theta_1500_t_adj * (1 - (bdod - 1.35) * 0.2)
    smfcf_frac_final = theta_33_t_adj * (1 - (bdod - 1.35) * 0.2)

    if DEBUG_LOGGING_COUNT < MAX_DEBUG_ROWS:
        logging.info(
            f"  INTERMEDIATES -> wp_frac_est: {theta_1500_t_adj:.4f}, fc_frac_est: {theta_33_t_adj:.4f}, porosity: {porosity:.4f}")

    # Apply sanity checks and assign to output
    params['SMW'] = np.clip(smw_frac_final, 0.01, 0.9)
    params['SMFCF'] = np.clip(smfcf_frac_final, params['SMW'] + 0.01, 0.9)
    params['SM0'] = np.clip(porosity, params['SMFCF'] + 0.01, 0.95)
    params['CRAIRC'] = max(0.005, params['SM0'] - params['SMFCF'])

    # --- Hydraulic Conductivity (Unchanged) ---
    sand_frac = sand_pct / 100.0
    clay_frac = clay_pct / 100.0
    k0_sand_effect = 200 * np.exp(sand_frac * 0.5)
    k0_clay_effect = np.exp(-clay_frac * 6)
    k0_cm_day = (k0_sand_effect * k0_clay_effect) / 10
    params['K0'] = np.clip(k0_cm_day, 1.0, 300.0)
    params['SOPE'] = params['K0'] * 0.5
    params['KSUB'] = params['K0'] * 0.1

    if DEBUG_LOGGING_COUNT < MAX_DEBUG_ROWS:
        logging.info(
            f"  OUTPUTS -> SMW: {params['SMW']:.4f}, SMFCF: {params['SMFCF']:.4f}, SM0: {params['SM0']:.4f}, CRAIRC: {params['CRAIRC']:.4f}")
        logging.info(f"--- PTF DEBUG END (Row {DEBUG_LOGGING_COUNT}) ---")
        DEBUG_LOGGING_COUNT += 1

    return pd.Series(params)


# ==============================================================================
# === MAIN SCRIPT (Largely unchanged, just calls the new PTF) ==================
# ==============================================================================
def build_static_features_v6():
    """V6: Generates the definitive static parameter file with corrected physics."""
    logging.info("--- Starting V6 Static Parameter Generation (STABLE & CORRECTED) ---")

    # --- 1-8. Data Acquisition and Preparation (Identical to previous script) ---
    try:
        geemap.ee.Initialize(project=config.GEE_PROJECT_ID, opt_url=config.GEE_HIGH_VOLUME_ENDPOINT)
    except Exception as e:
        logging.error(f"FATAL: GEE Init failed: {e}");
        return
    output_filepath = config.STATIC_FEATURES_OUTPUT_PATH
    if output_filepath.exists():
        logging.warning(f"Static features file '{output_filepath}' exists. Deleting and regenerating.")
        output_filepath.unlink()

    gdf_districts = gpd.read_file(config.DISTRICTS_GEOJSON_PATH)
    gdf_districts.rename(columns={'id': 'district_no'}, inplace=True)
    gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
    districts_ee = geemap.geopandas_to_ee(gdf_districts)
    farmland_mask_image = ee.Image(config.FARMLAND_MASK_ASSET_ID)

    dem_image = ee.Image(config.DEM_IMAGE).select('elevation')
    slope_image = ee.Terrain.slope(dem_image)
    image_list = [dem_image.rename('avg_elevation'), slope_image.rename('avg_slope')]
    for prop in config.SOIL_PROPERTIES:
        for depth in config.SOIL_DEPTHS:
            image = ee.Image(f"projects/soilgrids-isric/{prop}_mean").select(f"{prop}_{depth}_mean")
            image_list.append(image.rename(f"{prop}_{depth.replace('-', '_')}"))
    masked_data = ee.Image.cat(image_list).updateMask(farmland_mask_image)

    stats_ee = masked_data.reduceRegions(collection=districts_ee, reducer=ee.Reducer.mean(), scale=250)
    stats_df = geemap.ee_to_df(stats_ee).drop(columns=['geometry', 'system:index'], errors='ignore').dropna(
        subset=['district_no'])

    for depth in config.SOIL_DEPTHS:
        d_str = depth.replace('-', '_')
        if f'bdod_{d_str}' in stats_df.columns: stats_df[f'bdod_{d_str}'] /= 100
        if f'clay_{d_str}' in stats_df.columns: stats_df[f'clay_{d_str}'] /= 10
        if f'sand_{d_str}' in stats_df.columns: stats_df[f'sand_{d_str}'] /= 10
        if f'phh2o_{d_str}' in stats_df.columns: stats_df[f'phh2o_{d_str}'] /= 10
        if f'soc_{d_str}' in stats_df.columns:
            stats_df[f'som_{d_str}'] = (stats_df[f'soc_{d_str}'] / 100) * 1.724
            stats_df.drop(columns=[f'soc_{d_str}'], inplace=True)

    total_thickness_100 = sum(config.LAYER_THICKNESS[d] for d in config.ROOTZONE_LAYERS)
    for prop in ['bdod', 'clay', 'sand', 'som', 'phh2o']:
        weighted_sum = sum(
            stats_df[f'{prop}_{d.replace("-", "_")}'] * config.LAYER_THICKNESS[d] for d in config.ROOTZONE_LAYERS)
        stats_df[f'avg_{prop}_0_100cm'] = weighted_sum / total_thickness_100

    # --- 9. APPLY THE NEW, CORRECTED PTF ---
    logging.info("Applying CORRECTED & STABLE Pedotransfer Functions (PTFs)...")
    wofost_params_df = stats_df.apply(_parameterize_wofost_soil_inputs_v3, axis=1)  # USE THE V3 FUNCTION
    final_df = stats_df.join(wofost_params_df)

    # --- 10 & 11. Finalize and Save ---
    logging.info("Adding documented defaults for parameters not available from GEE...")
    final_df['RDMSOL'] = 200.0;
    final_df['NOTINF'] = 0.1;
    final_df['SSMAX'] = 1.0;
    final_df['DD'] = 500.0

    FINAL_COLS = [
        'district_no', 'avg_elevation', 'avg_slope', 'avg_sand_0_100cm', 'avg_clay_0_100cm',
        'avg_som_0_100cm', 'avg_bdod_0_100cm', 'SMW', 'SMFCF', 'SM0', 'CRAIRC', 'K0',
        'SOPE', 'KSUB', 'RDMSOL', 'NOTINF', 'SSMAX', 'DD'
    ]
    final_df = final_df[FINAL_COLS]
    logging.info(f"Saving final V6 static parameters to '{output_filepath}'")
    final_df.to_csv(output_filepath, index=False, float_format='%.4f')

    logging.info("--- SUCCESS: V6 static parameter file created! ---")
    print("\n--- V6 Static Parameters Preview ---")
    print(final_df.head())


if __name__ == "__main__":
    build_static_features_v6()