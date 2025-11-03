# File: run_detrending.py
# Description: Applies the champion detrending model to each member of the
#              WOFOST forecast ensemble to produce a set of plausible,
#              bias-corrected weather-driven yield components.
#
# REVISED VERSION: This script now operates on the full ensemble output.

import pandas as pd
from pathlib import Path
from pygam import LinearGAM, s
import logging
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MODIFIED: Define input path for the full ensemble ---
INPUT_ENSEMBLE_CSV = Path("data/06_model_output/multi_year_final/forecast_ensemble_2000-2020.csv")
OUTPUT_DIR = Path("data/05_model_input")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# The champion model is GAM, as determined previously.
# We define a function for it directly.
def fit_champion_gam(x, y):
    """Fits the champion GAM model."""
    return LinearGAM(s(0, n_splines=10)).fit(x, y)


def main():
    logging.info(f"--- Starting Ensemble Correction Pipeline ---")
    if not INPUT_ENSEMBLE_CSV.exists():
        logging.error(f"FATAL: Input ensemble file not found at {INPUT_ENSEMBLE_CSV}")
        return

    df_ensemble = pd.read_csv(INPUT_ENSEMBLE_CSV)

    # --- Convert yields to dt/ha for consistency with original detrending ---
    # Assuming dry matter content of 0.25
    dmc = 0.25
    df_ensemble['yield_wlp_dt'] = (df_ensemble['yield_water_limited_dry_kgha'] / dmc) / 100.0
    df_ensemble['yield_pp_dt'] = (df_ensemble['yield_potential_dry_kgha'] / dmc) / 100.0

    districts = df_ensemble['district_no'].unique()
    logging.info(f"Loaded ensemble data for {len(districts)} districts.")

    all_corrected_members = []

    # --- Main loop: Iterate through each district to process its full ensemble ---
    for district in tqdm(districts, desc="Detrending Ensembles by District"):
        district_ensemble_df = df_ensemble[df_ensemble['district_no'] == district]
        members = district_ensemble_df['member'].unique()

        # --- Nested loop: Apply correction to each member ---
        for member in members:
            member_df = district_ensemble_df[district_ensemble_df['member'] == member].sort_values('year')

            x_data = member_df['year'].values
            # We detrend both the water-limited and potential yields
            y_wlp = member_df['yield_wlp_dt'].values
            y_pp = member_df['yield_pp_dt'].values

            if len(x_data) < 3: continue

            # Fit the champion model to this specific member's time series
            gam_wlp = fit_champion_gam(x_data, y_wlp)
            gam_pp = fit_champion_gam(x_data, y_pp)

            # Generate the technological trend for this member
            y_tech_wlp = gam_wlp.predict(x_data)
            y_tech_pp = gam_pp.predict(x_data)

            # Calculate the weather-driven component (the residual)
            y_weather_wlp = y_wlp - y_tech_wlp

            # --- Calculate the water stress component ---
            # This is the weather-driven deviation of the WLP yield from the potential yield
            y_stress = y_wlp - y_pp

            member_results = pd.DataFrame({
                'district_no': district,
                'year': x_data,
                'member': member,
                'wofost_forecast_yield_fresh_dt': y_wlp,  # Original biased forecast
                'wofost_corrected_weather_yield': y_weather_wlp,  # The key output!
                'wofost_water_stress_component': y_stress  # The new powerful feature!
            })
            all_corrected_members.append(member_results)

    if not all_corrected_members:
        logging.error("FATAL: No corrected members were generated.")
        return

    # --- Aggregate and save the final dataset for XGBoost ---
    final_df = pd.concat(all_corrected_members, ignore_index=True)
    output_path = OUTPUT_DIR / 'wofost_corrected_ensemble_features.csv'
    final_df.to_csv(output_path, index=False)

    logging.info("--- Process Complete ---")
    logging.info(f"Full corrected ensemble saved to: {output_path}")


if __name__ == '__main__':
    main()