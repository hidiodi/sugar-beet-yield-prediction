# File: src/models/run_shap_analysis.py
# DESCRIPTION:
# This script loads the three final quantile models (lower, median, upper)
# and performs a SHAP (SHapley Additive exPlanations) analysis on each.
#
# It generates and saves two plots for each model:
# 1. A global feature importance bar plot.
# 2. A feature distribution/impact beeswarm plot.
#
# These plots help interpret which features are driving the model's predictions
# for the lower, median, and upper bounds of the forecast residuals.

import pandas as pd
import os
import joblib
import warnings
import numpy as np
import shap
import matplotlib.pyplot as plt
import json  # <-- Add import for json

warnings.filterwarnings("ignore")

# --- Configuration (Must match train_final_quantile_model.py) ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
MODEL_OUTPUT_DIR = 'src/models'
SHAP_OUTPUT_DIR = os.path.join('reports', 'shap_analysis')
SHAP_SAMPLE_SIZE = 5000  # Subsample for faster SHAP computation

# --- Feature List (Must match training script) ---
FEATURE_COLS = [
    # --- Original SEAS5 Weather Anomaly Features (Antecedent & Seasonal) ---
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
    'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
    'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
    'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast',

    # --- Original SEAS5 Weather Probability Features ---
    'spring_temp_prob_warm_forecast', 'spring_precip_prob_wet_forecast', 'summer_temp_prob_warm_forecast',
    'summer_precip_prob_wet_forecast',

    # --- Static Geographic & Soil Features ---
    'lat', 'lon', 'avg_elevation', 'avg_slope', 'avg_bdod_0_30cm', 'avg_clay_0_30cm',
    'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm',

    # --- Satellite Features (Early Season Condition) ---
    'winter_cropland_ndvi_mean', 'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean',
    'winter_cropland_LST_anomaly', 'winter_cropland_snow_cover_days',

    # --- Teleconnection Indices ---
    'nao_winter_avg', 'sca_winter_avg', 'enso_mei_winter_avg',

    # --- Lagged Economic Features & Anomalies ---
    'profit_margin_proxy_lag1', 'cost_of_inputs_lag1', 'producer_price_index_lag1_anomaly',
    'seed_price_index_lag1_anomaly', 'energy_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly', 'plant_protection_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme',

    # --- Stage 1 Model & Hybrid Features ---
    'stage1_forecast',
    'wofost_forecast_x_profit_margin',
    'has_wofost_data',

    # --- General Regional & Temporal Features ---
    'state_encoded',
    'year_trend',

    # --- Original Interaction & Polynomial Features ---
    'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip', 'summer_heat_x_profit_margin',
    'summer_precip_x_input_costs', 'hot_dry_interaction', 'lat_x_summer_temp', 'sandy_soil_x_drought',
    'antecedent_gdd_sum_anomaly_sq', 'spring_temp_prob_warm_forecast_sq',
    'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq',
    'summer_precip_prob_wet_forecast_sq', 'summer_precip_anomaly_forecast_sq',

    # --- NEW Physiologically-Grounded Features for Extremes ---
    'CASDI_Phase2_Count',  # Compounded Abiotic Stress (Heat & Drought)
    'NMSD_Phase2_Count',  # Nighttime Metabolic Stress Days
    'OSAW_Phase2_Count',  # Optimal Sugar Accumulation Window
    'ECES_Phase1_Cumulative',  # Early Canopy Establishment Stress
    'summer_days_tmax_gt_30c'
]

# --- MODIFIED: Re-enabled all quantile models ---
# QUANTILES = {'median': 0.5}
QUANTILES = {'lower': 0.025, 'median': 0.5, 'upper': 0.975}


def load_and_prep_data():
    """Loads and prepares the training data exactly as in the training script."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Apply same preprocessing as training script
    df.rename(columns={'wofost_forecast_yield_fresh_dt': 'stage1_forecast'}, inplace=True)
    df['forecast_residual'] = df['kreisYield'] - df['stage1_forecast']
    # Critical: Dropping rows where target is NaN is still required.
    df.dropna(subset=['stage1_forecast', 'forecast_residual'], inplace=True)

    # Per user request, do NOT drop rows with missing features.
    # XGBoost (and thus SHAP) can handle NaNs.
    # df.dropna(subset=FEATURE_COLS, inplace=True) # <-- This line is removed.

    X_train = df[FEATURE_COLS]
    print(f"Data loaded. Found {len(X_train)} samples (NaNs in features are kept).")
    return X_train


def run_analysis():
    """Main function to run the SHAP analysis loop."""
    print("--- Starting SHAP Analysis for Final Quantile Models ---")

    # Ensure output directory exists
    os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
    print(f"SHAP plots will be saved to: {SHAP_OUTPUT_DIR}")

    # Load and prep data
    X_train = load_and_prep_data()

    # Subsample data for SHAP analysis (for performance)
    if len(X_train) > SHAP_SAMPLE_SIZE:
        print(f"Dataset is large ({len(X_train)} samples). Subsampling to {SHAP_SAMPLE_SIZE} for SHAP analysis.")
        X_train_shap = shap.sample(X_train, SHAP_SAMPLE_SIZE, random_state=42)
    else:
        X_train_shap = X_train

    # Loop through each model
    for name in QUANTILES.keys():
        print(f"\n--- Analyzing {name.upper()} Model (Quantile: {QUANTILES[name]}) ---")

        # Load the model
        model_path = os.path.join(MODEL_OUTPUT_DIR, f'final_quantile_model_{name}.joblib')
        if not os.path.exists(model_path):
            print(f"⚠️  Model file not found, skipping: {model_path}")
            continue

        print(f"Loading model: {model_path}")
        model = joblib.load(model_path)

        # --- ROBUST WORKAROUND for base_score string issue ---
        # We must fix the config *inside* the booster object,
        # as this is what SHAP re-parses.
        try:
            booster = model.get_booster()
            config_str = booster.save_config()
            config_json = json.loads(config_str)

            # Navigate to the problematic parameter
            base_score_val = config_json.get('learner', {}).get('learner_model_param', {}).get('base_score')

            if base_score_val and isinstance(base_score_val, str):
                # Parse the string (e.g., '[-2.0966646E2]')
                parsed_score = float(base_score_val.strip('[]'))
                print(f"Warning: Model's internal 'base_score' was a string '{base_score_val}'.")
                print(f"Patching booster config with float: {parsed_score}")

                # Update the JSON dictionary
                config_json['learner']['learner_model_param']['base_score'] = parsed_score

                # Load the patched config (as a string) back into the booster
                booster.load_config(json.dumps(config_json))

                # Also update the python object attribute for good measure
                model.base_score = parsed_score

        except Exception as e:
            # Catch errors during patching, but still try to proceed
            print(f"Error: Could not parse or patch model.base_score. SHAP analysis might fail. Error: {e}")
            # If it failed, try the simple attribute set again just in case
            if isinstance(model.base_score, str):
                try:
                    model.base_score = float(model.base_score.strip('[]'))
                except:
                    pass
        # --- End of workaround ---

        # 1. Initialize Explainer and Calculate SHAP values
        print("Initializing SHAP Explainer...")

        # --- MODIFIED: Use model.predict and background data ---
        # This is the most robust, model-agnostic way to initialize the explainer.
        # It treats the model as a "black box" function and uses the data
        # to learn its behavior, avoiding the previous errors.
        explainer = shap.Explainer(model.predict, X_train_shap)
        # explainer = shap.Explainer(model) # <-- This failed with TypeError

        print(f"Calculating SHAP values for {len(X_train_shap)} samples... (This may take a moment)")
        shap_values = explainer(X_train_shap)
        print("SHAP values calculated.")

        # 2. Generate and Save Global Feature Importance (Bar Plot)
        print("Generating Global Feature Importance (bar) plot...")
        try:
            # Create a new figure
            plt.figure(figsize=(10, 16))
            plt.title(f'SHAP Global Feature Importance - {name.upper()} Model', fontsize=16, pad=20)

            # Create the SHAP plot on the current figure
            shap.summary_plot(
                shap_values,
                X_train_shap,
                plot_type='bar',
                show=False,
                max_display=30  # Show top 30 features
            )

            save_path = os.path.join(SHAP_OUTPUT_DIR, f'shap_global_importance_bar_{name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Use bbox_inches to prevent cutoff
            plt.close()  # Close the figure to free memory
            print(f"✅ Saved bar plot to {save_path}")

        except Exception as e:
            print(f"❌ Failed to generate bar plot for {name}: {e}")
            plt.close()  # Ensure figure is closed on error

        # 3. Generate and Save Feature Distribution (Beeswarm Plot)
        print("Generating Feature Distribution (beeswarm) plot...")
        try:
            # Create a new figure
            plt.figure(figsize=(10, 16))
            plt.title(f'SHAP Feature Distribution - {name.upper()} Model', fontsize=16, pad=20)

            # Create the SHAP plot on the current figure
            shap.summary_plot(
                shap_values,
                X_train_shap,
                plot_type='dot',  # 'dot' is the beeswarm plot
                show=False,
                max_display=30  # Show top 30 features
            )

            save_path = os.path.join(SHAP_OUTPUT_DIR, f'shap_distribution_beeswarm_{name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved beeswarm plot to {save_path}")

        except Exception as e:
            print(f"❌ Failed to generate beeswarm plot for {name}: {e}")
            plt.close()

        print(f"--- Completed analysis for {name.upper()} Model ---")

    print("\n--- SHAP Analysis Complete. All plots saved. ---")


if __name__ == "__main__":
    run_analysis()