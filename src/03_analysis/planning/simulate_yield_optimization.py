# File: src/planning/simulate_yield_optimization.py
# Description: The definitive proof of value. A 40-year simulation comparing a
# model-driven planting strategy vs. a historical-average strategy to quantify the total yield improvement.

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- Key Business Assumption ---
# This is the estimated yield (in dt/ha) required to break even.
# A planner would provide this based on operating costs, market prices, etc.
BREAK_EVEN_YIELD = 450.0  # dt/ha

# --- Configuration ---
MODEL_PATH = os.path.join('src/models', 'final_xgb_model_random_split.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
OUTPUT_DIR = os.path.join('reports', 'value_demonstration')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_optimization_simulation():
    """
    Runs a multi-decade simulation comparing two planting decision strategies.
    """
    print("--- Starting 40-Year Yield Optimization Simulation ---")
    print(f"Decision Threshold (Break-Even Yield): {BREAK_EVEN_YIELD} dt/ha")

    model = joblib.load(MODEL_PATH)
    full_data = pd.read_csv(DATA_PATH)

    simulation_years = sorted(full_data['year'].unique())
    annual_results = []

    # Use tqdm for a progress bar during the long simulation
    for year in tqdm(simulation_years, desc="Simulating Years"):

        # --- Prepare Data for the Current Year ---
        historical_df = full_data[full_data['year'] < year]
        if historical_df.empty: continue  # Cannot make a decision for the first year

        current_year_df = full_data[full_data['year'] == year].copy()
        if current_year_df.empty: continue

        # --- Generate Inputs for Both Strategies ---
        # 1. Historical Average (for the Status Quo strategy)
        historical_baseline = historical_df.groupby('district_no')['kreisYield'].mean().reset_index()
        historical_baseline.rename(columns={'kreisYield': 'historical_avg_yield'}, inplace=True)

        # 2. Model Forecast (for the Model-Driven strategy)
        feature_cols = model.get_booster().feature_names
        current_year_df['model_forecast'] = model.predict(current_year_df[feature_cols])

        # --- Make Planting Decisions for Each District ---
        decisions_df = pd.merge(current_year_df, historical_baseline, on='district_no', how='left')

        # Strategy A: Plant if the historical average is above break-even
        decisions_df['plant_historical'] = decisions_df['historical_avg_yield'] > BREAK_EVEN_YIELD

        # Strategy B: Plant if the model's forecast for THIS YEAR is above break-even
        decisions_df['plant_model'] = decisions_df['model_forecast'] > BREAK_EVEN_YIELD

        # --- Calculate the Outcome for Each Strategy ---
        # The yield is only collected if the decision was to plant
        yield_historical = decisions_df[decisions_df['plant_historical']]['kreisYield'].sum()
        yield_model = decisions_df[decisions_df['plant_model']]['kreisYield'].sum()

        annual_results.append({
            'year': year,
            'yield_historical': yield_historical,
            'yield_model': yield_model,
            'districts_planted_historical': decisions_df['plant_historical'].sum(),
            'districts_planted_model': decisions_df['plant_model'].sum(),
        })

    # --- Aggregate and Analyze the 40-Year Results ---
    results_df = pd.DataFrame(annual_results)
    results_df['cumulative_yield_historical'] = results_df['yield_historical'].cumsum()
    results_df['cumulative_yield_model'] = results_df['yield_model'].cumsum()

    total_yield_historical = results_df['cumulative_yield_historical'].iloc[-1]
    total_yield_model = results_df['cumulative_yield_model'].iloc[-1]
    total_yield_gain = total_yield_model - total_yield_historical
    percent_improvement = (total_yield_gain / total_yield_historical) * 100

    print("\n--- 40-Year Simulation Results ---")
    print(f"Total Yield (Historical Strategy): {total_yield_historical / 1000000:.2f} million dt")
    print(f"Total Yield (Model-Driven Strategy): {total_yield_model / 1000000:.2f} million dt")
    print("--------------------------------------------------")
    print(f"Total Yield Gained by Using the Model: {total_yield_gain / 1000000:.2f} million dt")
    print(f"Overall Improvement: +{percent_improvement:.2f}%")

    # --- Create the "Ultimate Proof" Visualization ---
    print("\nGenerating final proof-of-value visualization...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(results_df['year'], results_df['cumulative_yield_model'] / 1000000,
            label='Model-Driven Strategy', color='#2ca02c', linewidth=2.5)
    ax.plot(results_df['year'], results_df['cumulative_yield_historical'] / 1000000,
            label='Historical Average Strategy', color='#d62728', linestyle='--', linewidth=2)

    # Add fill between the lines to emphasize the gain
    ax.fill_between(results_df['year'],
                    results_df['cumulative_yield_model'] / 1000000,
                    results_df['cumulative_yield_historical'] / 1000000,
                    color='#2ca02c', alpha=0.2, label='Cumulative Yield Gain')

    # Add annotation for the final result
    final_year = results_df['year'].iloc[-1]
    final_gain_y = results_df['cumulative_yield_model'].iloc[-1] / 1000000
    ax.annotate(f'Total Gain:\n+{total_yield_gain / 1000000:.2f}M dt\n(+{percent_improvement:.2f}%)',
                xy=(final_year, final_gain_y),
                xytext=(final_year - 15, final_gain_y * 0.7),
                fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8))

    ax.set_title('The 40-Year Simulation of Planting Strategies', fontsize=18, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cumulative Total Yield (in Million dt)', fontsize=12)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    output_path = os.path.join(OUTPUT_DIR, 'simulation.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ 'Ultimate Proof' chart saved to {output_path}")


if __name__ == "__main__":
    run_optimization_simulation()