# File: run_forecasting_pipeline_walkforward.py
# Description: A definitive, consolidated script to perform a leak-free forecast.
#
# REVISED VERSION v4: Adds a detailed year-by-year performance analysis and visualization
# to better understand the model's behavior over time.

import pandas as pd
import os
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from pygam import LinearGAM, s
from tqdm import tqdm
import warnings

# --- Configuration & Setup ---
warnings.filterwarnings("ignore")

# ==============================================================================
# === G L O B A L   C O N F I G U R A T I O N ===
# ==============================================================================
CONFIG = {
    'FILE_PATHS': {
        'INPUT_YIELD_CSV': 'data/02_intermediate/sugarbeet_yield.csv',
        'OUTPUT_DIR': 'data/09_model_output_walkforward_final',
    },
    'ARIMA_ORDER': (1, 0, 0),
    'GAM_SPLINES': 10,
    'MIN_TRAIN_SIZE': 10
}

# ==============================================================================
# === S C R I P T   S T A R T S   H E R E ===
# ==============================================================================

# --- Setup logging ---
logging.getLogger().handlers = []
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)


def run_full_walk_forward_pipeline(df_yield):
    """
    Executes the entire walk-forward forecasting pipeline for all districts.
    """
    all_forecasts = []
    districts = df_yield['district_no'].unique()
    min_train_size = CONFIG['MIN_TRAIN_SIZE']

    logging.info(f"Starting unified walk-forward validation for {len(districts)} districts...")

    for district in tqdm(districts, desc="Forecasting Districts"):
        district_df = df_yield[df_yield['district_no'] == district].sort_values('year')

        if len(district_df) < min_train_size:
            continue

        for i in range(min_train_size, len(district_df)):
            train_df = district_df.iloc[:i]
            x_train, y_train = train_df['year'].values, train_df['yield'].values
            current_year, actual_yield = district_df.iloc[i]['year'], district_df.iloc[i]['yield']

            try:
                trend_model = LinearGAM(s(0, n_splines=CONFIG['GAM_SPLINES'])).fit(x_train, y_train)
                base_trend_forecast = trend_model.predict([current_year])[0]

                historical_residuals = y_train - trend_model.predict(x_train)
                residual_model = ARIMA(historical_residuals, order=CONFIG['ARIMA_ORDER']).fit()
                residual_forecast = residual_model.forecast(steps=1)[0]

                if not (np.isfinite(base_trend_forecast) and np.isfinite(residual_forecast)):
                    raise ValueError("A forecast component was non-finite.")

                final_forecast = base_trend_forecast + residual_forecast

                all_forecasts.append({
                    'district_no': district, 'year': current_year, 'actual_yield': actual_yield,
                    'base_trend_forecast': base_trend_forecast, 'final_corrected_forecast': final_forecast
                })
            except Exception as e:
                logging.debug(f"Could not forecast for {district} in {current_year}. Reason: {e}")
                continue

    return pd.DataFrame(all_forecasts)


def analyze_performance_by_year(df, output_dir):
    """
    Calculates and visualizes model performance for each year.
    """
    logging.info("--- Yearly Performance Analysis ---")

    # Calculate errors for aggregation
    df['base_ae'] = (df['actual_yield'] - df['base_trend_forecast']).abs()
    df['corrected_ae'] = (df['actual_yield'] - df['final_corrected_forecast']).abs()

    # Calculate yearly MAE
    yearly_mae = df.groupby('year').agg(
        base_mae=('base_ae', 'mean'),
        corrected_mae=('corrected_ae', 'mean')
    ).reset_index()

    # Calculate yearly R-squared
    yearly_r2 = []
    for year, group in df.groupby('year'):
        if len(group) > 1:
            base_r2 = r2_score(group['actual_yield'], group['base_trend_forecast'])
            corrected_r2 = r2_score(group['actual_yield'], group['final_corrected_forecast'])
            yearly_r2.append({'year': year, 'base_r2': base_r2, 'corrected_r2': corrected_r2})

    yearly_r2_df = pd.DataFrame(yearly_r2)

    yearly_performance = pd.merge(yearly_mae, yearly_r2_df, on='year')

    print("Yearly Performance Metrics:")
    print(yearly_performance.round(3).to_string())

    # --- Plotting Yearly MAE ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 8))

    bar_width = 0.4
    index = np.arange(len(yearly_performance['year']))

    ax.bar(index - bar_width / 2, yearly_performance['base_mae'], bar_width, label='Base Trend MAE', color='orange',
           alpha=0.8)
    ax.bar(index + bar_width / 2, yearly_performance['corrected_mae'], bar_width, label='Corrected MAE', color='purple',
           alpha=0.8)

    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Mean Absolute Error (dt/ha)', fontsize=14)
    ax.set_title('Year-by-Year Forecast Error Comparison', fontsize=18)
    ax.set_xticks(index)
    ax.set_xticklabels(yearly_performance['year'], rotation=45)
    ax.legend(fontsize=12)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'performance_over_time.png')
    plt.savefig(plot_path, dpi=300)
    logging.info(f"✓ Yearly performance plot saved to {plot_path}")
    plt.show()


def analyze_and_plot_results(df, output_dir):
    """
    Analyzes and plots the overall performance of the forecasting pipeline.
    """
    if df.empty:
        logging.error("No valid forecasts were generated. Cannot analyze results.")
        return

    logging.info("--- Overall Performance Analysis ---")

    df['base_trend_forecast'] = df['base_trend_forecast'].clip(lower=0)
    df['final_corrected_forecast'] = df['final_corrected_forecast'].clip(lower=0)

    scenarios = {
        'Base Trend Forecast': ('actual_yield', 'base_trend_forecast'),
        'Trend + ARIMA Correction': ('actual_yield', 'final_corrected_forecast')
    }

    print("\n" + "=" * 55)
    print("--- Model Performance (Walk-Forward Validation) ---")
    print("=" * 55)
    for name, (actual_col, pred_col) in scenarios.items():
        mae = mean_absolute_error(df[actual_col], df[pred_col])
        r2 = r2_score(df[actual_col], df[pred_col])
        print(f"  {name:<25}: MAE = {mae:5.2f}, R² = {r2:6.3f}")
    print("=" * 55 + "\n")

    # --- Overall Scatter Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle('Honest Forecast Performance (Unified Walk-Forward Validation)', fontsize=16)
    min_val, max_val = df['actual_yield'].min() * 0.9, df['actual_yield'].max() * 1.1

    axes[0].scatter(df['actual_yield'], df['base_trend_forecast'], alpha=0.5, color='orange')
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    axes[0].set_title('Base Trend Forecast (Out-of-Sample)', fontsize=14)
    axes[0].set_xlabel('Actual Yield (dt/ha)');
    axes[0].set_ylabel('Predicted Yield (dt/ha)')

    axes[1].scatter(df['actual_yield'], df['final_corrected_forecast'], alpha=0.5, color='purple')
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    axes[1].set_title('Trend + ARIMA Correction (Out-of-Sample)', fontsize=14)
    axes[1].set_xlabel('Actual Yield (dt/ha)')

    for ax in axes.flatten():
        ax.set_xlim(min_val, max_val);
        ax.set_ylim(min_val, max_val)
        ax.grid(True, alpha=0.3);
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(output_dir, 'final_walkforward_performance.png')
    plt.savefig(plot_path, dpi=300)
    logging.info(f"✓ Overall scatter plot saved to {plot_path}")
    plt.show()

    # --- Run the new yearly analysis ---
    analyze_performance_by_year(df, output_dir)


def main():
    """Main function to orchestrate the forecasting process."""
    output_dir = CONFIG['FILE_PATHS']['OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)

    logging.info("--- Step 1: Loading input data ---")
    try:
        df_yield = pd.read_csv(CONFIG['FILE_PATHS']['INPUT_YIELD_CSV'])
    except FileNotFoundError as e:
        logging.error(f"FATAL: Input yield file not found. Error: {e}")
        sys.exit(1)

    df_yield['district_no'] = df_yield['district_no'].astype(str).str.zfill(5)

    logging.info("--- Step 2: Running full walk-forward validation pipeline ---")
    df_forecasts = run_full_walk_forward_pipeline(df_yield)

    logging.info("--- Step 3: Saving final, honest forecasts to CSV ---")
    output_path = os.path.join(output_dir, 'final_honest_forecasts.csv')
    df_forecasts.to_csv(output_path, index=False)
    logging.info(f"✓ Forecast data saved to {output_path}")

    logging.info("--- Step 4: Analyzing and plotting final results ---")
    analyze_and_plot_results(df_forecasts, output_dir)

    logging.info("\n" + "=" * 70 + "\n✓ HONEST FORECASTING PIPELINE COMPLETED SUCCESSFULLY!\n" + "=" * 70)


if __name__ == "__main__":
    main()