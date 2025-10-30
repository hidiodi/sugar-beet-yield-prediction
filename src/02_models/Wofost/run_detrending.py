import pandas as pd
import numpy as np
from pathlib import Path
from pygam import LinearGAM, s
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import r2_score
import warnings
import logging

# --- Configuration & Setup ---

# Configure logging to see progress and key decisions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress common warnings from scipy and pygam for a cleaner output
warnings.filterwarnings('ignore', category=OptimizeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Define file paths using pathlib for cross-platform compatibility
INPUT_YIELD_CSV = Path("data/02_intermediate/sugarbeet_yield.csv")
OUTPUT_DIR = Path("data/02_intermediate/detrended_yield")

# Create the output directory; exist_ok=True prevents an error if it already exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.info(f"Output directory is set to: {OUTPUT_DIR}")

# Define the analysis time range and the realism check buffer
START_YEAR = 1981
END_YEAR = 2024
ACCELERATION_BUFFER = 1.5


# --- Model Definitions ---

# 1. Logistic Model
def logistic_func(x, L, k, x0):
    """The 3-parameter logistic function formula."""
    return L / (1 + np.exp(-k * (x - x0)))


def fit_logistic(x, y):
    """
    Fits a 3-parameter logistic model using scipy.optimize.curve_fit.
    Returns a callable model function and the number of parameters (3).
    Returns None, None if the model fails to converge.
    """
    try:
        # Provide reasonable initial guesses [L, k, x0] and bounds
        p0 = [np.max(y) * 1.1, 0.1, np.median(x)]
        bounds = ([np.min(y), 0, np.min(x)], [np.max(y) * 2.5, 1.0, np.max(x)])
        params, _ = curve_fit(logistic_func, x, y, p0=p0, bounds=bounds, maxfev=10000)

        # Return a lambda function that is "closed over" the fitted parameters
        model = lambda x_new: logistic_func(x_new, *params)
        return model, 3
    except (RuntimeError, ValueError):
        return None, None  # Indicates convergence failure


# 2. Linear Model
def fit_linear(x, y):
    """Fits a linear model using numpy.polyfit. Returns a callable model and num_params."""
    coeffs = np.polyfit(x, y, 1)
    model = np.poly1d(coeffs)
    return model, 2


# 3. Quadratic Model
def fit_quadratic(x, y):
    """Fits a quadratic model using numpy.polyfit. Returns a callable model and num_params."""
    coeffs = np.polyfit(x, y, 2)
    model = np.poly1d(coeffs)
    return model, 3


# 4. GAM (Generalized Additive Model)
def fit_gam(x, y):
    """Fits a GAM using pygam. Returns a callable model and effective degrees of freedom."""
    # Using 10 splines is a reasonable default for ~40 years of data
    model = LinearGAM(s(0, n_splines=10)).fit(x, y)
    # The predict function of a fitted GAM is the callable we need
    return model.predict, int(model.statistics_['edof'])


# --- Helper Functions for Validation and Metrics ---

def calculate_metrics(y_true, y_pred, n, k):
    """
    Calculates R-squared, AIC, and BIC for a given model fit.
    n = number of data points, k = number of model parameters.
    """
    if k <= 0 or n <= k:
        return -np.inf, np.inf, np.inf

    r2 = r2_score(y_true, y_pred)
    rss = np.sum((y_true - y_pred) ** 2)

    # Add a small epsilon to RSS to prevent log(0) errors if the fit is perfect
    if rss < 1e-9:
        rss = 1e-9

    # Using the AIC formula based on RSS for models with normally distributed errors
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + np.log(n) * k

    return r2, aic, bic


def apply_rejection_protocol(model, historical_years, start_year, end_year):
    """
    Applies the 3-part rejection protocol to test a model's extrapolation realism.
    Returns a status string: "PASSED" or "REJECTED (...)".
    """
    y_tech_hist = model(historical_years)
    y_tech_future = model(np.array([end_year + 1]))[0]

    # Rule 1: Unrealistic Acceleration
    annual_gains = np.diff(y_tech_hist)
    # Check if there are any positive gains to prevent errors with flat/declining trends
    if len(annual_gains) > 0 and np.any(annual_gains > 0):
        max_hist_gain = np.max(annual_gains)
        future_gain = y_tech_future - y_tech_hist[-1]

        # Only apply if future gain is positive
        if future_gain > 0 and max_hist_gain > 0:
            if future_gain > (max_hist_gain * ACCELERATION_BUFFER):
                return "REJECTED (Acceleration)"

    # Rule 2: Non-Reversing Trend
    if y_tech_future < y_tech_hist[-1]:
        return "REJECTED (Reversing)"

    # Rule 3: Plausible Floor
    if y_tech_future < y_tech_hist[0]:
        return "REJECTED (Floor)"

    return "PASSED"


# --- Main Orchestration Function ---

def main():
    """
    Executes the entire workflow: data loading, model competition, champion selection,
    and final data generation.
    """
    logging.info("--- Step 0: Loading and Preparing Data ---")
    if not INPUT_YIELD_CSV.exists():
        logging.error(f"FATAL: Input file not found at {INPUT_YIELD_CSV}")
        return

    df = pd.read_csv(INPUT_YIELD_CSV)
    df_filtered = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)].copy()

    if df_filtered.empty:
        logging.error(f"FATAL: No data found in the specified year range ({START_YEAR}-{END_YEAR}).")
        return

    districts = df_filtered['district_no'].unique()
    logging.info(f"Loaded {len(df_filtered)} records for {len(districts)} districts.")

    # --- Steps 1 & 2: Model Competition and Validation ---
    logging.info("--- Steps 1 & 2: Running Global Model Competition & Validation ---")

    competition_results = []
    models_to_compete = {
        'Linear': fit_linear,
        'Quadratic': fit_quadratic,
        'GAM': fit_gam,
        'Logistic': fit_logistic
    }

    for i, district in enumerate(districts):
        # Provide progress update
        if (i + 1) % 50 == 0:
            logging.info(f"Competing models on district {i + 1}/{len(districts)}...")

        district_df = df_filtered[df_filtered['district_no'] == district].sort_values('year')
        x_train = district_df['year'].values
        y_train = district_df['yield'].values

        if len(y_train) < 5: continue  # Skip districts with insufficient data for fitting

        for name, fit_func in models_to_compete.items():
            model_status = "PASSED"
            model, num_params = fit_func(x_train, y_train)

            # Handle Logistic model's specific fallback logic
            if name == 'Logistic' and model is None:
                model, num_params = fit_linear(x_train, y_train)
                model_status = "PASSED (Fallback to Linear)"

            # Apply rejection protocol
            status = apply_rejection_protocol(model, x_train, START_YEAR, END_YEAR)

            # Calculate metrics only for models that pass the realism check
            r2, aic, bic = -1, np.inf, np.inf
            if "REJECTED" not in status:
                y_pred = model(x_train)
                r2, aic, bic = calculate_metrics(y_train, y_pred, len(y_train), num_params)

            competition_results.append({
                'model': name,
                'status': status,
                'R2': r2, 'AIC': aic, 'BIC': bic
            })

    logging.info("Model competition finished for all districts.")

    # --- Champion Selection ---
    results_df = pd.DataFrame(competition_results)
    passed_df = results_df[~results_df['status'].str.contains("REJECTED")].copy()

    # Calculate summary statistics for non-rejected models
    summary = passed_df.groupby('model')[['AIC', 'BIC', 'R2']].mean().sort_values('AIC')

    # Calculate rejection rates for context
    total_attempts = results_df.groupby('model').size()
    rejection_counts = results_df[results_df['status'].str.contains("REJECTED")]['model'].value_counts()
    summary['rejection_rate_%'] = (rejection_counts / total_attempts * 100).fillna(0).round(2)

    if summary.empty:
        logging.error(
            "FATAL: All models were rejected for all districts. Cannot select a champion. Check rejection protocol logic.")
        return

    champion_model_name = summary.index[0]

    logging.info("--- Global Champion Selection Summary ---")
    print(summary)
    logging.info(f"CHAMPION MODEL SELECTED: '{champion_model_name}' (based on the lowest average AIC).")

    # --- Step 3: Data Generation using the Champion Model ---
    logging.info(f"--- Step 3: Generating Output Files using Champion Model: '{champion_model_name}' ---")

    champion_fit_func = models_to_compete[champion_model_name]

    all_tech_yields = []
    all_weather_yields = []

    for i, district in enumerate(districts):
        district_df = df_filtered[df_filtered['district_no'] == district].sort_values('year')
        x_data = district_df['year'].values
        y_observed = district_df['yield'].values

        # Fit the champion model to this district's data
        model, _ = champion_fit_func(x_data, y_observed)

        # Final safety check: if the champion (Logistic) fails, use Linear
        if model is None:
            model, _ = fit_linear(x_data, y_observed)

        # Generate trend and weather components
        y_tech = model(x_data)
        y_weather = y_observed - y_tech

        all_tech_yields.append(pd.DataFrame({'district_no': district, 'year': x_data, 'y_tech': y_tech}))
        all_weather_yields.append(pd.DataFrame({'district_no': district, 'year': x_data, 'y_weather': y_weather}))

    # Combine results and save to CSV
    final_tech_df = pd.concat(all_tech_yields, ignore_index=True)
    final_weather_df = pd.concat(all_weather_yields, ignore_index=True)

    tech_output_path = OUTPUT_DIR / 'Y_tech.csv'
    weather_output_path = OUTPUT_DIR / 'Y_weather.csv'

    final_tech_df.to_csv(tech_output_path, index=False)
    final_weather_df.to_csv(weather_output_path, index=False)

    logging.info("--- Process Complete ---")
    logging.info(f"Output file 1 (Trend):   {tech_output_path}")
    logging.info(f"Output file 2 (Weather): {weather_output_path}")


# --- Script Entry Point ---
if __name__ == '__main__':
    main()