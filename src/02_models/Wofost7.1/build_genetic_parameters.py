import json
from pathlib import Path
import sys
import logging

# --- (Setup Project Root as before) ---
try:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from src import config
except (ImportError, IndexError):
    print("Failed to import project config.")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG


def get_piecewise_value(year_to_calc, reference_year, base_value, periods):
    """
    Calculates the parameter value for a given year using a
    piecewise-linear (stepped) gain model.
    """
    # Start with the base value at the reference year
    current_value = base_value

    # Sort periods to be safe, from earliest to latest
    periods.sort(key=lambda p: p['until_year'])

    # Determine if we are calculating for the past or future
    if year_to_calc < reference_year:
        # --- CALCULATING FOR THE PAST ---
        # We need to iterate from the reference_year *backwards* to the target_year

        last_period_year = reference_year
        for period in reversed(periods):
            # Find the gain_rate for the *current* iteration's year
            # This period's gain_rate applies from its 'until_year' down to the previous one
            gain_rate = period['gain_rate']

            # The start of this calculation step is the "floor" for this period
            # It's either the target year, or the year the *next* period starts
            start_year = max(year_to_calc, (periods[periods.index(period) - 1]['until_year'] + 1) if periods.index(
                period) > 0 else -9999)

            if last_period_year <= start_year:
                continue  # This period is fully in the future relative to our step

            # Years to apply this *negative* gain_rate over
            num_years = last_period_year - start_year

            # Subtract the gain (e.g., 2017 -> 2000)
            current_value -= (gain_rate * num_years)

            # Set up for the next loop
            last_period_year = start_year
            if last_period_year == year_to_calc:
                break  # We've arrived at our target year

    elif year_to_calc > reference_year:
        # --- CALCULATING FOR THE FUTURE ---
        # We iterate from the reference_year *forwards* to the target_year

        last_period_year = reference_year
        for period in periods:
            # Find the gain_rate for the *current* iteration's year
            gain_rate = period['gain_rate']

            # The end of this calculation step is the "ceiling" for this period
            end_year = min(year_to_calc, period['until_year'])

            if last_period_year >= end_year:
                continue  # This period is fully in the past relative to our step

            # Years to apply this *positive* gain_rate over
            num_years = end_year - last_period_year

            # Add the gain
            current_value += (gain_rate * num_years)

            # Set up for the next loop
            last_period_year = end_year
            if last_period_year == year_to_calc:
                break  # We've arrived at our target year

    # If year_to_calc == reference_year, current_value remains base_value
    return current_value


def main():
    logging.info("--- Building Genetic Gain Factor File (Piecewise-Linear Model) ---")

    try:
        gg_config = CONFIG['GENETIC_GAIN_PARAMS']
        params_to_scale = gg_config['PARAMS_TO_SCALE']
        reference_year = gg_config['REFERENCE_YEAR']
        start_year = gg_config['START_YEAR']
        end_year = CONFIG['END_YEAR']

        logging.info(f"Using REFERENCE_YEAR: {reference_year} (Factor = 1.0)")
        all_years_factors = {}

        for year in range(start_year, end_year + 1):
            year_factors = {}

            for param_name, settings in params_to_scale.items():
                base_value = settings['base']
                periods = settings['periods']

                # Calculate the value for this year using the new piecewise function
                current_value = get_piecewise_value(year, reference_year, base_value, periods)

                # Calculate and store the factor
                factor = current_value / base_value
                factor_name = f"{param_name.upper()}_FACTOR"
                year_factors[factor_name] = factor

            all_years_factors[str(year)] = year_factors

        output_path = config.PROCESSED_DATA_DIR / 'GeneticGainFactors.json'
        with open(output_path, 'w') as f:
            json.dump(all_years_factors, f, indent=4)

        logging.info(f"--- GeneticGainFactors.json saved to {output_path} ---")

        # Log examples
        logging.info(f"Example factors for START YEAR ({start_year}):")
        logging.info(json.dumps(all_years_factors[str(start_year)], indent=4))

        logging.info(f"Example factors for REFERENCE YEAR ({reference_year}):")
        logging.info(json.dumps(all_years_factors[str(reference_year)], indent=4))

        logging.info(f"Example factors for (e.g.) 2005:")
        logging.info(json.dumps(all_years_factors[str(2005)], indent=4))

    except Exception as e:
        logging.error(f"FATAL: An error occurred. Check config 'GENETIC_GAIN_PARAMS'. Error: {e}",
                      exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()