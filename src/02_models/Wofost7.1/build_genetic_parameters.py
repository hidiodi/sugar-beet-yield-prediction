# File: src/02_models/Wofost7.1/build_genetic_parameters.py
# Description: Applies temporal genetic gain and creates a master parameter file for each year.

import yaml
import json
from pathlib import Path
import sys
import logging

# --- Setup Project Root ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src import config

# --- Configuration ---
# FIX: Corrected the format string from %(asctime=s to %(asctime)s
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = config.WOFOST_CONFIG

class ParameterDict(dict):
    """A dictionary that allows attribute-style access."""
    def add_variable(self, name, value, description=""):
        self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    def __setattr__(self, name, value):
        self[name] = value

def main():
    """
    Main function to build the genetic parameters file.
    """
    logging.info("--- Building Genetic Parameters (SugarbeetGenes.json) ---")

    # Load base crop parameters from YAML
    try:
        with open(CONFIG['FILE_PATHS']['CROP_YAML'], 'r') as f:
            cp = yaml.safe_load(f)['CropParameters']
        # Merging GenericC3, ecotype, and specific variety parameters
        base_params = {**cp.get('GenericC3', {}), **cp['EcoTypes']['sugarbeet'], **cp['Varieties']['Sugarbeet_601']}

        # Convert list-based values to single values
        cropdata = ParameterDict()
        for key, val in base_params.items():
            if key not in ['Metadata', '<<'] and isinstance(val, list) and len(val) > 0:
                cropdata.add_variable(key, val[0])
            elif key not in ['Metadata', '<<']:
                cropdata.add_variable(key, val)
        logging.info(f"Base parameters loaded. Initial TSUM1: {cropdata.get('TSUM1', 'N/A')}")

    except FileNotFoundError as e:
        logging.error(f"FATAL: Crop YAML file not found. Error: {e}"); sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: Could not parse crop YAML. Error: {e}", exc_info=True); sys.exit(1)

    # Load genetic gain configuration
    genetic_gain_config = CONFIG['GENETIC_GAIN_PARAMS']
    all_years_genes = {}

    # Loop through years and apply genetic gain
    start_year = genetic_gain_config['START_YEAR']
    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        year_params = cropdata.copy()
        years_since_start = year - start_year

        # Calculate new values based on genetic gain
        new_rue = genetic_gain_config['RUE']['base'] + (genetic_gain_config['RUE']['gain_rate'] * years_since_start)
        new_tsum1 = genetic_gain_config['TSUM1']['base'] + (genetic_gain_config['TSUM1']['gain_rate'] * years_since_start)
        new_amax = genetic_gain_config['AMAX']['base'] + (genetic_gain_config['AMAX']['gain_rate'] * years_since_start)

        # Overwrite the base parameters with the new values
        year_params['RUE'] = new_rue
        year_params['TSUM1'] = new_tsum1
        year_params['AMAX'] = new_amax

        all_years_genes[str(year)] = dict(year_params) # Convert ParameterDict to standard dict for JSON

    # Save the final JSON file
    output_path = config.PROCESSED_DATA_DIR / 'SugarbeetGenes.json'
    with open(output_path, 'w') as f:
        json.dump(all_years_genes, f, indent=4)

    logging.info(f"--- SugarbeetGenes.json saved to {output_path} ---")
    logging.info(f"Example parameters for {CONFIG['END_YEAR']}:")
    logging.info(f"  RUE: {all_years_genes[str(CONFIG['END_YEAR'])]['RUE']:.4f}")
    logging.info(f"  TSUM1: {all_years_genes[str(CONFIG['END_YEAR'])]['TSUM1']:.4f}")
    logging.info(f"  AMAX: {all_years_genes[str(CONFIG['END_YEAR'])]['AMAX']:.4f}")


if __name__ == "__main__":
    main()