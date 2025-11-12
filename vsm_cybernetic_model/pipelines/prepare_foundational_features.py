# 00_prepare_foundational_features.py
import pandas as pd
import logging

from vsm_cybernetic_model.configs import main_config as cfg

def prepare_all_foundational_features():
    """
    Loads the master dataset and saves it to the intermediate location.

    In this revised workflow, all required features already exist in a single
    master feature file. This script's job is to load that file and save it
    to the primary intermediate location that all downstream expert engines
    will consume.
    """
    print("--- Preparing All Foundational Features from Master File ---")

    master_file_path = cfg.DATA_DIR / "04_master" / "master_dataset.csv"

    try:
        df = pd.read_csv(master_file_path)
        print(f"Loaded master feature file from '{master_file_path}'")
    except FileNotFoundError:
        print(f"Error: Master feature file not found at '{master_file_path}'.")
        print("Please ensure your complete feature set is located at this path.")
        return

    # In this simplified, single-source-of-truth model, we just need to
    # save the main dataframe to the location that all other scripts expect.
    df.to_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN, index=False)

    print(f"Saved foundational features to '{cfg.FOUNDATIONAL_FEATURES_HUMAN}'")
    print("--- Foundational Feature Preparation Complete ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_all_foundational_features()
