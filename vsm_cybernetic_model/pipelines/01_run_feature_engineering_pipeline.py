# 01_run_feature_engineering_pipeline.py
import argparse
import pandas as pd
import joblib

from vsm_cybernetic_model.configs import main_config as cfg
from vsm_cybernetic_model.system_1_biophysical.model import (
    run_rpp_simulations,
    train_biophysical_engine
)
from vsm_cybernetic_model.system_2_coordination.model import train_coordination_engine
from vsm_cybernetic_model.system_3_control.model import train_economic_battery_engine
from vsm_cybernetic_model.system_4_strategy.model import train_strategy_engine
from vsm_cybernetic_model.system_5_policy.model import train_policy_engine


def run_training_pipeline():
    """Executes the full training pipeline for all Stage 1 Expert Engines."""
    print("--- Starting Stage 1 Expert Engine Training Pipeline ---")
    # Note: In a real-world scenario, you would run the preparation scripts first.
    # For this blueprint, we assume they have been run and the foundational
    # features exist.

    # VSM 1 depends on VSM 2 for calibration, so we must respect the order.
    run_rpp_simulations.run_rpp_simulations()
    train_biophysical_engine.train_biophysical_engine()
    train_coordination_engine.train_coordination_engine()
    train_economic_battery_engine.train_economic_battery_engine()
    train_strategy_engine.train_strategy_engine()
    train_policy_engine.train_policy_engine()
    print("--- All Expert Engines have been trained successfully. ---")


def _apply_feature_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a one-year lag to post-season features to simulate a forecast scenario.

    For training, we must ensure that the model only sees information that would
    have been available at the time of a pre-season forecast. This function
    lags all features that are only available post-season (e.g., annual
    economic accounts, subsidy payments) by one year.
    """
    print("Applying 1-year lag to post-season features...")

    # These columns are only available after the season is complete
    # and must be lagged for a realistic forecast training set.
    POST_SEASON_FEATURES = [
        'crop_area_variance_nuts3',
        'so_per_ha_n3',
        'total_cap_subsidy_nuts3'
    ]

    # Use a placeholder for features that might not exist in dummy data
    features_to_lag = [f for f in POST_SEASON_FEATURES if f in df.columns]

    # Sort by district and year to ensure correct lagging
    df = df.sort_values(by=['district_no', 'year'])

    # Apply lag within each district group
    df[features_to_lag] = df.groupby('district_no')[features_to_lag].shift(1)

    print(f"Lagged features: {features_to_lag}")
    return df


def run_transformation_pipeline():
    """
    Executes the transformation pipeline to generate the final feature matrix.

    This function loads all pre-trained Stage 1 Expert Engine artifacts and
    uses them to transform the foundational features into the final VSM indices
    that will be fed into the Stage 2 XGBoost regulator.
    """
    print("--- Starting Final Feature Matrix Transformation Pipeline ---")

    # Load all necessary foundational feature sets
    df_human = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN)
    df_rpp = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_RPP)
    # df_static_bio = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_STATIC_BIO)
    df_rpp['Soil_Water_Battery'] = 200 - (df_rpp['district_no'] % 20) * 2.5 # Dummy
    df_bio = df_rpp

    # Merge into a single dataframe for transformation
    df_merged = pd.merge(df_human, df_bio, on=['year', 'district_no'], how='inner')

    # ** NEW STEP: Apply lags to create a realistic forecast training set **
    df_merged = _apply_feature_lags(df_merged)

    final_features = df_merged[['year', 'district_no']].copy()

    # Loop through all VSM systems to transform data
    all_systems = [
        ('vsm1_biophysical', 'VSM1'),
        ('vsm2_coordination', 'VSM2'),
        ('vsm3_control', 'VSM3'),
        ('vsm4_strategy', 'VSM4'),
        ('vsm5_policy', 'VSM5'),
    ]

    for sys_name, sys_prefix in all_systems:
        print(f"Transforming features for {sys_name}...")
        sys_cfg = __import__(f"vsm_cybernetic_model.configs.{sys_name}", fromlist=[''])

        # Load artifacts
        scaler = joblib.load(cfg.STAGE_1_EXPERTS_DIR / getattr(sys_cfg, f"{sys_prefix}_SCALER_NAME"))
        engine = joblib.load(cfg.STAGE_1_EXPERTS_DIR / getattr(sys_cfg, f"{sys_prefix}_ENGINE_NAME"))

        # Select, scale, and transform data
        features = getattr(sys_cfg, f"{sys_prefix}_INPUT_FEATURES")
        X = df_merged[features].dropna() # Important to handle NaNs consistently

        if not X.empty:
            X_scaled = scaler.transform(X)
            X_transformed = engine.transform(X_scaled)

            # Add transformed components as new features
            for i in range(X_transformed.shape[1]):
                final_features.loc[X.index, f'{sys_prefix}_PC{i+1}'] = X_transformed[:, i]

    # Add the target variable (assuming it's in one of the foundational files)
    # In a real scenario, you would load the master dataset.
    # final_features['yield_gap'] = df_merged['real_yield'] - df_merged['RPP_mean_yield']

    # Save final feature matrix
    final_features.to_csv(cfg.FINAL_FEATURES_PATH, index=False)
    print(f"--- Final feature matrix saved to '{cfg.FINAL_FEATURES_PATH}' ---")


def main():
    """Main entry point for the feature engineering pipeline."""
    parser = argparse.ArgumentParser(description="VSM-CPS Feature Engineering Pipeline")
    parser.add_argument(
        'mode',
        choices=['train', 'transform'],
        help="Pipeline mode: 'train' to train expert engines, 'transform' to generate final features."
    )
    args = parser.parse_args()

    if args.mode == 'train':
        run_training_pipeline()
    elif args.mode == 'transform':
        run_transformation_pipeline()

if __name__ == "__main__":
    main()
