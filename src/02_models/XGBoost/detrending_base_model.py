# File: src/models/base_model.py
# Description: MODIFIED to use a time-based split for validation, holding out the last 5 years of data.
# *** FINAL RECOMMENDED VERSION: Combines detrending, an explicit trend feature, AND a regime-switching flag. ***

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion.joblib')
IMPORTANCE_PLOT_PATH = os.path.join('reports/figures', 'feature_importance.png')


def train_and_validate_with_holdout():
    """
    Loads pre-season data, DETRENDS the target variable, splits the data,
    trains a sophisticated XGBoost model, evaluates it, and retrains the final model.
    """
    file_path = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}. Please run the feature engineering script.")
        return

    # --- FINAL HYBRID Feature Set ---
    # Combines the explicit trend feature with the regime-switching flag
    # to give the model maximum flexibility and explanatory power.
    feature_cols = [
        # Static & Geographic Features
        'avg_elevation',
        'avg_soil_pawc',
        'lon',
        'lat',

        # Antecedent (Winter) Weather Features
        'winter_temp_anomaly',
        'winter_precip_anomaly',

        # Lagged Economic & Yield Features
        'national_avg_yield_lag1',
        'producer_price_index_lag1',

        # Satellite Features
        'winter_cropland_ndvi_mean',
        'winter_cropland_ndvi_anomaly',
        'winter_cropland_LST_mean',
        'winter_cropland_LST_anomaly',
        'winter_cropland_snow_cover_days',

        # --- COMBINED Evolutionary Features ---
        'year_trend',  # Models the continuous technological/genetic gain
        'has_satellite_data',  # Models the structural break between eras
        'post_quota_era'  # Models the 2017 economic shock
    ]
    target_col = 'kreisYield'

    # --- Data Integrity Check ---
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from the input file: {missing_cols}")
        return

    # ==============================================================================
    # === DETRENDING STAGE ===
    # ==============================================================================
    print("\n--- Applying Detrending to Target Variable ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')[target_col].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(method='ffill'))
    detrended_target_col = 'kreisYield_detrended'
    df[detrended_target_col] = df[target_col] - df['yield_trend']
    print(" -> Detrending complete. Model will now predict the anomaly from the trend.")

    # --- Time-Based Split ---
    last_year = df['year'].max()
    validation_start_year = last_year - 5

    print(f"\n--- Using Last 5 Years for Validation ---")
    print(f"Training data will be from years before {validation_start_year + 1}")
    print(f"Validation data will be from years {validation_start_year + 1} to {last_year}")

    train_df = df[df['year'] <= validation_start_year].copy()
    validation_df = df[df['year'] > validation_start_year].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[detrended_target_col]
    X_validation = validation_df[feature_cols]
    y_validation_actual = validation_df[target_col]

    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_validation)} samples")

    # --- Train and Evaluate the XGBoost Model ---
    xgb = XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.03,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    y_pred_detrended = xgb.predict(X_validation)
    y_pred_final = y_pred_detrended + validation_df['yield_trend']

    # --- Evaluate on the original, non-detrended scale ---
    r2 = r2_score(y_validation_actual, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_validation_actual, y_pred_final))

    print("\n--- Validation Performance ---")
    print(f"  R-squared (R2): {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} dt/ha")
    print("-------------------------------------------------")

    # --- Plot and Save Feature Importance ---
    try:
        importance_scores = xgb.feature_importances_
        feature_importance = sorted(zip(feature_cols, importance_scores), key=lambda x: x[1], reverse=False)
        features, scores = zip(*feature_importance)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(features, scores)
        ax.set_title('Feature Importance (Final Hybrid Model)')
        ax.set_xlabel('Feature Importance Score')
        ax.set_ylabel('Features')
        plt.tight_layout()

        os.makedirs(os.path.dirname(IMPORTANCE_PLOT_PATH), exist_ok=True)
        plt.savefig(IMPORTANCE_PLOT_PATH, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to {IMPORTANCE_PLOT_PATH}")
    except Exception as e:
        print(f"❌ Warning: Could not save the feature importance plot. Error: {e}")

    # --- Final Model Training ---
    print("\n--- Training Final Model on Data Before the Holdout Period for Deployment ---")
    final_model = XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.03,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    final_model.fit(df[df['year'] <= validation_start_year][feature_cols],
                    df[df['year'] <= validation_start_year][detrended_target_col])

    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(final_model, MODEL_PATH)
        print(
            f"\n✅ Final XGBoost model successfully trained on data up to {validation_start_year} and saved to {MODEL_PATH}")
    except Exception as e:
        print(f"\n❌ Warning: Could not save the final model. Error: {e}")


if __name__ == "__main__":
    train_and_validate_with_holdout()