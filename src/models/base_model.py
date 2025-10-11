import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge  # Regularized Linear Model to handle VIF
from xgboost import XGBRegressor  # High-performance Gradient Boosting
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
import joblib  # <--- ADDED IMPORT for saving/loading models

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the path where the model will be saved
MODEL_PATH = os.path.join('src/models', 'stage1_xgb_model.joblib')


def load_and_prepare_data():
    """Loads the final dataset, defines features, and performs data split and scaling."""

    file_path = os.path.join('data', '05_model_input', 'final_imputed_dataset.csv')

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(
            f"Error: Dataset not found at {file_path}. Please check the path and ensure the feature engineering script has run successfully.")
        return None, None, None, None, None, None, None

    # Final predictor features (9 features)
    feature_cols = [
        'district_no',
        'kreisField_ha',
        'producer_price_index',
        'precip_total_peak_growth',
        'heat_stress_days_peak_growth',
        'avg_elevation',
        'winter_temp_anomaly',
        'winter_precip_anomaly',
        'yield_density'
    ]
    target_col = 'kreisYield'

    X = df[feature_cols]
    y = df[target_col]

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing (Scaling for Ridge Regression)
    # Fit scaler only on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrame for feature naming (needed for Ridge/XGBoost interpretation)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_cols


def run_advanced_baselines(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_cols):
    """Trains and evaluates advanced baseline models (Ridge Regression and XGBoost)."""

    print("\n--- Starting Advanced Baseline Model Training ---")
    results = {}

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # --- MODEL 1: RIDGE REGRESSION (Regularized Linear Model) ---
    print("Training Ridge Regression (Alpha=1.0) to handle VIF...")
    # Ridge is used to stabilize coefficients of high-VIF features (like producer_price_index)
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)

    results['RidgeRegression'] = {
        'R2': r2_score(y_test, y_pred_ridge),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'Coefficients': dict(zip(feature_cols, ridge.coef_))
    }

    # --- MODEL 2: XGBOOST REGRESSOR (High-Performance Ensemble) ---
    print("Training XGBoost Regressor...")
    # XGBoost is a powerful, non-linear model that should maximize R2
    xgb = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )
    # XGBoost handles non-scaled data well
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    results['XGBoost'] = {
        'R2': r2_score(y_test, y_pred_xgb),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'Feature_Importances': dict(zip(feature_cols, xgb.feature_importances_))
    }

    # --- MODEL SAVING CODE ---
    try:
        joblib.dump(xgb, MODEL_PATH)
        print(f"\n✅ XGBoost model successfully saved to {MODEL_PATH}")
    except Exception as e:
        print(f"\n❌ Warning: Could not save the XGBoost model. Error: {e}")

    # --- Print and Log Results ---
    print("\n--- Advanced Baseline Model Results ---")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"  R-squared (R2): {metrics['R2']:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f} dt/ha")

        if 'Coefficients' in metrics:
            print("\n  Top 5 Coefficients (Standardized):")
            sorted_coeffs = sorted(metrics['Coefficients'].items(), key=lambda item: abs(item[1]), reverse=True)
            for feature, coeff in sorted_coeffs[:5]:
                print(f"    {feature}: {coeff:+.4f}")

        if 'Feature_Importances' in metrics:
            print("\n  Top 5 Feature Importances:")
            sorted_importances = sorted(metrics['Feature_Importances'].items(), key=lambda item: item[1], reverse=True)
            for feature, importance in sorted_importances[:5]:
                print(f"    {feature}: {importance:.4f}")

    print("\n--- Advanced Baseline Analysis Complete ---")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_cols = load_and_prepare_data()

    if X_train is not None:
        run_advanced_baselines(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_cols)