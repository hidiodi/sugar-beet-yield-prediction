import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import shap

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_final_stage1_features():
    """
    Performs an advanced analysis of the final Stage 1 training data using SHAP.
    """
    logging.info("--- Starting Advanced Stage 1 Feature Analysis (with SHAP) ---")

    # --- Define Paths ---
    input_file = Path('data/05_model_input/stage1_final_training_data.csv')
    reports_path = Path('reports/stage1_final_analysis/')
    figures_path = reports_path / 'figures'
    reports_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_file)
        logging.info(f"Successfully loaded dataset from '{input_file}'. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"FATAL: Input file not found at '{input_file}'.")
        return

    # --- 1. Correlation of Key Drivers (Weather & Forecasts) ---
    logging.info("\nAnalyzing correlations of primary driver features...")
    driver_cols = [
        'yield', 'avg_elevation', 'avg_soil_pawc',
        'winter_temp_anomaly', 'winter_precip_anomaly',
        'forecasted_temp_anomaly', 'forecasted_precip_anomaly'
    ]
    # Ensure all driver columns exist before trying to correlate them
    driver_cols_exist = [col for col in driver_cols if col in df.columns]

    if len(driver_cols_exist) > 1:
        df_driver_corr = df[driver_cols_exist].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_driver_corr, cmap='coolwarm', annot=True, fmt=".2f")
        plt.title('Correlation Heatmap of Primary Drivers and Yield')
        fig_path = figures_path / 'primary_drivers_heatmap.png'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        logging.info(f" -> Saved primary drivers heatmap to '{fig_path}'")

    # --- 2. Visualize Forecast Impact ---
    logging.info("Visualizing impact of seasonal forecast anomalies on yield...")
    for col in ['forecasted_temp_anomaly', 'forecasted_precip_anomaly']:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.regplot(data=df, x=col, y='yield', line_kws={"color": "red"})
            plt.title(f'Impact of {col.replace("_", " ").title()} on Yield')
            plt.xlabel(f'{col.replace("_", " ").title()}')
            plt.ylabel('Yield')
            plt.grid(True)
            fig_path = figures_path / f'{col}_vs_yield.png'
            plt.savefig(fig_path)
            plt.close()
            logging.info(f" -> Saved {col} plot to '{fig_path}'")

    # --- 3. Advanced Feature Importance with SHAP ---
    logging.info("\nCalculating advanced feature importance using SHAP...")

    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        logging.error("Scikit-learn is not installed. Please run 'pip install scikit-learn'.")
        return

    # Prepare data for the model
    df_model = df.drop(columns=['district_no', 'year']).dropna()

    if df_model.empty or 'yield' not in df_model.columns:
        logging.warning("Dataset is empty or 'yield' column is missing. Cannot calculate SHAP values.")
        return

    X = df_model.drop(columns=['yield'])
    y = df_model['yield']

    # Train a Random Forest model
    logging.info("Training a Random Forest model to explain...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Calculate SHAP values
    logging.info("Calculating SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- Generate SHAP Summary Plot (Beeswarm) ---
    logging.info("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Mean Absolute SHAP Value)')
    fig_path = figures_path / 'shap_feature_importance_bar.png'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    # Beeswarm plot shows feature impact
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Summary Plot (Impact of Features on Yield Prediction)')
    fig_path = figures_path / 'shap_summary_plot_beeswarm.png'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    logging.info(f" -> SHAP plots saved to '{figures_path}'")

    # --- Save SHAP Importance Report ---
    # Calculate mean absolute SHAP value for each feature
    shap_importance = pd.DataFrame(np.abs(shap_values).mean(axis=0), index=X.columns, columns=['mean_abs_shap_value'])
    shap_importance = shap_importance.sort_values(by='mean_abs_shap_value', ascending=False)

    report_path = reports_path / 'shap_feature_importance.csv'
    shap_importance.to_csv(report_path)
    logging.info(f" -> SHAP importance report saved to '{report_path}'")

    print("\n--- SHAP Feature Importance (Top 20) ---")
    print(shap_importance.head(20))

    logging.info("\n--- Analysis Complete ---")


if __name__ == '__main__':
    analyze_final_stage1_features()