# File: src/models/evaluate_model_robustly.py
# Description: A definitive evaluation script that uses a robust backtesting
#              methodology (rolling forecast origin) to get a stable estimate
#              of model performance and generates insightful state-level plots.

import pandas as pd
import geopandas as gpd
import joblib
from xgboost import XGBRegressor
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Define Paths and Backtesting Configuration ---
MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion.joblib')
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = os.path.join('reports', 'figures', 'final_model_evaluation')

BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2021


def run_backtest(df: pd.DataFrame, model_template: XGBRegressor):
    """
    Performs a rolling forecast origin backtest.
    """
    print(f"\n--- Starting Robust Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")
    all_predictions = []
    feature_cols = model_template.feature_names_in_

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Years"):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()

        if test_df.empty or train_df.empty:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df['kreisYield_detrended']
        X_test = test_df[feature_cols]

        model = clone(model_template)
        model.fit(X_train, y_train)

        detrended_predictions = model.predict(X_test)
        final_predictions = detrended_predictions + test_df['yield_trend']

        fold_results = test_df[['district_no', 'year', 'state_name', 'kreisYield']].copy()
        fold_results['predicted_yield'] = final_predictions
        all_predictions.append(fold_results)

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()

    print("\nBacktest complete.")
    return results_df


def plot_state_level_performance(results_df: pd.DataFrame):
    """Diagnostic 1: State-level facet grid to analyze regional performance."""
    print("Generating Diagnostic 1: State-Level Performance Grid...")

    state_avg = results_df.groupby(['state_name', 'year']).agg(
        actual_yield=('kreisYield', 'mean'),
        predicted_yield=('predicted_yield', 'mean')
    ).reset_index()

    state_avg_melted = state_avg.melt(id_vars=['state_name', 'year'],
                                      value_vars=['actual_yield', 'predicted_yield'],
                                      var_name='Yield Type', value_name='Yield')

    g = sns.FacetGrid(state_avg_melted, col='state_name', col_wrap=4, height=3, aspect=1.5, sharey=True)
    g.map_dataframe(sns.lineplot, x='year', y='Yield', hue='Yield Type', style='Yield Type', markers=True)
    g.set_titles("{col_name}")
    g.add_legend()
    g.fig.suptitle(f"State-Level Performance (Backtest {BACKTEST_START_YEAR}-{BACKTEST_END_YEAR})",
                   fontsize=16, y=1.02)

    save_path = os.path.join(REPORT_DIR, '01_state_level_performance.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_error_over_time(results_df: pd.DataFrame):
    """Diagnostic 2: Show the stability of model performance over the backtest period."""
    print("Generating Diagnostic 2: Error Over Time...")

    mae_by_year = results_df.groupby('year')['abs_error'].mean().reset_index()
    r2_by_year = results_df.groupby('year').apply(
        lambda g: r2_score(g['kreisYield'], g['predicted_yield'])
    ).reset_index(name='r2_score')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    sns.lineplot(data=mae_by_year, x='year', y='abs_error', marker='o', ax=ax1)
    ax1.set_title("Model Stability: Mean Absolute Error per Year", fontsize=14)
    ax1.set_ylabel("Mean Absolute Error (dt/ha)")
    ax1.set_ylim(bottom=0)

    sns.lineplot(data=r2_by_year, x='year', y='r2_score', marker='o', color='green', ax=ax2)
    ax2.set_title("Model Stability: R-squared per Year", fontsize=14)
    ax2.set_ylabel("R-squared (R²)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(r2_by_year['year'].astype(int))

    fig.suptitle(f"Model Performance Across Backtest Folds ({BACKTEST_START_YEAR}-{BACKTEST_END_YEAR})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(REPORT_DIR, '02_error_over_time.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


def main():
    """Main function to orchestrate the robust evaluation pipeline."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting Robust Model Evaluation Pipeline ---")

    try:
        model_template = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)

        # ============================ THE FIX ============================
        # ### CRITICAL FIX: Correctly load and rename columns from the GeoJSON ###
        gdf_districts = gpd.read_file(GEOJSON_PATH)
        # Rename 'id' to 'district_no' AND 'state' to 'state_name'
        gdf_districts.rename(columns={'id': 'district_no', 'state': 'state_name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        state_lookup = gdf_districts[['district_no', 'state_name']]

        df = pd.merge(df, state_lookup, on='district_no', how='left')
        # ===============================================================

        print("Model template, data, and state names loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: A required file was not found. Details: {e}")
        return

    df.sort_values(by=['district_no', 'year'], inplace=True)

    # --- FIX: Apply Causal (Trailing Mean) Detrending to Prevent Data Leakage ---
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x
    )
    df.dropna(subset=['yield_trend'], inplace=True)
    # ----------------------------------------------------------------------------

    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']

    backtest_results = run_backtest(df, model_template)

    plot_state_level_performance(backtest_results)
    plot_error_over_time(backtest_results)

    final_mae = backtest_results['abs_error'].mean()
    final_r2 = r2_score(backtest_results['kreisYield'], backtest_results['predicted_yield'])
    print("\n--- Overall Backtest Performance Summary ---")
    print(f"  Mean Absolute Error (MAE): {final_mae:.2f} dt/ha")
    print(f"  R-squared (R²):            {final_r2:.4f}")
    print("------------------------------------------")


if __name__ == "__main__":
    main()