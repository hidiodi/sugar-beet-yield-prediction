# File: src/models/explain_stage1_model.py
# Description: ADVANCED analysis script V2. Deconstructs multiple model failures to find systematic blind spots.

import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
import numpy as np
import os
import warnings
from sklearn.inspection import PartialDependenceDisplay

warnings.filterwarnings("ignore")

# --- Configuration ---
MODEL_PATH = os.path.join('src/models', 'final_xgb_model_champion.joblib')  # Make sure to use the V2 model
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
REPORT_DIR = os.path.join('reports', 'final_model_explanation_advanced')

# Align data splits with the definitive training script
VALIDATION_START_YEAR = 2007
TEST_START_YEAR = 2015


def load_model_and_data(feature_list):
    # (This function remains unchanged from the previous version)
    print("--- 1. Loading Model and Recreating Data Splits ---")
    try:
        df = pd.read_csv(DATA_PATH)
        missing_cols = [col for col in feature_list if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Data file is missing required feature columns: {missing_cols}")
        print(f"✅ Successfully loaded {len(df)} samples from {DATA_PATH}.")
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {DATA_PATH}.")
        return [None] * 6

    try:
        xgb_model = joblib.load(MODEL_PATH)
        print(f"✅ Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {MODEL_PATH}.")
        return [None] * 6

    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x)
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']

    validation_df = df[(df['year'] >= VALIDATION_START_YEAR) & (df['year'] < TEST_START_YEAR)].copy()
    test_df = df[df['year'] >= TEST_START_YEAR].copy()

    X_validation = validation_df[feature_list]
    y_validation = validation_df['kreisYield_detrended']
    X_test = test_df[feature_list]
    y_test = test_df['kreisYield_detrended']

    print(f"Validation set size: {len(X_validation)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print("-" * 50)
    return xgb_model, X_validation, y_validation, X_test, y_test, test_df


# (analyze_shap_globally, analyze_shap_feature_dependence, plot_pdp_with_ice remain unchanged)
def analyze_shap_globally(explainer, shap_values, data, data_split_name):
    print(f"\n--- 2. Global Model Behavior (SHAP Beeswarm) on {data_split_name} data ---")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, data, show=False, plot_type='dot')
    plt.title(f"SHAP Summary (Beeswarm) Plot - {data_split_name}")
    plt.xlabel("SHAP Value (Impact on Detrended Yield Prediction)")
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, f'shap_global_beeswarm_{data_split_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ SHAP Beeswarm Plot saved to {save_path}")
    plt.close()

    print(f"\n--- 3. Top Feature Interactions (SHAP) on {data_split_name} data ---")
    shap_interaction_values = explainer.shap_interaction_values(data)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_interaction_values, data, max_display=8, show=False)
    plt.title(f"Top Feature Interactions by SHAP Value - {data_split_name}")
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, f'shap_global_interactions_{data_split_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ SHAP Interaction Plot saved to {save_path}")
    plt.close()


def analyze_shap_feature_dependence(shap_values, data, features, data_split_name):
    print(f"\n--- 4. Detailed Feature Dependence (SHAP) on {data_split_name} data ---")
    for feature in features:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature, shap_values, data, interaction_index="auto", show=False)
        plt.title(f"SHAP Dependence for '{feature}' - {data_split_name}")
        plt.ylabel("SHAP Value (Impact on Detrended Yield)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(REPORT_DIR, f'shap_dependence_{feature}_{data_split_name}.png')
        plt.savefig(save_path)
        plt.close()
    print(f"✅ SHAP Dependence Plots saved for top features.")


# <<< NEW FUNCTION REPLACES THE OLD ONE >>>
def analyze_top_n_failures(xgb_model, explainer, X_test, y_test, test_df, n=10):
    """
    Finds the top N worst predictions, generates individual waterfall plots for each,
    and creates a summary plot of the average SHAP contributions across all failures.
    """
    print(f"\n--- 5. Deconstructing the Top {n} Worst Predictions on Test Set ---")

    y_pred_detrended = pd.Series(xgb_model.predict(X_test), index=y_test.index)
    errors = y_pred_detrended - y_test
    absolute_errors = np.abs(errors)

    top_n_indices = absolute_errors.nlargest(n).index

    failure_analysis_results = []

    print(f"Generating individual deconstruction plots for top {n} failures...")
    for i, idx in enumerate(top_n_indices):
        instance_info = test_df.loc[idx]
        actual_yield = instance_info['kreisYield']
        trend_yield = instance_info['yield_trend']
        predicted_yield = y_pred_detrended.loc[idx] + trend_yield

        single_instance_df = X_test.loc[[idx]]
        shap_values_single = explainer.shap_values(single_instance_df)

        failure_data = {
            'rank': i + 1, 'index': idx, 'district': instance_info['district_no'], 'year': instance_info['year'],
            'actual_yield': actual_yield, 'predicted_yield': predicted_yield, 'error': predicted_yield - actual_yield
        }
        failure_data.update({f'shap_{col}': val for col, val in zip(X_test.columns, shap_values_single[0])})
        failure_analysis_results.append(failure_data)

        plt.figure(figsize=(15, 8))
        shap.waterfall_plot(shap.Explanation(
            values=shap_values_single[0], base_values=explainer.expected_value, data=single_instance_df.iloc[0]
        ), show=False, max_display=15)

        title = (
            f"Deconstruction of Failure #{i + 1} (District {instance_info['district_no']}, Year {instance_info['year']})\n"
            f"Pred: {predicted_yield:.2f}, Actual: {actual_yield:.2f}, Error: {failure_data['error']:.2f}")
        plt.title(title)
        plt.tight_layout()
        save_path = os.path.join(REPORT_DIR, f'failure_deconstruction_rank_{i + 1}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"✅ Individual waterfall plots saved.")

    failures_df = pd.DataFrame(failure_analysis_results)
    csv_path = os.path.join(REPORT_DIR, 'top_failures_detailed_analysis.csv')
    failures_df.to_csv(csv_path, index=False)
    print(f"✅ Detailed failure analysis saved to {csv_path}")

    shap_cols = [col for col in failures_df.columns if col.startswith('shap_')]
    avg_shap_failures = failures_df[shap_cols].mean()
    avg_shap_failures.index = [idx.replace('shap_', '') for idx in avg_shap_failures.index]
    avg_shap_failures = avg_shap_failures.sort_values()

    plt.figure(figsize=(10, 12))
    avg_shap_failures.plot(kind='barh', color=(avg_shap_failures > 0).map({True: 'r', False: 'b'}))
    plt.title(f'Average SHAP Contribution Across Top {n} Model Failures')
    plt.xlabel('Average SHAP Value (Impact on Detrended Yield)')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    summary_plot_path = os.path.join(REPORT_DIR, 'top_failures_summary_plot.png')
    plt.savefig(summary_plot_path)
    print(f"✅ Summary plot of failure drivers saved to {summary_plot_path}")
    plt.close()


def plot_pdp_with_ice(xgb_model, X_data, features):
    print("\n--- 6. Analyzing Average vs. Individual Effects (PDP with ICE) ---")
    try:
        n_features = len(features)
        n_cols = min(n_features, 3)
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(7 * n_cols, 5 * n_rows), sharey=True)
        axes = axes.flatten()

        display = PartialDependenceDisplay.from_estimator(
            xgb_model, X_data, features, kind='both',
            ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
            pd_line_kw={"color": "tab:red", "linewidth": 2},
            n_jobs=-1, grid_resolution=50, ax=axes
        )
        display.figure_.suptitle(
            "Partial Dependence with Individual Conditional Expectation (ICE)\nRed line is the average effect (PDP), Blue lines are individual samples",
            fontsize=16
        )
        display.figure_.get_axes()[0].set_ylabel("Predicted Detrended Yield")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(REPORT_DIR, 'pdp_with_ice_plots.png')
        plt.savefig(save_path)
        print(f"✅ PDP & ICE Plots saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"❌ Error generating PDP/ICE plots: {e}")


if __name__ == "__main__":
    os.makedirs(REPORT_DIR, exist_ok=True)
    try:
        temp_model = joblib.load(MODEL_PATH)
        FEATURE_COLS = temp_model.feature_names_in_
        print("✅ Automatically loaded feature list from the trained model.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load model to get feature list. Error: {e}")
        exit()

    xgb_model, X_validation, y_validation, X_test, y_test, test_df = load_model_and_data(FEATURE_COLS)

    if xgb_model is not None and X_test is not None:
        print(f"\n--- Starting Advanced Explanation Analysis for Final Model ---")
        X_test_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values_test = explainer.shap_values(X_test_sample)

        analyze_shap_globally(explainer, shap_values_test, X_test_sample, "test_set")

        shap_sum = np.abs(shap_values_test).mean(axis=0)
        top_features_df = pd.DataFrame(shap_sum, index=X_test_sample.columns, columns=['SHAP_Importance'])
        top_features_df = top_features_df.sort_values('SHAP_Importance', ascending=False)
        top_6_features = top_features_df.head(6).index.tolist()
        print(f"\nTop 6 features determined by SHAP for deep dive: {top_6_features}")

        analyze_shap_feature_dependence(shap_values_test, X_test_sample, top_6_features, "test_set")

        # <<< CALL THE NEW, MORE POWERFUL ANALYSIS FUNCTION >>>
        analyze_top_n_failures(xgb_model, explainer, X_test, y_test, test_df, n=10)

        plot_pdp_with_ice(xgb_model, X_validation, top_6_features)

        print("\n✅ Advanced Analysis Complete. Outputs saved to directory: " + REPORT_DIR)
    else:
        print("\n❌ Analysis failed due to missing model or data.")