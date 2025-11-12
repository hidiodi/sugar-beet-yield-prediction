# vsm_cybernetic_model/pipelines/run_final_evaluation.py
import pandas as pd
from xgboost import XGBRegressor
import sys
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from vsm_cybernetic_model.configs import main_config as cfg
from vsm_cybernetic_model.pipelines.prepare_foundational_features import prepare_all_foundational_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sns.set_theme(style="whitegrid")


def run_vsm_backtest():
    """Performs a rigorous time-series backtest of the entire VSM + XGBoost pipeline."""
    logging.info("--- Starting Full VSM-XGBoost Backtest Evaluation ---")

    prepare_all_foundational_features()
    full_df = pd.read_csv(cfg.FOUNDATIONAL_FEATURES_HUMAN)

    all_systems = {
        'VSM1': {'cfg_name': 'system_1_biophysical', 'features': None, 'n_comp': 7},
        'VSM2': {'cfg_name': 'system_2_coordination', 'features': None, 'n_comp': 2},
        'VSM3': {'cfg_name': 'system_3_control', 'features': None, 'n_comp': 3},
        'VSM4': {'cfg_name': 'system_4_strategy', 'features': None, 'n_comp': 2},
        'VSM5': {'cfg_name': 'system_5_policy', 'features': None, 'n_comp': 0},
    }

    for prefix, details in all_systems.items():
        sys_cfg = __import__(f"vsm_cybernetic_model.configs.{details['cfg_name']}", fromlist=[''])
        details['features'] = getattr(sys_cfg, f"{prefix}_INPUT_FEATURES")

    all_predictions = []
    backtest_start_year = 2010
    backtest_end_year = 2022

    for year in tqdm(range(backtest_start_year, backtest_end_year + 1), desc="Backtesting Years"):
        train_df = full_df[full_df['year'] < year].copy()
        test_df = full_df[full_df['year'] == year].copy()
        if test_df.empty or len(train_df) < 500: continue

        X_train_vsm = pd.DataFrame(index=train_df.index)
        X_test_vsm = pd.DataFrame(index=test_df.index)

        for prefix, details in all_systems.items():
            if not details['features']: continue

            train_subset = train_df[details['features']].dropna()
            test_subset = test_df[details['features']].copy()
            if train_subset.empty: continue

            scaler = StandardScaler()
            pca = PCA(n_components=details['n_comp'], random_state=42)

            X_train_scaled = scaler.fit_transform(train_subset)
            X_train_pca = pca.fit_transform(X_train_scaled)

            test_subset.fillna(train_df[details['features']].mean(), inplace=True)
            X_test_scaled = scaler.transform(test_subset)
            X_test_pca = pca.transform(X_test_scaled)

            for i in range(details['n_comp']):
                X_train_vsm.loc[train_subset.index, f'{prefix}_PC{i + 1}'] = X_train_pca[:, i]
                X_test_vsm.loc[test_subset.index, f'{prefix}_PC{i + 1}'] = X_test_pca[:, i]

        vsm_cols = [col for col in X_train_vsm.columns if col.startswith('VSM')]
        train_merged = train_df.join(X_train_vsm).dropna(subset=vsm_cols)
        test_merged = test_df.join(X_test_vsm).dropna(subset=vsm_cols)

        X_train, y_train = train_merged[vsm_cols], train_merged['kreisYield']
        X_test = test_merged[vsm_cols]

        regulator = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        regulator.fit(X_train, y_train)

        predictions = regulator.predict(X_test)
        fold_results = test_merged[['district_no', 'year', 'kreisYield']].copy()
        fold_results['predicted_yield'] = predictions
        all_predictions.append(fold_results)

    if not all_predictions:
        logging.error("Backtest produced no results.");
        return

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['abs_error'] = (results_df['predicted_yield'] - results_df['kreisYield']).abs()

    logging.info("\n\n--- VSM-XGBoost Backtest Performance Summary ---")
    r2 = r2_score(results_df['kreisYield'], results_df['predicted_yield'])
    mae = results_df['abs_error'].mean()

    print(f"  Overall R-squared (R²): {r2:.4f}")
    print(f"  Overall Mean Absolute Error (MAE): {mae:.2f} dt/ha")

    report_dir = cfg.VERIFICATION_DIR / "final_evaluation"
    report_dir.mkdir(exist_ok=True, parents=True)

    yearly_avg = results_df.groupby('year').agg(actual=('kreisYield', 'mean'),
                                                pred=('predicted_yield', 'mean')).reset_index()
    plt.figure(figsize=(14, 8))
    plt.plot(yearly_avg['year'], yearly_avg['actual'], label='Actual Yield', color='navy', marker='o')
    plt.plot(yearly_avg['year'], yearly_avg['pred'], label='VSM-XGBoost Prediction', color='red', linestyle='--')
    plt.title("VSM Model: National Average Yield vs. Prediction", fontsize=16)
    plt.legend();
    plt.grid(True, linestyle=':');
    plt.savefig(report_dir / 'vsm_national_timeline.png')
    plt.close()
    logging.info(f"Saved national timeline plot to '{report_dir}'")
    logging.info("\n--- Evaluation Complete ---")


if __name__ == "__main__":
    run_vsm_backtest()