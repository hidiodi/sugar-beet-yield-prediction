import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# --- Setup ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
from src import config


def check_correlation(df, forecast_col, observed_col, name):
    """Calculates and plots correlation between Forecast and Truth."""
    # Filter for valid data
    valid = df[[forecast_col, observed_col]].dropna()

    if valid.empty:
        print(f"\n[WARNING] No valid data found for {name} ({forecast_col} vs {observed_col})")
        return

    # Calculate Metrics
    p_corr, _ = pearsonr(valid[forecast_col], valid[observed_col])
    s_corr, _ = spearmanr(valid[forecast_col], valid[observed_col])

    print(f"\n--- {name} ANALYSIS ---")
    print(f"Feature Pair:  {forecast_col}  vs  {observed_col}")
    print(f"Count:         {len(valid)}")
    print(f"Pearson Corr:  {p_corr:.4f} (Linear)")
    print(f"Spearman Corr: {s_corr:.4f} (Rank)")

    # Plot
    plt.figure(figsize=(8, 6))
    sns.regplot(data=valid, x=forecast_col, y=observed_col,
                scatter_kws={'alpha': 0.1, 's': 10}, line_kws={'color': 'red'})
    plt.title(f"{name}: Forecast vs Truth\nR={p_corr:.3f}")
    plt.xlabel(f"Forecast (March): {forecast_col}")
    plt.ylabel(f"Observed (Sept): {observed_col}")
    plt.grid(True, alpha=0.3)

    out_file = config.DATA_DIR / f'06_model_output/corr_check_{name}.png'
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")


def main():
    print("Loading Data...")
    df = pd.read_csv(config.XGBOOST_TRAINING_CONFIG['DATA_PATH'])

    # 1. HEAT CHECK
    # Does the ECMWF 'Probability of Warmth' actually predict 'Hot Days'?
    check_correlation(
        df,
        forecast_col='summer_temp_prob_warm_forecast',
        observed_col='summer_days_tmax_gt_30c',
        name='HEAT_FORECAST'
    )

    # 2. DROUGHT CHECK (Indirect)
    # Does the ECMWF 'Precip Anomaly' correlate with 'Heat Days'?
    # (Expect Negative Correlation: Wet summers are usually cool)
    check_correlation(
        df,
        forecast_col='summer_precip_anomaly_forecast',
        observed_col='summer_days_tmax_gt_30c',
        name='PRECIP_VS_HEAT'
    )

    # 3. WET STRESS CHECK
    # Does ECMWF 'Precip Anomaly' predict 'Anoxia Events' (Waterlogging)?
    if 'anoxia_events' in df.columns:
        check_correlation(
            df,
            forecast_col='summer_precip_anomaly_forecast',
            observed_col='anoxia_events',
            name='WET_STRESS_FORECAST'
        )


if __name__ == "__main__":
    main()