import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setup
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))
from src import config


def run_adaptive_conformal_scaling(predictions_df, alpha=0.05, gamma=0.15):
    """
    Applies Adaptive Conformal Inference (ACI) to guarantee 95% coverage.
    """
    df = predictions_df.sort_values(['district_no', 'year']).copy()

    # Init Scores
    df['q_t'] = 0.0
    df['cqr_lower'] = df['predicted_yield_lower']
    df['cqr_upper'] = df['predicted_yield_upper']
    df['covered'] = 0

    districts = df['district_no'].unique()
    results = []

    for district in districts:
        d_data = df[df['district_no'] == district].copy()
        q = 0.0

        for idx in d_data.index:
            # Current Bounds
            lower_raw = d_data.loc[idx, 'predicted_yield_lower']
            upper_raw = d_data.loc[idx, 'predicted_yield_upper']
            actual = d_data.loc[idx, 'kreisYield']

            # Apply Adjustment
            cqr_lower = lower_raw - q
            cqr_upper = upper_raw + q

            d_data.loc[idx, 'cqr_lower'] = cqr_lower
            d_data.loc[idx, 'cqr_upper'] = cqr_upper
            d_data.loc[idx, 'q_t'] = q

            # Check Coverage
            is_covered = (actual >= cqr_lower) and (actual <= cqr_upper)
            d_data.loc[idx, 'covered'] = int(is_covered)

            # Update q (Adaptive)
            if not is_covered:
                q += gamma * (1 - alpha)  # Expand
            else:
                q -= gamma * alpha  # Shrink

        results.append(d_data)

    return pd.concat(results)


def main():
    # Load from the Hybrid Backtest Folder
    input_path = Path(
        "reports/figures/district_level_diagnostics/final_quantile_champion/full_backtest_predictions.csv")

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run backtest first.")
        return

    df = pd.read_csv(input_path)

    # 1. Calculate Raw Stats
    raw_mae = np.mean(np.abs(df['kreisYield'] - df['predicted_yield_median']))
    raw_cov = df.apply(
        lambda x: (x['kreisYield'] >= x['predicted_yield_lower']) and (x['kreisYield'] <= x['predicted_yield_upper']),
        axis=1).mean()

    print(f"--- Baseline Performance ---")
    print(f"MAE: {raw_mae:.2f} dt/ha")
    print(f"Raw Coverage: {raw_cov:.1%} (Target 95%)")

    # 2. Run CQR
    cqr_df = run_adaptive_conformal_scaling(df, alpha=0.05)

    # 3. Calculate CQR Stats
    cqr_cov = cqr_df['covered'].mean()
    cqr_width = (cqr_df['cqr_upper'] - cqr_df['cqr_lower']).mean()

    print(f"\n--- Conformal Prediction (ACI) ---")
    print(f"Coverage: {cqr_cov:.1%} (calibrated)")
    print(f"Avg Interval Width: {cqr_width:.2f} dt/ha")

    # 4. Save
    output_path = input_path.parent / "conformal_predictions.csv"
    cqr_df.to_csv(output_path, index=False)
    print(f"\nSaved certified forecasts to: {output_path}")


if __name__ == "__main__":
    main()