import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- Setup ---
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src import config

# --- Config ---
DATA_PATH = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
OUTPUT_DIR = config.DATA_DIR / '06_model_output'


def optimize_clusters():
    print("Loading Scenario Data for Cluster Optimization...")
    df = pd.read_csv(DATA_PATH)

    # 1. Select the "Physical Signature" Features
    # These define the 'State of the World'
    features = [
        'summer_days_tmax_gt_30c',  # Heat Stress
        'summer_water_balance_anomaly',  # Drought Stress
        'effective_winter_water',  # Soil Memory
        'anoxia_events',  # Wet Stress
        'sowing_doy_anomaly',  # Management Stress
        'solar_capture_potential'  # Energy Availability
    ]

    # Drop rows with missing features
    data = df[features + ['year', 'district_no', 'kreisYield']].dropna()
    X = data[features]

    # 2. Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    # 3. Iterate k from 2 to 8
    print("\n--- SEARCHING FOR OPTIMAL K ---")
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Metric 1: Silhouette (Mathematical compactness)
        sil = silhouette_score(X_scaled, labels)

        # Metric 2: Yield Separation (Pragmatic utility)
        # We calculate the difference between the Highest Yield Cluster and Lowest Yield Cluster
        data['Temp_Cluster'] = labels
        yield_means = data.groupby('Temp_Cluster')['kreisYield'].mean()
        yield_spread = yield_means.max() - yield_means.min()

        results.append({
            'k': k,
            'silhouette': sil,
            'yield_spread': yield_spread,
            'model': kmeans,
            'labels': labels
        })
        print(f"k={k}: Silhouette={sil:.3f} | Yield Spread={yield_spread:.1f} dt/ha")

    # 4. Find the "Best" K (Maximizing Yield Spread is usually best for Regime Switching)
    best_result = sorted(results, key=lambda x: x['yield_spread'], reverse=True)[0]
    best_k = best_result['k']

    print(f"\n>>> WINNER: k={best_k} (Max Yield Spread) <<<")

    # 5. Deep Dive into the Winner
    data['Cluster'] = best_result['labels']

    # Cluster DNA
    print(f"\n--- CLUSTER DNA (k={best_k}) ---")
    profile = data.groupby('Cluster')[features + ['kreisYield']].mean()
    # Add count of years
    profile['Count'] = data.groupby('Cluster')['year'].count()
    print(profile.round(2).to_string())

    # Key Years Check
    print(f"\n--- WHERE ARE THE KEY YEARS? (k={best_k}) ---")
    for year in [2003, 2007, 2014, 2018]:
        sub = data[data['year'] == year]
        if len(sub) == 0: continue
        mode = sub['Cluster'].mode()[0]
        count = len(sub[sub['Cluster'] == mode])
        total = len(sub)
        print(f"Year {year}: Cluster {mode} ({count}/{total} districts)")

    # 6. Visualization: Yield Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='kreisYield', data=data, palette='viridis')
    plt.title(f"Yield Distribution by Cluster (k={best_k})\nProof that Regimes differentiate Outcome")
    plt.ylabel("Actual Yield (dt/ha)")

    out_path = OUTPUT_DIR / f'cluster_yield_distribution_k{best_k}.png'
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    optimize_clusters()