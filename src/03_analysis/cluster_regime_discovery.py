import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src import config


def analyze_regimes():
    print("--- STARTING REGIME DETECTION (Broad Feature Set) ---")

    # 1. Load Data
    data_path = config.XGBOOST_TRAINING_CONFIG['DATA_PATH']
    df = pd.read_csv(data_path)

    # Calculate Residuals (Ground Truth for validation ONLY)
    if 'stage1_forecast' in df.columns:
        df['target_residual'] = df['kreisYield'] - df['stage1_forecast']
    else:
        print("CRITICAL: stage1_forecast missing. Cannot validate regimes.")
        return

    # 2. Select Clustering Features (Dynamic Drivers from your list)
    # We use a mix of Raw Weather, WOFOST Biology, Satellite, and V14 Indices
    cluster_features = [
        # --- The V14 Normalized Signals ---
        'z_heat', 'z_bal', 'z_tank', 'z_anoxia', 'z_sow',
        'Index_Failure', 'Index_Bumper',

        # --- Physical Drivers (Energy & Water) ---
        'summer_solar_rad_anomaly_forecast',  # Energy
        'summer_water_balance_anomaly',  # Raw Balance
        'effective_winter_water',  # Raw Tank

        # --- Biological Drivers (WOFOST) ---
        'wofost_yield_water_limited',  # The theoretical ceiling
        'cumulative_water_stress',  # Integrated drought
        'anoxia_events',  # Integrated flood risk
        'prob_sowing_failure',  # Operational risk

        # --- Spring/Start Conditions ---
        'spring_soil_temp_l1_anomaly_forecast',  # Early growth speed
        'sowing_doy_anomaly',  # Relative timing

        # --- Observations (Satellite) ---
        'winter_cropland_ndvi_anomaly',  # Plant health check

        # --- Teleconnections (Global State) ---
        'nao_winter_avg'  # North Atlantic Oscillation
    ]

    # Check if features exist in df
    valid_features = [f for f in cluster_features if f in df.columns]
    missing = set(cluster_features) - set(valid_features)
    if missing:
        print(f"Warning: Missing features from list: {missing}")

    print(f"Clustering on {len(valid_features)} features.")

    # Drop rows with NaNs in these columns
    df_clean = df.dropna(subset=valid_features).copy()
    X = df_clean[valid_features]

    # 3. Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Find Optimal K (2 to 15)
    print("\n... Scanning for optimal Cluster count (k=2..15) ...")
    results = []

    for k in tqdm(range(2, 16)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        # Validation: Separation Power
        # How different are the Yield Residuals between these clusters?
        df_clean['temp_cluster'] = labels
        # Calculate the standard deviation of the cluster means (Higher = Clusters describe yield well)
        yield_separation = df_clean.groupby('temp_cluster')['target_residual'].mean().std()

        results.append({
            'k': k,
            'silhouette': score,
            'yield_separation': yield_separation
        })

    res_df = pd.DataFrame(results)

    # 5. Select Best K
    # We weigh Yield Separation higher because we want clusters that explain Yield.
    # Normalize metrics
    res_df['sil_norm'] = (res_df['silhouette'] - res_df['silhouette'].min()) / (
                res_df['silhouette'].max() - res_df['silhouette'].min())
    res_df['sep_norm'] = (res_df['yield_separation'] - res_df['yield_separation'].min()) / (
                res_df['yield_separation'].max() - res_df['yield_separation'].min())

    # Heuristic: Combined Score
    res_df['score'] = res_df['sil_norm'] + (1.5 * res_df['sep_norm'])

    best_k = int(res_df.loc[res_df['score'].idxmax(), 'k'])

    print("\n=== CLUSTERING RESULTS ===")
    print(res_df.round(3).to_string(index=False))
    print(f"\n>> WINNER: k={best_k}")

    # 6. Apply Best Clustering
    print(f"\n... Analyzing Regimes for k={best_k} ...")
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_clean['Regime_ID'] = final_model.fit_predict(X_scaled)

    # 7. Regime Profiling (What defines them?)
    print("\n=== REGIME PROFILES (Mean Z-Scores / Values) ===")

    # We show key indicators to interpret the clusters
    key_indicators = ['target_residual', 'Index_Failure', 'Index_Bumper', 'z_heat', 'z_bal', 'z_tank']
    profile = df_clean.groupby('Regime_ID')[key_indicators].mean()
    count = df_clean['Regime_ID'].value_counts().sort_index()
    profile.insert(0, 'Count', count)

    print(profile.round(2).to_string())

    # 8. Year Mapping (Global Context)
    print("\n=== DOMINANT REGIME BY YEAR (Problem Years) ===")
    year_regimes = df_clean.groupby('year')['Regime_ID'].agg(lambda x: x.mode()[0])

    check_years = [2003, 2013, 2014, 2018, 2024]

    print(f"{'Year':<6} | {'Regime':<8} | {'Avg Yield Res':<14} | {'Interpretation'}")
    print("-" * 60)

    for y in check_years:
        if y not in year_regimes.index: continue
        rid = year_regimes[y]
        res = profile.loc[rid, 'target_residual']

        interp = "Normal"
        if res < -50:
            interp = "CRASH"
        elif res > 50:
            interp = "BUMPER"

        print(f"{y:<6} | {rid:<8} | {res:>10.2f}     | {interp}")

    # 9. Save
    output_path = config.DATA_DIR / '05_model_input/stage1_features_with_regimes.csv'
    # Merge regime back to original DF (keep NaNs as -1 or separate)
    df_final = df.merge(df_clean[['district_no', 'year', 'Regime_ID']], on=['district_no', 'year'], how='left')
    df_final.to_csv(output_path, index=False)
    print(f"\n>> Saved Regime-Labeled Data to: {output_path}")


if __name__ == "__main__":
    analyze_regimes()