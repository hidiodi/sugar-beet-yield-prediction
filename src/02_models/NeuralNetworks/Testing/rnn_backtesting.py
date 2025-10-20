# File: src/models/rnn_backtesting.py
# Description: A deep-dive diagnostic script to evaluate the DEEP ATTENTION RNN model performance
#              at the district level using a robust, rolling-forecast backtest.

import pandas as pd
import numpy as np
import geopandas as gpd
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Layer
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Configuration ---
DATA_PATH = os.path.join('data', '05_model_input', 'stage1_preseason_features.csv')
GEOJSON_PATH = os.path.join('data', '01_raw', 'districts_official.geojson')
REPORT_DIR = os.path.join('reports', 'figures', 'rnn_district_level_diagnostics_deep_v5')

# --- Backtesting Parameters ---
BACKTEST_START_YEAR = 2000
BACKTEST_END_YEAR = 2021
TIME_STEPS = 5  # Sequence length for the RNN
LOW_DATA_THRESHOLD = 10  # Districts with fewer years of data than this will be flagged
MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT = 5

# ============================ V2 FEATURE SET ============================
# Must be consistent with the features used for training the RNN model.
FEATURE_COLS = [
    'antecedent_frost_days_anomaly', 'antecedent_heavy_precip_days_anomaly', 'antecedent_gdd_sum_anomaly',
    'spring_temp_anomaly_forecast', 'spring_precip_anomaly_forecast', 'spring_solar_rad_anomaly_forecast',
    'spring_evaporation_anomaly_forecast', 'spring_runoff_anomaly_forecast', 'spring_soil_temp_l1_anomaly_forecast',
    'spring_snowfall_anomaly_forecast', 'summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast',
    'summer_solar_rad_anomaly_forecast', 'summer_evaporation_anomaly_forecast', 'summer_runoff_anomaly_forecast',
    'summer_soil_temp_l1_anomaly_forecast', 'summer_snowfall_anomaly_forecast', 'spring_temp_prob_warm_forecast',
    'spring_precip_prob_wet_forecast', 'spring_solar_rad_prob_wet_forecast', 'spring_evaporation_prob_wet_forecast',
    'spring_runoff_prob_wet_forecast', 'spring_soil_temp_l1_prob_warm_forecast', 'spring_snowfall_prob_wet_forecast',
    'summer_temp_prob_warm_forecast', 'summer_precip_prob_wet_forecast', 'summer_solar_rad_prob_wet_forecast',
    'summer_evaporation_prob_wet_forecast', 'summer_runoff_prob_wet_forecast', 'summer_soil_temp_l1_prob_warm_forecast',
    'summer_snowfall_prob_wet_forecast', 'lat', 'lon', 'avg_elevation', 'avg_slope', 'avg_bdod_0_30cm',
    'avg_clay_0_30cm', 'avg_sand_0_30cm', 'avg_som_0_30cm', 'avg_phh2o_0_30cm', 'avg_bdod_0_100cm',
    'avg_clay_0_100cm', 'avg_sand_0_100cm', 'avg_som_0_100cm', 'avg_phh2o_0_100cm', 'winter_cropland_ndvi_mean',
    'winter_cropland_ndvi_anomaly', 'winter_cropland_LST_mean', 'winter_cropland_LST_anomaly',
    'winter_cropland_snow_cover_days', 'national_avg_yield_lag1', 'producer_price_index_lag1',
    'seed_price_index_lag1', 'energy_price_index_lag1', 'fertilizer_price_index_lag1',
    'plant_protection_price_index_lag1', 'profit_margin_proxy_lag1', 'cost_of_inputs_lag1',
    'producer_price_index_lag1_anomaly', 'seed_price_index_lag1_anomaly', 'energy_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly', 'plant_protection_price_index_lag1_anomaly',
    'fertilizer_price_index_lag1_anomaly_capped', 'is_fertilizer_price_extreme', 'is_summer_forecast_dry',
    'gdd_x_fertilizer_price', 'spring_temp_x_spring_precip', 'antecedent_gdd_sum_anomaly_sq',
    'summer_heat_x_profit_margin', 'summer_precip_x_input_costs', 'spring_temp_prob_warm_forecast_sq',
    'summer_temp_prob_warm_forecast_sq', 'spring_precip_prob_wet_forecast_sq', 'summer_precip_prob_wet_forecast_sq'
]


# ======================================================================

class Attention(Layer):
    """ Custom Keras Attention Layer. """

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def create_sequences(X, y, time_steps=1):
    """Creates sequences from numpy arrays for RNN model training."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def build_model(input_shape):
    """Defines and compiles a deeper, more regularized RNN model with stacked LSTMs and Attention."""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.4),
        Attention(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def run_backtest(df: pd.DataFrame, feature_cols: list, time_steps: int):
    """
    Performs a rolling forecast origin backtest for the RNN model, ensuring
    sequences are created on a per-district basis to prevent data leakage.
    """
    print(f"\n--- Starting Deep RNN Backtest from {BACKTEST_START_YEAR} to {BACKTEST_END_YEAR} ---")
    all_predictions = []

    for year_to_predict in tqdm(range(BACKTEST_START_YEAR, BACKTEST_END_YEAR + 1), desc="Backtesting Years"):
        train_df = df[df['year'] < year_to_predict].copy()
        test_df = df[df['year'] == year_to_predict].copy()

        if test_df.empty or len(train_df) < (time_steps + 1):
            continue

        scaler = StandardScaler()
        train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

        X_train_list, y_train_list = [], []
        for district_id in train_df['district_no'].unique():
            district_df = train_df[train_df['district_no'] == district_id].sort_values('year')
            if len(district_df) > time_steps:
                X, y = create_sequences(district_df[feature_cols].values, district_df['kreisYield_detrended'].values,
                                        time_steps)
                X_train_list.append(X)
                y_train_list.append(y)

        if not X_train_list: continue
        X_train_seq = np.concatenate(X_train_list)
        y_train_seq = np.concatenate(y_train_list)

        X_test_list, aligned_test_dfs = [], []
        for district_id in test_df['district_no'].unique():
            history_df = train_df[train_df['district_no'] == district_id].sort_values('year')
            if len(history_df) >= time_steps:
                current_year_df = test_df[test_df['district_no'] == district_id]
                full_series_df = pd.concat([history_df, current_year_df])
                sequence_features = full_series_df.tail(time_steps)[feature_cols].values
                if sequence_features.shape[0] == time_steps:
                    X_test_list.append(sequence_features)
                    aligned_test_dfs.append(current_year_df)

        if not X_test_list: continue
        X_test_seq = np.array(X_test_list)
        test_df_aligned = pd.concat(aligned_test_dfs)

        if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0: continue

        model = build_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

        early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=7, min_lr=0.00001)

        model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=64, verbose=0, callbacks=[early_stopping, reduce_lr])

        predicted_detrended = model.predict(X_test_seq, verbose=0).flatten()

        final_predictions = predicted_detrended + test_df_aligned['yield_trend'].values
        fold_results = test_df_aligned[['district_no', 'year', 'kreisYield', 'name']].copy()
        fold_results['predicted_yield'] = final_predictions
        all_predictions.append(fold_results)

    if not all_predictions:
        print("❌ CRITICAL ERROR: No predictions were made. Check year ranges and data availability.")
        return pd.DataFrame()

    results_df = pd.concat(all_predictions, ignore_index=True)
    results_df['error'] = results_df['predicted_yield'] - results_df['kreisYield']
    results_df['abs_error'] = results_df['error'].abs()
    print("\nDeep RNN Backtest complete.")
    return results_df


def calculate_district_metrics(results_df: pd.DataFrame):
    """Calculates R², MAE, and data point count for each district."""
    print("Calculating performance metrics and data counts for each district...")

    def r2_safe(g):
        return r2_score(g['kreisYield'], g['predicted_yield']) if len(g) > 1 else -99

    performance = results_df.groupby('district_no').apply(
        lambda g: pd.Series({
            'r2': r2_safe(g),
            'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield']),
            'name': g['name'].iloc[0] if not g.empty else 'Unknown',
            'data_point_count': len(g)
        })
    ).reset_index()
    performance['is_low_data'] = performance['data_point_count'] < LOW_DATA_THRESHOLD
    save_path = os.path.join(REPORT_DIR, 'district_level_performance_metrics.csv')
    performance.to_csv(save_path, index=False)
    print(f"Detailed district metrics saved to {save_path}")
    return performance


def analyze_yearly_performance(results_df: pd.DataFrame):
    """Calculates and plots R² and MAE for each year in the backtest."""
    print("Calculating performance metrics for each year...")
    yearly_perf = results_df.groupby('year').apply(
        lambda g: pd.Series({
            'r2': r2_score(g['kreisYield'], g['predicted_yield']),
            'mae': mean_absolute_error(g['kreisYield'], g['predicted_yield'])
        })
    ).reset_index()
    print("--- Yearly Performance Summary ---")
    print(yearly_perf)
    print("----------------------------------")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(yearly_perf['year'], yearly_perf['r2'], color='blue', marker='o', label='Yearly R²')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('R-squared (R²)', fontsize=12)
    ax.axhline(0, color='grey', linestyle='--')
    ax.grid(True, which='both', linestyle='--')
    plt.title('Deep RNN Model Performance Over Time (Backtest)', fontsize=16)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, '03_performance_over_time.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Yearly performance plot saved to {save_path}")
    return yearly_perf


def plot_performance_map_with_hatching(district_performance: pd.DataFrame, gdf_districts: gpd.GeoDataFrame):
    """Diagnostic: Geographic map of R² with hatching for low-data districts."""
    print("Generating R-squared Map...")
    merged_gdf = gdf_districts.merge(district_performance, on='district_no', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged_gdf.plot(column='r2', cmap='RdYlGn', linewidth=0.5, ax=ax, edgecolor='0.8',
                    legend=True, legend_kwds={'label': "R-squared (R²)", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'}, vmin=-1, vmax=1)
    low_data_gdf = merged_gdf[merged_gdf['is_low_data'] == True]
    if not low_data_gdf.empty:
        low_data_gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black', linewidth=0.5)
    hatch_patch = mpatches.Patch(hatch='//', facecolor='white', edgecolor='black',
                                 label=f'Low Data (< {LOW_DATA_THRESHOLD} years)')
    plt.legend(handles=[hatch_patch], loc='lower left', title='Data Availability')
    ax.set_title('Deep RNN Model Performance (R²) by District', fontsize=16)
    ax.set_axis_off()
    save_path = os.path.join(REPORT_DIR, '01_r_squared_map_with_hatching.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("Performance map saved.")


def plot_r2_vs_data_count(district_performance: pd.DataFrame):
    """Diagnostic: Scatter plot to show the relationship between R² and data count."""
    print("Generating R² vs. Data Count plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=district_performance, x='data_point_count', y='r2', hue='is_low_data',
                    palette={True: 'red', False: 'blue'}, alpha=0.6)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='grey', linestyle='--')
    plt.title("Relationship Between Data Availability and Deep RNN Performance", fontsize=16)
    plt.xlabel("Number of Years in Backtest per District")
    plt.ylabel("R-squared (R²)")
    plt.legend(title='Is Low Data?')
    save_path = os.path.join(REPORT_DIR, '02_r2_vs_data_count.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("R² vs. Count plot saved.")


def plot_best_worst_district_timelines(district_performance: pd.DataFrame, backtest_results: pd.DataFrame):
    """Diagnostic: Plot prediction timelines for the 3 best and 3 worst districts."""
    print("Generating timeline plots for best and worst performing districts...")
    filtered_perf = district_performance[
        (district_performance['data_point_count'] > 1) & (district_performance['r2'] != -99)].sort_values('r2',
                                                                                                          ascending=False)
    best_districts = filtered_perf.head(3)
    worst_districts_filtered = district_performance[
        (district_performance['data_point_count'] >= MIN_DATAPOINTS_FOR_WORST_DISTRICTS_PLOT) & (
                    district_performance['r2'] != -99)].sort_values('r2', ascending=True)
    worst_districts = worst_districts_filtered.head(3)
    districts_to_plot = pd.concat([best_districts, worst_districts])
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    axes = axes.flatten()
    for i, (_, district_info) in enumerate(districts_to_plot.iterrows()):
        district_no = district_info['district_no']
        district_name = district_info['name']
        district_r2 = district_info['r2']
        district_data = backtest_results[backtest_results['district_no'] == district_no].sort_values('year')
        ax = axes[i]
        ax.plot(district_data['year'], district_data['kreisYield'], label='Actual Yield', color='navy', marker='o',
                markersize=4)
        ax.plot(district_data['year'], district_data['predicted_yield'], label='Predicted Yield', color='red',
                linestyle='--')
        title_prefix = "Best" if i < 3 else "Worst"
        ax.set_title(f"{title_prefix}: {district_name}\n(R² = {district_r2:.2f})", fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle=':')
    plt.suptitle("Prediction Timelines for 3 Best and 3 Worst Performing Districts (Deep RNN)", fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(REPORT_DIR, '04_best_worst_district_timelines.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Best/Worst district timelines saved to {save_path}")


def main():
    """Main function to orchestrate the district-level evaluation pipeline."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("--- Starting Deep RNN District-Level Model Evaluation Pipeline ---")
    try:
        df = pd.read_csv(DATA_PATH)
        gdf_districts = gpd.read_file(GEOJSON_PATH)
        df['district_no'] = df['district_no'].astype(str).str.zfill(5)
        gdf_districts.rename(columns={'id': 'district_no', 'name': 'name'}, inplace=True)
        gdf_districts['district_no'] = gdf_districts['district_no'].astype(str).str.zfill(5)
        df = pd.merge(df, gdf_districts[['district_no', 'name']], on='district_no', how='left')
        print("Data and geo-data loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during loading. Details: {e}")
        return

    print("\n--- Applying Causal (Trailing Mean) Detrending to Prevent Data Leakage ---")
    df.sort_values(by=['district_no', 'year'], inplace=True)
    df['yield_trend'] = df.groupby('district_no')['kreisYield'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(lambda x: x.fillna(method='ffill'))
    df['yield_trend'] = df.groupby('district_no')['yield_trend'].transform(
        lambda x: x.fillna(x.iloc[0]) if not x.isnull().all() else x)
    df.dropna(subset=['yield_trend'], inplace=True)
    df['kreisYield_detrended'] = df['kreisYield'] - df['yield_trend']
    print(" -> Detrending complete.")

    backtest_results = run_backtest(df, FEATURE_COLS, TIME_STEPS)
    if backtest_results.empty:
        print("❌ Backtest did not produce results. Terminating.")
        return

    district_performance = calculate_district_metrics(backtest_results)
    analyze_yearly_performance(backtest_results)
    plot_performance_map_with_hatching(district_performance, gdf_districts)
    plot_r2_vs_data_count(district_performance)
    plot_best_worst_district_timelines(district_performance, backtest_results)

    mae_total = backtest_results['abs_error'].mean()
    r2_total = r2_score(backtest_results['kreisYield'], backtest_results['predicted_yield'])
    print("\n--- Overall Performance Summary (Deep RNN - All Districts) ---")
    print(f"  Mean Absolute Error (MAE):    {mae_total:.2f} dt/ha")
    print(f"  R-squared (R²):               {r2_total:.4f}")
    print("-----------------------------------------------------")

    reliable_districts = district_performance[~district_performance['is_low_data']]
    reliable_results = backtest_results[backtest_results['district_no'].isin(reliable_districts['district_no'])]
    if reliable_results.empty:
        print("⚠️ Warning: No districts were classified as 'reliable'. Cannot compute summary metrics.")
    else:
        mae_reliable = reliable_results['abs_error'].mean()
        r2_reliable = r2_score(reliable_results['kreisYield'], reliable_results['predicted_yield'])
        print("\n--- Performance Summary on RELIABLE Districts (>= 10 years of data) ---")
        print(f"  Number of reliable districts: {len(reliable_districts)} / {len(district_performance)}")
        print(f"  Mean Absolute Error (MAE):    {mae_reliable:.2f} dt/ha")
        print(f"  R-squared (R²):               {r2_reliable:.4f}")
        print("---------------------------------------------------------------------")


if __name__ == "__main__":
    main()

