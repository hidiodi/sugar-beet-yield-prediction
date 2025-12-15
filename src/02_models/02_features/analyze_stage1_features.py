import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pandas import read_csv
from plotly.subplots import make_subplots


# =========================================
# 1. SETUP & DUMMY DATA GENERATION
# =========================================
# (Replace this entire section with: df = pd.read_csv('your_file.csv') )


print("Generating dummy data for demonstration...")
df = read_csv('data/02_intermediate/ecmwf51_forecast_features_BY_MEMBER.csv')
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.\n")

# =========================================
# 2. INITIAL DATA PREP & DEFINITIONS
# =========================================

# Identify feature columns (excluding the keys)
feature_cols = [c for c in df.columns if c not in ['year', 'district_no', 'seas5_member']]
spring_cols = [c for c in feature_cols if 'spring' in c]
summer_cols = [c for c in feature_cols if 'summer' in c]

# Create an ensemble-aggregated version (mean of all members for each year/district)
# This is usually preferred for high-level trend analysis.
df_ens_mean = df.groupby(['year', 'district_no'])[feature_cols].mean().reset_index()

# =========================================
# 3. QUICK STATISTICAL ANALYSIS
# =========================================

print("--- QUICK STATS (Ensemble Means) ---")
print(df_ens_mean[feature_cols].describe().T[['mean', 'std', 'min', 'max']])
print("\n")

# =========================================
# 4. SMART VISUALS
# =========================================

# --- VISUAL 1: Correlation Heatmap (Seaborn) ---
# Why? Crucial to see if forecasts match physics (e.g., High Solar Rad should correlate with High Temp)
plt.figure(figsize=(12, 10))
corr = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
plt.title('Correlation Matrix of Forecast Anomalies (All Members)', fontsize=16)
plt.tight_layout()
plt.show()

# --- VISUAL 2: Interactive Trends with Uncertainty (Plotly) ---
# Why? You need to see trends over time, but also how much the ensemble members disagree (uncertainty).
# We will melt the data to make it "tidy" for Plotly's faceting.

print("Preparing interactive trend dashboard...")

# 1. Aggregate to get mean, min, and max across ensemble members for shading
df_agg = df.groupby(['year', 'district_no'])[feature_cols].agg(['mean', 'min', 'max']).reset_index()
# Flatten MultiIndex columns
df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]

# Let's pick one key variable to demonstrate, e.g., Summer Temperature, for all districts
var_base = 'summer_temp_anomaly_forecast'
fig_trend = go.Figure()

# Generate a color scale for districts
colors = px.colors.qualitative.Plotly

for i, district in enumerate(sorted(df_agg['district_no'].unique())):
    d_data = df_agg[df_agg['district_no'] == district]
    color = colors[i % len(colors)]

    # Add the ensemble range (uncertainty ribbon)
    fig_trend.add_trace(go.Scatter(
        x=pd.concat([d_data['year'], d_data['year'][::-1]]),
        y=pd.concat([d_data[f'{var_base}_max'], d_data[f'{var_base}_min'][::-1]]),
        fill='toself',
        fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) + (0.2,)}',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name=f'District {district} Uncertainty'
    ))

    # Add the ensemble mean line
    fig_trend.add_trace(go.Scatter(
        x=d_data['year'],
        y=d_data[f'{var_base}_mean'],
        line=dict(color=color, width=3),
        mode='lines+markers',
        name=f'District {district} Mean'
    ))

fig_trend.update_layout(
    title=f'Interactive Trend: {var_base} (Ensemble Mean & Spread)',
    yaxis_title='Anomaly',
    xaxis_title='Year',
    hovermode="x unified",
    template="plotly_white"
)
fig_trend.show()

# --- VISUAL 3:  Smart Comparison - Spring vs Summer (Plotly Scatter Matrix) ---
# Why? Quickly spot if a hot spring forecasts a hot summer.

fig_scatter = px.scatter_matrix(
    df_ens_mean,
    dimensions=['spring_temp_anomaly_forecast', 'summer_temp_anomaly_forecast',
                'spring_precip_anomaly_forecast', 'summer_precip_anomaly_forecast'],
    color="district_no",
    title="Spring vs Summer Interactions (Ensemble Means by District)",
    labels={col: col.replace('_anomaly_forecast', '') for col in df.columns}  # Shorten labels for readability
)
fig_scatter.update_traces(diagonal_visible=False, showupperhalf=False)
fig_scatter.update_layout(height=800)
fig_scatter.show()

# --- VISUAL 4: District Anomaly Distributions (Seaborn Violin) ---
# Why? To see which districts have the most extreme or most variable forecasts.

# Melt data for easy faceting: District vs Value, colored by Variable
melted_df = df.melt(id_vars=['year', 'district_no', 'seas5_member'],
                    value_vars=['summer_temp_anomaly_forecast', 'summer_precip_anomaly_forecast'],
                    var_name='Variable', value_name='Anomaly')

plt.figure(figsize=(14, 6))
sns.violinplot(data=melted_df, x='district_no', y='Anomaly', hue='Variable', split=False, inner='quartile',
               palette='muted')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title('Distribution of Key Summer Anomalies by District (All Years & Members)')
plt.grid(True, axis='y', alpha=0.3)
plt.show()