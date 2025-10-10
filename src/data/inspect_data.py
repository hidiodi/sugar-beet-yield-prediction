# inspect_data.py

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# --- Step 1: Open the dataset ---
try:
    filepath = "data/02_intermediate/agera5_germany_2017_2024_merged.nc"
    ds = xr.open_dataset(filepath).load()
except FileNotFoundError:
    print(f"ERROR: The file was not found at '{filepath}'")
    exit()

print("--- Data loaded successfully. Generating corrected analysis plots... ---")

# --- Step 2: Create a multi-plot dashboard ---
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('AgERA5 Data Quality & Analysis Dashboard (2017-2024)', fontsize=20, y=0.95)

# --- PLOT 1: Time Series of Mean Temperature (Germany Average) ---
ax1 = axes[0, 0]
german_avg_temp_k = ds['Temperature_Air_2m_Mean_24h'].mean(dim=['lat', 'lon'])
# --- CHANGE: Convert from Kelvin to Celsius ---
german_avg_temp_c = german_avg_temp_k - 273.15
german_avg_temp_c.plot(ax=ax1, color='blue')
ax1.set_title('A) Daily Mean Temperature (Average over Germany)', fontsize=14)
ax1.set_xlabel('Year')
# --- CHANGE: Label is now correct ---
ax1.set_ylabel('Temperature (°C)')
ax1.grid(True, linestyle='--', alpha=0.6)

# --- PLOT 2: Time Series of Total Precipitation (Germany Average) ---
ax2 = axes[0, 1]
german_avg_precip = ds['Precipitation_Flux'].mean(dim=['lat', 'lon'])
german_avg_precip.plot(ax=ax2, color='green')
ax2.set_title('B) Daily Precipitation Flux (Average over Germany)', fontsize=14)
ax2.set_xlabel('Year')
ax2.set_ylabel('Precipitation Flux (kg m⁻² s⁻¹)')
ax2.grid(True, linestyle='--', alpha=0.6)

# --- PLOT 3: Spatial Map of Long-Term Average Temperature ---
ax3 = axes[1, 0]
long_term_mean_temp_k = ds['Temperature_Air_2m_Mean_24h'].mean(dim='time')
# --- CHANGE: Convert from Kelvin to Celsius ---
long_term_mean_temp_c = long_term_mean_temp_k - 273.15
# --- CHANGE: Label is now correct ---
long_term_mean_temp_c.plot(ax=ax3, cmap='coolwarm', cbar_kwargs={'label': 'Avg. Temperature (°C)'})
ax3.set_title('C) Long-Term Mean Temperature (2017-2024)', fontsize=14)
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')

# --- PLOT 4: Histogram of Precipitation Values ---
ax4 = axes[1, 1]
precip_values = ds['Precipitation_Flux'].values.flatten()
precip_values_rain_only = precip_values[precip_values > 0]
ax4.hist(precip_values_rain_only, bins=100, color='skyblue', log=True)
ax4.set_title('D) Distribution of Precipitation Values (Rainy Days)', fontsize=14)
ax4.set_xlabel('Precipitation Flux (kg m⁻² s⁻¹)')
ax4.set_ylabel('Frequency (Log Scale)')
ax4.grid(True, linestyle='--', alpha=0.6)

# --- Finalize and Save ---
plt.tight_layout(rect=[0, 0, 1, 0.95])
plot_filename = "agera5_analysis_dashboard_celsius.png"
plt.savefig(plot_filename, dpi=150)

print(f"\nSUCCESS: Corrected dashboard saved to '{plot_filename}'.")
print("Temperature is now correctly displayed in Celsius.")

ds.close()