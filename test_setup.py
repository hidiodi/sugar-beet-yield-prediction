import pandas as pd
import numpy as np
from pathlib import Path
import sys
import shutil
import importlib

# Setup
project_root = Path('.').resolve()
sys.path.insert(0, str(project_root))

from src import config as global_config
analysis_config = importlib.import_module("src.03_analysis.config")

CONFIG = analysis_config.MODEL_COMPARISON_CONFIG
output_dir = Path(CONFIG['OUTPUT_DIR'])
output_dir.mkdir(parents=True, exist_ok=True)
input_file = output_dir / 'super_ensemble_training_data.csv'

# Create Mock Data
years = list(range(2000, 2025))
districts = [1001, 1002, 1003]
data = []

np.random.seed(42)

for y in years:
    for d in districts:
        trend = 50 + (y - 2000) * 0.5 + np.random.normal(0, 2)
        trend = float(trend)
        actual = trend + np.random.normal(0, 5) # Actual yield with noise
        actual = float(actual)

        # Component models
        # Hybrid XGB is good
        hybrid = actual + np.random.normal(0, 3)
        # Robust Linear is good too
        robust = actual + np.random.normal(0, 2.5)
        # V31 Solar Gated is okay
        v31 = actual + np.random.normal(0, 4)

        # Signals (Model - Trend)
        sig_hybrid = hybrid - trend
        sig_robust = robust - trend
        sig_v31 = v31 - trend

        # Context features
        casdi = np.random.randint(0, 10)
        nmsd = np.random.randint(0, 5)

        row = {
            'year': y,
            'district_no': d,
            'kreisYield': actual,
            'Statistical_Trend_pred': trend,
            'V31_Solar_Gated_pred': v31,
            'Hybrid_XGB_pred': hybrid,
            'Robust_Linear_pred': robust,
            'Signal_V31_Solar_Gated': sig_v31,
            'Signal_Hybrid_XGB': sig_hybrid,
            'Signal_Robust_Linear': sig_robust,
            'CASDI_Phase2_Count': casdi,
            'NMSD_Phase2_Count': nmsd,
            'latitude': 50.0,
            'longitude': 10.0,
            # Extra columns that might be there but should be ignored
            'Best_Model': 'Robust Linear',
            'Oracle_Error': 1.0,
            'Error_Statistical_Trend': abs(actual - trend),
            'Error_V31_Solar_Gated': abs(actual - v31),
            'Error_Hybrid_XGB': abs(actual - hybrid),
            'Error_Robust_Linear': abs(actual - robust),
            # Add some columns that might be there but should be ignored
            'Predicted_Model': 'Robust Linear',
            'Switch_Prediction': 0.0,
            'Target_Encoded': 0,
            'Is_Garbage_Data': 0,
            'Raw_Bias': 0.0,
            'Regret_Weight': 1.0,
            'Median_Error': 1.0
        }
        data.append(row)

df = pd.DataFrame(data)
df.to_csv(input_file, index=False)
print(f"Created mock data at {input_file} with shape {df.shape}")
print(f"Columns: {df.columns.tolist()}")
