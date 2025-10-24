import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(f'data/03_primary/historical_residuals_2018_TEST.csv')

# Create a scatter plot
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df, x='lintul_yield', y='actual_yield')

# Add a y=x line. A perfect model would have all points on this line.
plt.plot([df['lintul_yield'].min(), df['lintul_yield'].max()],
         [df['lintul_yield'].min(), df['lintul_yield'].max()],
         'r--', lw=2, label='Perfect 1:1 Fit')

plt.title('Crop Model Performance for 2018 (using AgERA5 data)')
plt.xlabel('Simulated Yield (dt/ha) - LINTUL')
plt.ylabel('Actual Yield (dt/ha) - Observed')
plt.grid(True)
plt.legend()
plt.show()