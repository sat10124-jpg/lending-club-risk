import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 Loading preprocessed data for EDA...")
df = pd.read_csv('lc_loan_clean.csv')

# Set styling for clean corporate plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# --- INSIGHT 1: FICO Average vs. Early Default ---
print("📊 Generating FICO vs. Default visualization...")
plt.figure(figsize=(10, 6))

# Bin FICO scores into increments of 20 for a cleaner visualization
df['fico_bin'] = pd.cut(df['fico_average'], bins=np.arange(600, 850, 20))

# Calculate the default rate for each FICO bin
fico_defaults = df.groupby('fico_bin', observed=False)['early_default'].mean().reset_index()
fico_defaults['early_default'] *= 100 # Convert to percentage

sns.barplot(data=fico_defaults, x='fico_bin', y='early_default', hue='fico_bin', palette='RdYlGn_r', legend=False)
plt.title('Default Rate (%) by Borrower FICO Score Range')
plt.xlabel('FICO Score Range')
plt.ylabel('Default Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fico_vs_default.png', dpi=300)
plt.close()

# --- INSIGHT 2: Loan Purpose vs. Early Default ---
print("📊 Generating Loan Purpose vs. Default visualization...")
plt.figure(figsize=(12, 6))

# Calculate default rate by purpose and sort it
purpose_defaults = df.groupby('purpose')['early_default'].mean().reset_index()
purpose_defaults['early_default'] *= 100
purpose_defaults = purpose_defaults.sort_values(by='early_default', ascending=False)

sns.barplot(data=purpose_defaults, x='early_default', y='purpose', hue='purpose', palette='viridis', legend=False)
plt.title('Default Rate (%) by Loan Purpose')
plt.xlabel('Default Rate (%)')
plt.ylabel('Stated Loan Purpose')
plt.tight_layout()
plt.savefig('purpose_vs_default.png', dpi=300)
plt.close()

print("✨ Success! Two high-resolution charts saved: 'fico_vs_default.png' and 'purpose_vs_default.png'")