import pandas as pd
import numpy as np

print(" Loading the Lending Club dataset... (This might take a moment)")

# 1. Load the dataset
df = pd.read_csv('lc_loan.csv')

# 2. Print basic structural characteristics
print("\n=== DATASET SHAPE ===")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n=== DATA TYPES AND NON-NULL COUNTS ===")
df.info(verbose=True, show_counts=True)

print("\n=== TOP 15 MISSING VALUE PERCENTAGES ===")
missing_pct = (df.isnull().sum() / len(df)) * 100
# Filter to show only columns that have missing values, sorted highest to lowest
print(missing_pct[missing_pct > 0].sort_values(ascending=False).head(15))