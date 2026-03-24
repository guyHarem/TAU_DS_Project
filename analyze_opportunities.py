import pandas as pd
import numpy as np
from pathlib import Path

# Load BTCUSD data
data_path = Path('data/featured_data/featured_BTCUSD_data.csv')
df = pd.read_csv(data_path)

print(f"Total rows: {len(df)}")
print(f"\nSpread statistics:")
print(f"  Min: {df['spread_close_pct'].min():.6f}")
print(f"  Max: {df['spread_close_pct'].max():.6f}")
print(f"  Mean: {df['spread_close_pct'].mean():.6f}")
print(f"  Median: {df['spread_close_pct'].median():.6f}")
print(f"  Std: {df['spread_close_pct'].std():.6f}")

# Opportunities at different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

print(f"\nOpportunities by threshold:")
for threshold in thresholds:
    opportunities = (df['spread_close_pct'] >= threshold).sum()
    percentage = (opportunities / len(df)) * 100
    print(f"  >= {threshold}%: {opportunities} ({percentage:.2f}%)")

# Check if there's is_real_opportunity column
if 'is_real_opportunity' in df.columns:
    real_opps = df['is_real_opportunity'].sum()
    real_opp_pct = (real_opps / len(df)) * 100
    print(f"\nis_real_opportunity: {real_opps} ({real_opp_pct:.2f}%)")

# Check is_opportunity column
if 'is_opportunity' in df.columns:
    opps = df['is_opportunity'].sum()
    opp_pct = (opps / len(df)) * 100
    print(f"is_opportunity: {opps} ({opp_pct:.2f}%)")
