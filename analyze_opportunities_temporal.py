import pandas as pd
import numpy as np
from pathlib import Path

# Load BTCUSD data
data_path = Path('data/featured_data/featured_BTCUSD_data.csv')
df = pd.read_csv(data_path)

print(f"Total rows: {len(df)}")
print(f"Total real opportunities (>= 0.3%): {(df['is_real_opportunity'] >= 1).sum()}\n")

# Divide data into 10 equal chunks and analyze opportunity distribution
chunk_size = len(df) // 10
print("Opportunity distribution by decile (10% chunks):")
print("-" * 60)

for i in range(10):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i < 9 else len(df)
    
    chunk = df.iloc[start_idx:end_idx]
    n_opps = (chunk['is_real_opportunity'] >= 1).sum()
    pct = (n_opps / len(chunk)) * 100
    
    print(f"Decile {i+1:2d} (rows {start_idx:6d}-{end_idx:6d}): {n_opps:5d} opps ({pct:6.2f}%)")

print("\n" + "=" * 60)
print("First 20% vs Last 80%:")
split_idx = int(len(df) * 0.2)
first_20 = df.iloc[:split_idx]
last_80 = df.iloc[split_idx:]

opps_first_20 = (first_20['is_real_opportunity'] >= 1).sum()
opps_last_80 = (last_80['is_real_opportunity'] >= 1).sum()

pct_first_20 = (opps_first_20 / len(first_20)) * 100
pct_last_80 = (opps_last_80 / len(last_80)) * 100

print(f"First 20%: {opps_first_20:5d} opps / {len(first_20):6d} rows = {pct_first_20:6.2f}%")
print(f"Last 80%:  {opps_last_80:5d} opps / {len(last_80):6d} rows = {pct_last_80:6.2f}%")
print(f"Ratio (First 20% / Last 80%): {pct_first_20 / pct_last_80:.2f}x")
