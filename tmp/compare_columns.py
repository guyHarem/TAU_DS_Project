#!/usr/bin/env python3
"""
Compare column differences between featured datasets
"""

import pandas as pd
from pathlib import Path

FEATURED_DATA_PATH = Path("/Users/guyharem/TAU/Projects/TAU_DS_Project/data/featured_data")

csv_files = {
    'BTCUSD': FEATURED_DATA_PATH / "featured_BTCUSD_data.csv",
    'ETHUSD': FEATURED_DATA_PATH / "featured_ETHUSD_data.csv",
    'DOGEUSD': FEATURED_DATA_PATH / "featured_DOGEUSD_data.csv",
    'LINKUSD': FEATURED_DATA_PATH / "featured_LINKUSD_data.csv",
    'SOLUSD': FEATURED_DATA_PATH / "featured_SOLUSD_data.csv",
    'XRPUSD': FEATURED_DATA_PATH / "featured_XRPUSD_data.csv",
}

# Load all datasets
datasets = {name: pd.read_csv(path) for name, path in csv_files.items()}

# Get all columns
all_columns = {}
for coin, df in datasets.items():
    all_columns[coin] = set(df.columns)
    print(f"{coin}: {len(df.columns)} columns")

print("\n" + "="*80)
print("COMPARING BTCUSD (67 cols) vs XRPUSD (60 cols)")
print("="*80)

btc_cols = all_columns['BTCUSD']
xrp_cols = all_columns['XRPUSD']

missing_in_xrp = btc_cols - xrp_cols
extra_in_xrp = xrp_cols - btc_cols

if missing_in_xrp:
    print(f"\n❌ MISSING in XRPUSD ({len(missing_in_xrp)} columns):")
    for col in sorted(missing_in_xrp):
        print(f"  - {col}")
else:
    print("\n✓ No missing columns in XRPUSD")

if extra_in_xrp:
    print(f"\n✓ EXTRA in XRPUSD ({len(extra_in_xrp)} columns):")
    for col in sorted(extra_in_xrp):
        print(f"  + {col}")
else:
    print("\n✓ No extra columns in XRPUSD")

print("\n" + "="*80)
print("ANALYSIS: Missing columns are from which exchange?")
print("="*80)

# Group by exchange
missing_by_exchange = {}
for col in missing_in_xrp:
    for exchange in ["BINANCE", "BITFINEX", "COINBASE", "GATEIO", "KRAKEN"]:
        if exchange in col:
            if exchange not in missing_by_exchange:
                missing_by_exchange[exchange] = []
            missing_by_exchange[exchange].append(col)
            break

for exchange in ["BINANCE", "BITFINEX", "COINBASE", "GATEIO", "KRAKEN"]:
    if exchange in missing_by_exchange:
        print(f"\n{exchange}: {len(missing_by_exchange[exchange])} missing columns")
        for col in sorted(missing_by_exchange[exchange]):
            print(f"  - {col}")
    else:
        print(f"\n✓ {exchange}: All columns present")

# Check raw data
print("\n" + "="*80)
print("VERIFYING: Checking raw data for exchange availability")
print("="*80)

RAW_DATA_PATH = Path("/Users/guyharem/TAU/Projects/TAU_DS_Project/data/raw_data")

xrp_raw = pd.read_csv(RAW_DATA_PATH / "combined_XRPUSD_data.csv")
print(f"\nXRPUSD raw data columns ({len(xrp_raw.columns)}):")
for col in sorted(xrp_raw.columns):
    print(f"  - {col}")

# Check which exchanges have data
print(f"\nExchange availability in XRPUSD raw data:")
exchanges_in_raw = set()
for col in xrp_raw.columns:
    for exchange in ["BINANCE", "BITFINEX", "COINBASE", "GATEIO", "KRAKEN"]:
        if exchange in col:
            exchanges_in_raw.add(exchange)
            break

for exchange in ["BINANCE", "BITFINEX", "COINBASE", "GATEIO", "KRAKEN"]:
    if exchange in exchanges_in_raw:
        print(f"  ✓ {exchange}: Present")
    else:
        print(f"  ❌ {exchange}: MISSING")
