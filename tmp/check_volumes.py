import pandas as pd
import numpy as np

file_path = 'data/featured_data/featured_BTCUSD_data.csv'
df = pd.read_csv(file_path)

print("="*80)
print("CURRENT STATE: Checking volume_ratio risk")
print("="*80)

print("\nRaw exchange volumes (will create volume_buy_exchange in Layer 3):\n")

for exchange in ['COINBASE', 'BINANCE', 'GATEIO']:
    vol_col = f'{exchange}:volume'
    if vol_col in df.columns:
        zero_count = (df[vol_col] == 0).sum()
        print(f"{exchange:12} volume zeros: {zero_count:5}")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

zero_volumes_exist = False
for exchange in ['COINBASE', 'BINANCE', 'GATEIO']:
    vol_col = f'{exchange}:volume'
    if vol_col in df.columns and (df[vol_col] == 0).sum() > 0:
        zero_volumes_exist = True

if not zero_volumes_exist:
    print("\n✓ NO ZERO VOLUMES - volume_ratio will be SAFE")
    print("  (no division by zero will occur)")
else:
    print("\n✗ Zero volumes exist - volume_ratio needs protection")
