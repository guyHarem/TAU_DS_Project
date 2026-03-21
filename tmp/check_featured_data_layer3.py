#!/usr/bin/env python3
"""
Check featured data after Layer 3 for data quality issues
"""

import pandas as pd
import numpy as np
from pathlib import Path

FEATURED_DATA_PATH = Path("/Users/guyharem/TAU/Projects/TAU_DS_Project/data/featured_data")
CSV_FILES = list(FEATURED_DATA_PATH.glob("featured_*.csv"))

print("=" * 100)
print("POST-LAYER 3 DATA QUALITY CHECK")
print("=" * 100)

for csv_file in sorted(CSV_FILES):
    coin = csv_file.stem.replace("featured_", "").replace("_data", "")
    df = pd.read_csv(csv_file)
    
    print(f"\n{'='*100}")
    print(f"COIN: {coin}")
    print(f"{'='*100}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # 1. Total empty cells
    total_cells = df.shape[0] * df.shape[1]
    empty_cells = df.isna().sum().sum()
    empty_pct = (empty_cells / total_cells) * 100
    
    print(f"\nEmpty cells: {empty_cells:,} / {total_cells:,} ({empty_pct:.2f}%)")
    
    # 2. Columns with nulls
    null_by_column = df.isna().sum()
    columns_with_nulls = null_by_column[null_by_column > 0].sort_values(ascending=False)
    
    if len(columns_with_nulls) > 0:
        print(f"\nColumns with missing data ({len(columns_with_nulls)} columns):")
        for col, count in columns_with_nulls.items():
            null_pct = (count / df.shape[0]) * 100
            print(f"  - {col:40} {count:6,} nulls ({null_pct:5.2f}%)")
    else:
        print("\n✓ No null values found!")
    
    # 3. Check specific Layer 3 columns
    print(f"\nLayer 3 Features Status:")
    layer3_features = [
        'spread_close_absolute', 'spread_close_pct',
        'spread_highlow_absolute', 'spread_highlow_pct',
        'volume_buy_exchange', 'volume_sell_exchange',
        'price_ratio_buy_sell',
        'price_position_buy_exchange', 'price_position_sell_exchange',
        'volatility_avg', 'volatility_max', 'volatility_min'
    ]
    
    for feat in layer3_features:
        if feat in df.columns:
            nulls = df[feat].isna().sum()
            null_pct = (nulls / len(df)) * 100
            status = "✓" if nulls == 0 else "⚠"
            print(f"  {status} {feat:40} {nulls:6,} nulls ({null_pct:5.2f}%)")
    
    # 4. Check for INF values
    print(f"\nInfinite values:")
    inf_count = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_in_col = np.isinf(df[col]).sum()
        if inf_in_col > 0:
            print(f"  ⚠ {col:40} {inf_in_col:6,} INF values")
            inf_count += inf_in_col
    
    if inf_count == 0:
        print("  ✓ No infinite values found!")
    
    # 5. Check for all-zero columns
    print(f"\nColumns with unusual distributions:")
    all_zero_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if (df[col] == 0).sum() == len(df):
            all_zero_cols.append(col)
            print(f"  ⚠ {col:40} ALL ZEROS")
    
    if not all_zero_cols:
        print("  ✓ No all-zero numeric columns")
    
    # 6. Summary stats on spread features
    if 'spread_close_pct' in df.columns:
        spread = df['spread_close_pct'].dropna()
        print(f"\nSpread stats (spread_close_pct):")
        print(f"  Min: {spread.min():.6f}%")
        print(f"  Max: {spread.max():.6f}%")
        print(f"  Mean: {spread.mean():.6f}%")
        print(f"  Median: {spread.median():.6f}%")
        print(f"  Std: {spread.std():.6f}%")
    
    # 7. Check buy/sell exchange consistency
    if 'buy_exchange' in df.columns and 'sell_exchange' in df.columns:
        same_exchange = (df['buy_exchange'] == df['sell_exchange']).sum()
        same_pct = (same_exchange / len(df)) * 100
        print(f"\nBuy/Sell exchange consistency:")
        print(f"  Same exchange: {same_exchange:,} rows ({same_pct:.2f}%)")
    
    # 8. Overall quality assessment
    print(f"\n{'─'*100}")
    print("QUALITY ASSESSMENT:")
    print(f"{'─'*100}")
    
    issues = []
    
    if empty_pct > 10:
        issues.append(f"  ⚠ HIGH missing data: {empty_pct:.2f}%")
    elif empty_pct > 5:
        issues.append(f"  ⚠ MODERATE missing data: {empty_pct:.2f}%")
    
    if inf_count > 0:
        issues.append(f"  ⚠ Found {inf_count} infinite values")
    
    if len(columns_with_nulls) > 5:
        issues.append(f"  ⚠ Many columns with nulls: {len(columns_with_nulls)} columns")
    
    if not issues:
        print("  ✓ Data quality looks good!")
    else:
        for issue in issues:
            print(issue)

print("\n" + "=" * 100)
print("END OF REPORT")
print("=" * 100)
