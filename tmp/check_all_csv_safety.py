#!/usr/bin/env python3
"""
Comprehensive safety check for all featured CSV files:
- Empty cells and missing values
- Zero values in critical columns
- Exchange availability per row
- Volume safety verification
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Featured data files to check
FEATURED_DATA_PATH = Path("/Users/guyharem/TAU/Projects/TAU_DS_Project/data/featured_data")
CSV_FILES = list(FEATURED_DATA_PATH.glob("featured_*.csv"))

print("=" * 80)
print("COMPREHENSIVE DATA SAFETY CHECK - ALL FEATURED CSVs")
print("=" * 80)
print()

# Track summary statistics
summary_stats = {}

for csv_file in sorted(CSV_FILES):
    coin = csv_file.stem.replace("featured_", "").replace("_data", "")
    df = pd.read_csv(csv_file)
    
    print(f"\n{'='*80}")
    print(f"COIN: {coin}")
    print(f"{'='*80}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # 1. COUNT EMPTY/NULL CELLS
    total_cells = df.shape[0] * df.shape[1]
    empty_cells = df.isna().sum().sum()
    empty_pct = (empty_cells / total_cells) * 100
    print(f"\nEmpty cells: {empty_cells:,} / {total_cells:,} ({empty_pct:.2f}%)")
    
    # 2. IDENTIFY WHICH COLUMNS HAVE NULLS
    null_by_column = df.isna().sum()
    columns_with_nulls = null_by_column[null_by_column > 0]
    if len(columns_with_nulls) > 0:
        print(f"Columns with nulls:")
        for col, count in columns_with_nulls.items():
            null_pct = (count / df.shape[0]) * 100
            print(f"  - {col:30} {count:6,} nulls ({null_pct:5.2f}%)")
    else:
        print("No null values found!")
    
    # 3. COUNT ZERO VALUES
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_values = (df[numeric_cols] == 0).sum().sum()
    print(f"\nZero values: {zero_values:,}")
    
    # 4. CHECK EXCHANGE AVAILABILITY
    # Count how many exchange columns have non-null values per row
    exchange_cols = [col for col in df.columns if ':' in col]
    price_exchanges = {}
    for col in exchange_cols:
        exchange = col.split(':')[0]
        if exchange not in price_exchanges:
            price_exchanges[exchange] = []
        price_exchanges[exchange].append(col)
    
    print(f"\nExchange availability per row:")
    exchange_availability = {}
    for exchange, cols in price_exchanges.items():
        available = ((df[cols].notna()).sum(axis=1) > 0).sum()
        available_pct = (available / len(df)) * 100
        exchange_availability[exchange] = (available, available_pct)
        print(f"  - {exchange:12} {available:6,} rows available ({available_pct:5.2f}%)")
    
    # Distribution of exchanges per row
    num_exchanges_per_row = ((df[exchange_cols].notna()).sum(axis=1) / len(price_exchanges)).astype(int)
    print(f"\nExchanges per row distribution:")
    for n_exch in sorted(num_exchanges_per_row.unique()):
        count = (num_exchanges_per_row == n_exch).sum()
        pct = (count / len(df)) * 100
        print(f"  - {n_exch} exchanges: {count:6,} rows ({pct:5.2f}%)")
    
    min_exchanges = num_exchanges_per_row.min()
    max_exchanges = num_exchanges_per_row.max()
    print(f"Range: {min_exchanges} to {max_exchanges} exchanges per row")
    
    # 5. VERIFY NO ZERO VOLUMES
    volume_cols = [col for col in df.columns if col.endswith(':volume')]
    zero_volumes = 0
    for col in volume_cols:
        zero_volumes += (df[col] == 0).sum()
    print(f"\nZero volumes: {zero_volumes} (across {len(volume_cols)} volume columns)")
    
    # 6. SAFETY ASSESSMENT
    print(f"\n{'─'*80}")
    print("SAFETY ASSESSMENT:")
    print(f"{'─'*80}")
    
    issues = []
    
    # Check for problematic nulls
    if 'buy_exchange' in df.columns and df['buy_exchange'].isna().sum() > 0:
        issues.append(f"  ⚠ {df['buy_exchange'].isna().sum()} null values in buy_exchange")
    
    if 'sell_exchange' in df.columns and df['sell_exchange'].isna().sum() > 0:
        issues.append(f"  ⚠ {df['sell_exchange'].isna().sum()} null values in sell_exchange")
    
    # Check for zero volumes in any exchange
    if zero_volumes > 0:
        issues.append(f"  ⚠ Found {zero_volumes} zero volume values - check volume_ratio safety")
    
    # Check minimum exchanges
    if min_exchanges < 2:
        issues.append(f"  ⚠ Some rows have only {min_exchanges} exchange - may cause spread issues")
    
    if not issues:
        print("  ✓ No safety issues detected")
        print("  ✓ Buy/sell exchanges properly assigned")
        print("  ✓ No zero volumes (division-by-zero safe)")
        print("  ✓ Minimum 2 exchanges per row (spreads safe)")
    else:
        for issue in issues:
            print(issue)
    
    # Store summary
    summary_stats[coin] = {
        'rows': df.shape[0],
        'empty_cells_pct': empty_pct,
        'zero_values': zero_values,
        'min_exchanges': min_exchanges,
        'zero_volumes': zero_volumes,
        'healthy': len(issues) == 0
    }

# SUMMARY TABLE
print("\n" + "=" * 80)
print("SUMMARY - ALL COINS")
print("=" * 80)
print(f"{'Coin':<12} {'Rows':<8} {'Empty%':<10} {'Zeros':<8} {'Min_Exch':<10} {'Vol_0s':<8} {'Safe':<6}")
print("─" * 80)
for coin in sorted(summary_stats.keys()):
    stats = summary_stats[coin]
    safe_str = "✓" if stats['healthy'] else "✗"
    print(f"{coin:<12} {stats['rows']:<8} {stats['empty_cells_pct']:<10.2f} "
          f"{stats['zero_values']:<8} {stats['min_exchanges']:<10} "
          f"{stats['zero_volumes']:<8} {safe_str:<6}")

print("\n" + "=" * 80)
all_healthy = all(stats['healthy'] for stats in summary_stats.values())
if all_healthy:
    print("✓ ALL CSVs PASSED SAFETY CHECK - Ready for feature engineering")
else:
    print("✗ Some CSVs have safety concerns - Review above")
print("=" * 80)
