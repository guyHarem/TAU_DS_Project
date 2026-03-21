#!/usr/bin/env python3
"""
Check for dangerous spread cases:
1. buy_exchange == sell_exchange (same exchange min/max)
2. min_close == max_close (all exchanges have same price - spread = 0)
3. Very small spreads (< threshold)
"""

import pandas as pd
import numpy as np
from pathlib import Path

FEATURED_DATA_PATH = Path("/Users/guyharem/TAU/Projects/TAU_DS_Project/data/featured_data")
CSV_FILES = list(FEATURED_DATA_PATH.glob("featured_*.csv"))

TRADING_COST_PCT = 0.2  # 0.2% trading cost
SAFETY_MARGIN_PCT = 0.1  # 0.1% safety margin
MIN_PROFITABLE_SPREAD = TRADING_COST_PCT + SAFETY_MARGIN_PCT  # 0.3%

print("=" * 90)
print("DANGEROUS SPREAD ANALYSIS - All Featured CSVs")
print("=" * 90)
print(f"\nThreshold for profitable arbitrage: {MIN_PROFITABLE_SPREAD}%")
print()

summary_stats = {}

for csv_file in sorted(CSV_FILES):
    coin = csv_file.stem.replace("featured_", "").replace("_data", "")
    df = pd.read_csv(csv_file)
    
    # Pre-calculate spread for all rows
    df['spread_close_pct_calc'] = (df['max_close'] - df['min_close']) / df['min_close'] * 100
    
    print(f"\n{'='*90}")
    print(f"COIN: {coin}")
    print(f"{'='*90}")
    
    # Check 1: Same exchange for buy and sell
    same_exchange = (df['buy_exchange'] == df['sell_exchange']).sum()
    same_exchange_pct = (same_exchange / len(df)) * 100
    
    print(f"\n1. SAME EXCHANGE (buy_exchange == sell_exchange)")
    print(f"   Count: {same_exchange:,} rows ({same_exchange_pct:.2f}%)")
    
    if same_exchange > 0:
        same_ex_df = df[df['buy_exchange'] == df['sell_exchange']]
        print(f"   Sample rows with same exchange:")
        print(f"   {same_ex_df[['min_close', 'max_close', 'buy_exchange', 'sell_exchange']].head().to_string()}")
        
        # Check their spreads
        spreads_when_same = same_ex_df['spread_close_pct_calc'].values
        print(f"   Spreads for these rows: min={spreads_when_same.min():.6f}%, max={spreads_when_same.max():.6f}%, mean={spreads_when_same.mean():.6f}%")
    
    # Check 2: min_close == max_close (spread = 0)
    zero_spread = (df['min_close'] == df['max_close']).sum()
    zero_spread_pct = (zero_spread / len(df)) * 100
    
    print(f"\n2. ZERO SPREAD (min_close == max_close)")
    print(f"   Count: {zero_spread:,} rows ({zero_spread_pct:.2f}%)")
    
    if zero_spread > 0:
        zero_spread_df = df[df['min_close'] == df['max_close']]
        print(f"   Sample rows with zero spread:")
        print(f"   {zero_spread_df[['min_close', 'max_close', 'buy_exchange', 'sell_exchange', 'num_exchanges_available']].head().to_string()}")
    
    # Check 3: Calculate potential spread pct (Layer 2 only)
    # Spread already calculated above as spread_close_pct_calc
    
    unprofitable = (df['spread_close_pct_calc'] < MIN_PROFITABLE_SPREAD).sum()
    unprofitable_pct = (unprofitable / len(df)) * 100
    
    print(f"\n3. UNPROFITABLE SPREADS (< {MIN_PROFITABLE_SPREAD}%)")
    print(f"   Count: {unprofitable:,} rows ({unprofitable_pct:.2f}%)")
    
    # Include zero spread in unprofitable
    if unprofitable > 0:
        unprofitable_df = df[df['spread_close_pct_calc'] < MIN_PROFITABLE_SPREAD]
        print(f"   Spread stats for unprofitable rows:")
        print(f"   Min: {unprofitable_df['spread_close_pct_calc'].min():.6f}%")
        print(f"   Max: {unprofitable_df['spread_close_pct_calc'].max():.6f}%")
        print(f"   Mean: {unprofitable_df['spread_close_pct_calc'].mean():.6f}%")
        print(f"   Median: {unprofitable_df['spread_close_pct_calc'].median():.6f}%")
        print(f"\n   Examples:")
        print(unprofitable_df[['spread_close_pct_calc', 'buy_exchange', 'sell_exchange', 'num_exchanges_available']].head(10).to_string())
    
    # Check 4: NaN spreads (impossible to calculate)
    nan_spread = df['spread_close_pct_calc'].isna().sum()
    nan_spread_pct = (nan_spread / len(df)) * 100
    
    print(f"\n4. NaN SPREADS (cannot calculate)")
    print(f"   Count: {nan_spread:,} rows ({nan_spread_pct:.2f}%)")
    
    if nan_spread > 0:
        nan_df = df[df['spread_close_pct_calc'].isna()]
        print(f"   Rows with NaN spreads - buy/sell exchanges:")
        print(nan_df[['buy_exchange', 'sell_exchange', 'num_exchanges_available']].value_counts().head().to_string())
    
    # Safety assessment
    print(f"\n{'─'*90}")
    print("SAFETY ASSESSMENT:")
    print(f"{'─'*90}")
    
    issues = []
    
    if same_exchange > 0:
        issues.append(f"  ⚠ {same_exchange:,} rows with same buy/sell exchange")
    
    if zero_spread > 0:
        issues.append(f"  ✗ {zero_spread:,} rows with ZERO spread (no arbitrage possible)")
    
    if unprofitable > 0:
        issues.append(f"  ⚠ {unprofitable:,} rows with unprofitable spreads (< {MIN_PROFITABLE_SPREAD}%)")
    
    if nan_spread > 0:
        issues.append(f"  ⚠ {nan_spread:,} rows with NaN spreads (cannot determine opportunity)")
    
    if not issues:
        print("  ✓ No dangerous spread cases detected")
    else:
        for issue in issues:
            print(issue)
    
    # Store summary
    summary_stats[coin] = {
        'same_exchange': same_exchange,
        'zero_spread': zero_spread,
        'unprofitable': unprofitable,
        'nan_spread': nan_spread,
        'has_issues': len(issues) > 0
    }

# Summary table
print("\n" + "=" * 90)
print("SUMMARY - ALL COINS")
print("=" * 90)
print(f"{'Coin':<12} {'Same_Exch':<12} {'Zero_Spread':<13} {'Unprofitable':<15} {'NaN_Spread':<12} {'Issues':<6}")
print("─" * 90)
for coin in sorted(summary_stats.keys()):
    stats = summary_stats[coin]
    issues_str = "✓" if not stats['has_issues'] else "✗"
    print(f"{coin:<12} {stats['same_exchange']:<12} {stats['zero_spread']:<13} "
          f"{stats['unprofitable']:<15} {stats['nan_spread']:<12} {issues_str:<6}")

print("\n" + "=" * 90)
all_safe = not any(stats['has_issues'] for stats in summary_stats.values())
if all_safe:
    print("✓ ALL CSVs SAFE - No dangerous spread cases detected")
else:
    print("✗ Some CSVs have dangerous spread cases - Review above")
print("=" * 90)
