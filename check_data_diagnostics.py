"""
Comprehensive data diagnostics to identify issues for test fixes.

Run with: python3 check_data_diagnostics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "=" * 90)
print("DATA DIAGNOSTICS - ALL 7 CHECKS")
print("=" * 90)

# ============================================================================
# 1. BITFINEX ACROSS SYMBOLS
# ============================================================================
print("\n[1/7] CHECKING BITFINEX SPARSITY ACROSS SYMBOLS")
print("-" * 90)

data_dir = Path('data/featured_data')
csv_files = sorted(data_dir.glob('featured_*_data.csv'))[:5]

bitfinex_results = {}
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    symbol = csv_file.name.replace('featured_', '').replace('_data.csv', '')
    
    bitfinex_cols = [c for c in df.columns if 'BITFINEX' in c.upper()]
    if bitfinex_cols:
        missing_pct = df[bitfinex_cols[0]].isnull().sum() / len(df) * 100
        bitfinex_results[symbol] = missing_pct
        print(f"  {symbol}: {missing_pct:.1f}% missing (col: {bitfinex_cols[0]})")
    else:
        print(f"  {symbol}: No BITFINEX columns")

if bitfinex_results:
    avg_missing = sum(bitfinex_results.values()) / len(bitfinex_results)
    print(f"\n  → Average BITFINEX missing: {avg_missing:.1f}%")
    if avg_missing > 80:
        print(f"  ⚠️  ISSUE: BITFINEX is consistently sparse! Likely data collection problem.")
    elif any(v > 80 for v in bitfinex_results.values()):
        print(f"  ⚠️  ISSUE: BITFINEX sparse for some symbols. Check if exchange was unavailable.")

# ============================================================================
# 2. EXCHANGE PRICES
# ============================================================================
print("\n[2/7] CHECKING EXCHANGE PRICES (SOLUSD)")
print("-" * 90)

df = pd.read_csv(data_dir / 'featured_SOLUSD_data.csv')
exchanges = ['COINBASE', 'BINANCE', 'KRAKEN', 'BITFINEX']

print("  Exchange price ranges:")
exchange_prices = {}
for ex in exchanges:
    price_col = f'{ex}:open'
    if price_col in df.columns:
        valid = df[price_col].dropna()
        if len(valid) > 0:
            exchange_prices[ex] = {
                'min': valid.min(),
                'max': valid.max(),
                'mean': valid.mean(),
                'count': len(valid)
            }
            print(f"    {ex}: min={valid.min():.2f}, max={valid.max():.2f}, mean={valid.mean():.2f} (n={len(valid)})")
        else:
            print(f"    {ex}: All NaN")
    else:
        print(f"    {ex}: NO {price_col} column")

# Check for suspicious price differences
if len(exchange_prices) > 1:
    prices_mean = [v['mean'] for v in exchange_prices.values()]
    max_price = max(prices_mean)
    min_price = min(prices_mean)
    ratio = max_price / min_price if min_price > 0 else float('inf')
    print(f"\n  → Price ratio (max/min exchange mean): {ratio:.2f}")
    if ratio > 1.5:
        print(f"  ⚠️  WARNING: Large exchange price differences ({ratio:.2f}x). Check for data corruption.")
    else:
        print(f"  ✓ Exchange prices are consistent")

# ============================================================================
# 3. VOLUME VALUES
# ============================================================================
print("\n[3/7] CHECKING VOLUME VALUES")
print("-" * 90)

vol_cols = [c for c in df.columns if 'volume' in c.lower()][:8]
print(f"  Found {len(vol_cols)} volume columns. Checking first 8:")

volume_info = {}
for col in vol_cols:
    valid = df[col].dropna()
    if len(valid) > 0:
        has_neg = (valid < 0).sum()
        has_zero = (valid == 0).sum()
        volume_info[col] = {
            'min': valid.min(),
            'max': valid.max(),
            'mean': valid.mean(),
            'negatives': has_neg,
            'zeros': has_zero
        }
        print(f"    {col}:")
        print(f"      Range: [{valid.min():.6f}, {valid.max():.6f}]")
        print(f"      Negatives: {has_neg}, Zeros: {has_zero}")

if any(v['negatives'] > 0 for v in volume_info.values()):
    print(f"\n  ⚠️  ISSUE: Some volumes have negative values! Data corruption detected.")
elif any(v['max'] < 1 for v in volume_info.values()):
    print(f"\n  → Volumes are normalized (0-1 range)")
else:
    print(f"\n  → Volumes appear to be raw values")

# ============================================================================
# 4. SPREAD_ZSCORE_5 FEATURE
# ============================================================================
print("\n[4/7] CHECKING spread_zscore_5 FEATURE")
print("-" * 90)

if 'spread_zscore_5' in df.columns:
    non_null = df['spread_zscore_5'].notna().sum()
    null_count = df['spread_zscore_5'].isnull().sum()
    print(f"  Non-null values: {non_null}/{len(df)} ({100*non_null/len(df):.1f}%)")
    print(f"  Null values: {null_count}/{len(df)} ({100*null_count/len(df):.1f}%)")
    
    if non_null == 0:
        print(f"  ⚠️  ISSUE: spread_zscore_5 is entirely NaN! Feature is broken/useless.")
    elif non_null < len(df) * 0.1:
        print(f"  ⚠️  WARNING: spread_zscore_5 is >90% NaN. Should probably be removed.")
    else:
        valid = df['spread_zscore_5'].dropna()
        print(f"  Range: [{valid.min():.6f}, {valid.max():.6f}]")
        print(f"  ✓ Feature has data")
else:
    print(f"  Column not found")

# ============================================================================
# 5. PRICE RATIO (BINANCE vs COINBASE)
# ============================================================================
print("\n[5/7] CHECKING price_ratio_BINANCE_COINBASE")
print("-" * 90)

if 'price_ratio_BINANCE_COINBASE' in df.columns:
    ratio = df['price_ratio_BINANCE_COINBASE'].dropna()
    print(f"  Non-null values: {len(ratio)}/{len(df)}")
    print(f"  Range: [{ratio.min():.6f}, {ratio.max():.6f}]")
    print(f"  Mean: {ratio.mean():.6f}")
    print(f"  Std: {ratio.std():.6f}")
    
    extreme = (ratio > 2).sum() + (ratio < 0.5).sum()
    print(f"  Extreme ratios (>2 or <0.5): {extreme}/{len(ratio)} ({100*extreme/len(ratio):.1f}%)")
    
    if ratio.max() > 2:
        print(f"  ⚠️  WARNING: Very high ratio ({ratio.max():.2f}). Likely data issue.")
    else:
        print(f"  ✓ Ratios appear reasonable")
else:
    print(f"  Column not found")

# ============================================================================
# 6. OBJECT/CATEGORICAL COLUMNS
# ============================================================================
print("\n[6/7] CHECKING OBJECT/CATEGORICAL COLUMNS")
print("-" * 90)

object_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"  Found {len(object_cols)} object columns:")
for col in object_cols:
    unique = df[col].nunique()
    print(f"    {col}: {unique} unique values")
    if unique <= 10:
        print(f"      Values: {df[col].unique()[:5]}")

# ============================================================================
# 7. OVERALL DATA QUALITY
# ============================================================================
print("\n[7/7] OVERALL DATA QUALITY METRICS")
print("-" * 90)

print(f"  Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Missing data
total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
missing_pct = (total_missing / total_cells) * 100
print(f"  Total missing data: {total_missing} cells ({missing_pct:.2f}%)")

# Missing per column
missing_by_col = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
high_missing = missing_by_col[missing_by_col > 20]
if len(high_missing) > 0:
    print(f"\n  Columns with >20% missing:")
    for col, pct in high_missing.head(5).items():
        print(f"    {col}: {pct:.1f}%")

# Feature statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\n  Numeric features: {len(numeric_cols)}")
print(f"  Object features: {len(object_cols)}")

# Target variable
if 'spread_close_pct' in df.columns:
    target = df['spread_close_pct']
    print(f"\n  Target (spread_close_pct):")
    print(f"    Min: {target.min():.6f}")
    print(f"    Max: {target.max():.6f}")
    print(f"    Mean: {target.mean():.6f}")
    print(f"    Std: {target.std():.6f}")
    print(f"    Missing: {target.isnull().sum()}")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 90)
print("SUMMARY & RECOMMENDATIONS FOR TEST FIXES")
print("=" * 90)

issues = []

if bitfinex_results and sum(bitfinex_results.values())/len(bitfinex_results) > 80:
    issues.append("❌ BITFINEX: Consistently >80% missing - likely data collection issue")
    
if any(v['negatives'] > 0 for v in volume_info.values()):
    issues.append("❌ VOLUMES: Negative values detected - data corruption")
    
if 'spread_zscore_5' in df.columns and df['spread_zscore_5'].notna().sum() == 0:
    issues.append("❌ SPREAD_ZSCORE_5: Entirely NaN - feature is broken")

if len(issues) == 0:
    print("\n✓ No major data issues detected!")
    print("\nRecommendations for tests:")
    print("  1. ✅ Allow all object/categorical columns")
    print("  2. ✅ Increase missing data threshold to 15%")
    print("  3. ✅ Skip all-NaN features in validation")
    print("  4. ✅ Only validate specific price columns (not 'price_position_*')")
    print("  5. ✅ Accept normalized volumes or check accordingly")
    print("  6. ✅ Use 5x IQR for outlier detection (not 10x)")
    print("  7. ✅ Allow extreme price ratios across exchanges")
else:
    print("\n⚠️  ISSUES FOUND:\n")
    for issue in issues:
        print(f"  {issue}")
    print("\nREQUIRED ACTIONS:")
    print("  → Investigate data collection pipeline")
    print("  → Clean or remove problematic columns")

print("\n" + "=" * 90 + "\n")