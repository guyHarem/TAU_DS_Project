import pandas as pd
import numpy as np

# Load BTC data
data_path = '../data'
btc = pd.read_csv(f'{data_path}/combined_BTCUSD_data.csv')
btc['time'] = pd.to_datetime(btc['time'])

print("=== SPREAD DIAGNOSTIC ===\n")

# Get available exchanges
exchanges = ["BINANCE", "BITFINEX", "COINBASE", "GATEIO", "MEXC", "KRAKEN"]
close_cols = [f"{ex}:close" for ex in exchanges if f"{ex}:close" in btc.columns]
high_cols = [f"{ex}:high" for ex in exchanges if f"{ex}:high" in btc.columns]
low_cols = [f"{ex}:low" for ex in exchanges if f"{ex}:low" in btc.columns]

print(f"Available exchanges: {[col.split(':')[0] for col in close_cols]}\n")

# Get rows with data from multiple exchanges
btc_analysis = btc[['time'] + close_cols + high_cols + low_cols].dropna(thresh=3).copy()

print(f"Total rows with at least 2 exchanges: {len(btc_analysis)}\n")

# Calculate CLOSE-based spreads (realistic)
btc_analysis['min_close'] = btc_analysis[close_cols].min(axis=1)
btc_analysis['max_close'] = btc_analysis[close_cols].max(axis=1)
btc_analysis['spread_close'] = btc_analysis['max_close'] - btc_analysis['min_close']
btc_analysis['spread_close_pct'] = (btc_analysis['spread_close'] / btc_analysis['min_close']) * 100

# Calculate HIGH-LOW spreads (theoretical maximum)
btc_analysis['min_low'] = btc_analysis[low_cols].min(axis=1)
btc_analysis['max_high'] = btc_analysis[high_cols].max(axis=1)
btc_analysis['spread_highlow'] = btc_analysis['max_high'] - btc_analysis['min_low']
btc_analysis['spread_highlow_pct'] = (btc_analysis['spread_highlow'] / btc_analysis['min_low']) * 100

# Identify exchanges
btc_analysis['buy_exchange_close'] = btc_analysis[close_cols].idxmin(axis=1).str.split(':').str[0]
btc_analysis['sell_exchange_close'] = btc_analysis[close_cols].idxmax(axis=1).str.split(':').str[0]
btc_analysis['buy_exchange_highlow'] = btc_analysis[low_cols].idxmin(axis=1).str.split(':').str[0]
btc_analysis['sell_exchange_highlow'] = btc_analysis[high_cols].idxmax(axis=1).str.split(':').str[0]

print("="*70)
print("CLOSE-BASED SPREAD (REALISTIC)")
print("="*70)
print(btc_analysis['spread_close_pct'].describe())

print("\n" + "="*70)
print("HIGH-LOW SPREAD (THEORETICAL MAXIMUM)")
print("="*70)
print(btc_analysis['spread_highlow_pct'].describe())

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Average close spread: {btc_analysis['spread_close_pct'].mean():.4f}%")
print(f"Average high-low spread: {btc_analysis['spread_highlow_pct'].mean():.4f}%")
print(f"Difference (opportunity loss): {(btc_analysis['spread_highlow_pct'].mean() - btc_analysis['spread_close_pct'].mean()):.4f}%")
print(f"\nMax close spread: {btc_analysis['spread_close_pct'].max():.4f}%")
print(f"Max high-low spread: {btc_analysis['spread_highlow_pct'].max():.4f}%")

print("\n" + "="*70)
print("SAMPLE ROWS (first 5)")
print("="*70)
sample_cols = ['time', 'min_close', 'max_close', 'spread_close_pct', 
               'min_low', 'max_high', 'spread_highlow_pct',
               'buy_exchange_close', 'sell_exchange_close']
print(btc_analysis[sample_cols].head().to_string(index=False))

print("\n" + "="*70)
print("SPREAD PERCENTILES COMPARISON")
print("="*70)
print(f"{'Percentile':<12} {'Close Spread':<15} {'High-Low Spread':<15} {'Difference'}")
print("-" * 60)
for p in [10, 25, 50, 75, 90, 95, 99, 100]:
    close_val = btc_analysis['spread_close_pct'].quantile(p/100)
    highlow_val = btc_analysis['spread_highlow_pct'].quantile(p/100)
    diff = highlow_val - close_val
    print(f"{p:>3}th {'':<7} {close_val:<14.6f}% {highlow_val:<14.6f}% {diff:.6f}%")

print("\n" + "="*70)
print("THRESHOLD ANALYSIS - CLOSE SPREAD")
print("="*70)
thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
for threshold in thresholds:
    count = (btc_analysis['spread_close_pct'] >= threshold).sum()
    pct = (count / len(btc_analysis)) * 100
    print(f"Spread >= {threshold:.2f}%: {count:>4} rows ({pct:>5.2f}%)")

print("\n" + "="*70)
print("THRESHOLD ANALYSIS - HIGH-LOW SPREAD")
print("="*70)
for threshold in thresholds:
    count = (btc_analysis['spread_highlow_pct'] >= threshold).sum()
    pct = (count / len(btc_analysis)) * 100
    print(f"Spread >= {threshold:.2f}%: {count:>4} rows ({pct:>5.2f}%)")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
max_close_spread = btc_analysis['spread_close_pct'].max()
max_highlow_spread = btc_analysis['spread_highlow_pct'].max()

print(f"Maximum CLOSE spread: {max_close_spread:.4f}%")
print(f"Maximum HIGH-LOW spread: {max_highlow_spread:.4f}%")
print(f"Your current threshold: 0.50%")

if max_close_spread < 0.50:
    print("\n⚠️  PROBLEM: Your max CLOSE spread is LOWER than your trading cost threshold!")
    print("   This means NO arbitrage is possible with CLOSE prices.")
    print(f"\n   However, HIGH-LOW spread shows theoretical max of {max_highlow_spread:.4f}%")
    if max_highlow_spread >= 0.50:
        print("   ✅ HIGH-LOW spread exceeds threshold - opportunities exist in theory")
        print("   ⚠️  But they're not achievable with simple market orders")
        print("\nRecommendations:")
        print("   1. Use HIGH-LOW spread to identify volatile periods")
        print("   2. Use CLOSE spread for realistic profit calculations")
        print("   3. Consider limit orders to capture more of the HIGH-LOW spread")
    else:
        print("   ❌ Even HIGH-LOW spread doesn't exceed threshold")
        print("   These exchanges are too efficient for profitable arbitrage")
else:
    median_spread = btc_analysis['spread_close_pct'].median()
    print(f"\n✅ Arbitrage opportunities exist!")
    print(f"   Median CLOSE spread: {median_spread:.4f}%")
    count_profitable = (btc_analysis['spread_close_pct'] >= 0.50).sum()
    print(f"   Opportunities above 0.50%: {count_profitable} ({count_profitable/len(btc_analysis)*100:.1f}%)")
    
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("• CLOSE spread = What you can actually achieve with market orders")
print("• HIGH-LOW spread = Theoretical maximum if you time everything perfectly")
print("• Gap between them = What you lose due to timing and execution")
print("\nFor realistic analysis: Use CLOSE spread")
print("For opportunity detection: Use HIGH-LOW spread, but apply execution discount")