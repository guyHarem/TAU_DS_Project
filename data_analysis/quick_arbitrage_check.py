import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== QUICK ARBITRAGE OPPORTUNITY CHECK ===\n")

# Load BTC data
data_path = '../data'
btc = pd.read_csv(f'{data_path}/combined_BTCUSD_data.csv')
btc['time'] = pd.to_datetime(btc['time'])

print(f"Data loaded: {len(btc)} rows")
print(f"Time range: {btc['time'].min()} to {btc['time'].max()}\n")

# Get all close price columns
exchanges = ["BINANCE", "BITFINEX", "COINBASE", "GATEIO", "MEXC", "KRAKEN"]
close_cols = [f"{ex}:close" for ex in exchanges]

# Check which exchanges have data
print("Data availability:")
for col in close_cols:
    if col in btc.columns:
        non_null = btc[col].notna().sum()
        pct = (non_null / len(btc)) * 100
        print(f"  {col}: {non_null}/{len(btc)} ({pct:.1f}%)")
    else:
        print(f"  {col}: NOT FOUND")

# Filter to only rows where we have data from multiple exchanges
btc_analysis = btc[['time'] + [col for col in close_cols if col in btc.columns]].copy()
available_cols = [col for col in close_cols if col in btc.columns]

# Drop rows where we don't have at least 2 exchanges
btc_analysis = btc_analysis.dropna(thresh=3)  # At least time + 2 exchanges
print(f"\nRows with data from at least 2 exchanges: {len(btc_analysis)}")

if len(btc_analysis) == 0:
    print("ERROR: No rows with multiple exchanges. Cannot calculate arbitrage.")
    exit()

# Calculate min and max prices across exchanges
btc_analysis['min_price'] = btc_analysis[available_cols].min(axis=1)
btc_analysis['max_price'] = btc_analysis[available_cols].max(axis=1)
btc_analysis['spread'] = btc_analysis['max_price'] - btc_analysis['min_price']
btc_analysis['spread_pct'] = (btc_analysis['spread'] / btc_analysis['min_price']) * 100

# Identify which exchanges have min/max
btc_analysis['buy_exchange'] = btc_analysis[available_cols].idxmin(axis=1).str.split(':').str[0]
btc_analysis['sell_exchange'] = btc_analysis[available_cols].idxmax(axis=1).str.split(':').str[0]

# Statistics
print("\n" + "="*70)
print("SPREAD STATISTICS")
print("="*70)
print(f"Mean spread: {btc_analysis['spread_pct'].mean():.4f}%")
print(f"Median spread: {btc_analysis['spread_pct'].median():.4f}%")
print(f"Std dev: {btc_analysis['spread_pct'].std():.4f}%")
print(f"Min spread: {btc_analysis['spread_pct'].min():.4f}%")
print(f"Max spread: {btc_analysis['spread_pct'].max():.4f}%")

print("\nPercentiles:")
for p in [50, 75, 90, 95, 99]:
    val = btc_analysis['spread_pct'].quantile(p/100)
    print(f"  {p}th: {val:.4f}%")

# Trading costs
trading_fee = 0.15 * 2      # 0.30%
transfer_fee = 0.05         # 0.05%
slippage = 0.10             # 0.10%
time_risk = 0.05            # 0.05%
total_cost = trading_fee + transfer_fee + slippage + time_risk  # 0.50%

# Safety margin for "real opportunities"
safety_margin = 0.10        # 0.10% extra buffer
real_opportunity_threshold = total_cost + safety_margin  # 0.60%

print(f"\nEstimated trading costs: {total_cost}%")
print(f"Real opportunity threshold (with safety margin): {real_opportunity_threshold}%")

# Find opportunities (any spread above cost)
opportunities = btc_analysis[btc_analysis['spread_pct'] >= total_cost].copy()

# Find REAL opportunities (spread above cost + margin)
real_opportunities = btc_analysis[btc_analysis['spread_pct'] >= real_opportunity_threshold].copy()

print("\n" + "="*70)
print(f"OPPORTUNITIES (spread >= {total_cost}%)")
print("="*70)
print(f"Count: {len(opportunities)}")
print(f"Percentage of all data: {len(opportunities)/len(btc_analysis)*100:.2f}%")

if len(opportunities) > 0:
    opportunities['profit_per_btc'] = (opportunities['spread_pct'] - total_cost) * opportunities['min_price'] / 100
    print(f"\nIf trading 1 BTC per opportunity:")
    print(f"  Total potential profit: ${opportunities['profit_per_btc'].sum():.2f}")
    print(f"  Average per trade: ${opportunities['profit_per_btc'].mean():.2f}")

print("\n" + "="*70)
print(f"REAL OPPORTUNITIES (spread >= {real_opportunity_threshold}%)")
print("="*70)
print(f"Count: {len(real_opportunities)}")
print(f"Percentage of all data: {len(real_opportunities)/len(btc_analysis)*100:.2f}%")
if len(opportunities) > 0:
    print(f"Quality ratio: {len(real_opportunities)/len(opportunities)*100:.1f}% of opportunities are 'real'")

if len(real_opportunities) > 0:
    print(f"\nTop 10 REAL opportunities:")
    top10 = real_opportunities.nlargest(10, 'spread_pct')[['time', 'buy_exchange', 'sell_exchange', 'min_price', 'max_price', 'spread_pct']]
    print(top10.to_string(index=False))
    
    # Most common exchange pairs
    print(f"\nMost common exchange pairs for REAL opportunities:")
    pairs = real_opportunities['buy_exchange'] + ' → ' + real_opportunities['sell_exchange']
    print(pairs.value_counts().head(10))
    
    # Calculate potential profit for REAL opportunities
    real_opportunities['profit_per_btc'] = (real_opportunities['spread_pct'] - total_cost) * real_opportunities['min_price'] / 100
    print(f"\nIf trading 1 BTC per REAL opportunity:")
    print(f"  Total potential profit: ${real_opportunities['profit_per_btc'].sum():.2f}")
    print(f"  Average per trade: ${real_opportunities['profit_per_btc'].mean():.2f}")
    print(f"  Best trade: ${real_opportunities['profit_per_btc'].max():.2f}")
else:
    print("\n⚠️  NO REAL OPPORTUNITIES FOUND (above {real_opportunity_threshold}%)")

# Visualizations
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Spread over time
ax1 = axes[0]
ax1.plot(btc_analysis['time'], btc_analysis['spread_pct'], linewidth=0.5, alpha=0.7)
ax1.axhline(y=total_cost, color='orange', linestyle='--', linewidth=1.5, label=f'Min threshold ({total_cost}%)')
ax1.axhline(y=real_opportunity_threshold, color='red', linestyle='--', linewidth=2, label=f'Real opportunity ({real_opportunity_threshold}%)')
ax1.fill_between(btc_analysis['time'], 0, total_cost, alpha=0.2, color='red', label='Unprofitable zone')
ax1.set_xlabel('Time')
ax1.set_ylabel('Spread (%)')
ax1.set_title('BTC/USD Price Spread Across Exchanges Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution of spreads
ax2 = axes[1]
ax2.hist(btc_analysis['spread_pct'], bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(x=total_cost, color='orange', linestyle='--', linewidth=1.5, label=f'Min threshold ({total_cost}%)')
ax2.axvline(x=real_opportunity_threshold, color='red', linestyle='--', linewidth=2, label=f'Real opportunity ({real_opportunity_threshold}%)')
ax2.set_xlabel('Spread (%)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Price Spreads')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Exchange pair frequency (for REAL opportunities)
ax3 = axes[2]
if len(real_opportunities) > 0:
    pairs = real_opportunities['buy_exchange'] + ' → ' + real_opportunities['sell_exchange']
    pair_counts = pairs.value_counts().head(10)
    pair_counts.plot(kind='barh', ax=ax3)
    ax3.set_xlabel('Count')
    ax3.set_title('Most Common REAL Opportunity Exchange Pairs')
else:
    ax3.text(0.5, 0.5, 'No real opportunities found', 
             ha='center', va='center', fontsize=14)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quick_arbitrage_check.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Plots saved to: quick_arbitrage_check.png")
plt.show()

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
if len(real_opportunities) > 0:
    print("✅ YES - Real arbitrage opportunities exist in this data!")
    print(f"   Found {len(real_opportunities)} real opportunities across {(btc_analysis['time'].max() - btc_analysis['time'].min()).total_seconds()/3600:.1f} hours")
    print(f"   ({len(opportunities)} total opportunities above break-even)")
elif len(opportunities) > 0:
    print("⚠️  Marginal opportunities exist, but profits are very thin")
    print(f"   Found {len(opportunities)} opportunities at break-even ({total_cost}%)")
    print(f"   But 0 with comfortable margin (>{real_opportunity_threshold}%)")
else:
    print("❌ NO - No profitable arbitrage opportunities found")
    print(f"   Max spread: {btc_analysis['spread_pct'].max():.4f}%")
    print(f"   Trading costs: {total_cost}%")
    print("   Spreads are too small to overcome trading costs")