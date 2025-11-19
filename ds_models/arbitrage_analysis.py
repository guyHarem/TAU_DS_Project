import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_data(csv_file):
    """Load the combined cryptocurrency data from CSV"""
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    return df

def analyze_arbitrage(df, exchange1='BINANCE', exchange2='BITFINEX'):
    """
    Analyze real arbitrage opportunities in the data
    """
    print("="*70)
    print(f"ARBITRAGE ANALYSIS: {exchange1} vs {exchange2}")
    print("="*70)
    
    # Get price columns
    col1_high = f'{exchange1}:high'
    col2_high = f'{exchange2}:high'
    col1_close = f'{exchange1}:close'
    col2_close = f'{exchange2}:close'
    
    # Filter rows with both exchanges
    df_filtered = df[['time', col1_high, col2_high, col1_close, col2_close]].dropna().copy()
    
    print(f"\nTotal data points: {len(df_filtered)}")
    print(f"Time range: {df_filtered['time'].min()} to {df_filtered['time'].max()}")
    
    # Calculate price differences
    df_filtered['price_diff'] = df_filtered[col1_close] - df_filtered[col2_close]
    df_filtered['price_diff_pct'] = (df_filtered['price_diff'] / df_filtered[col2_close]) * 100
    df_filtered['abs_price_diff_pct'] = abs(df_filtered['price_diff_pct'])
    
    # Trading cost assumptions
    trading_fee = 0.2  # 0.1% per trade x 2 trades
    slippage = 0.1
    total_cost = trading_fee + slippage
    
    print("\n" + "="*70)
    print("PRICE DIFFERENCE STATISTICS")
    print("="*70)
    print(f"Mean difference: {df_filtered['price_diff_pct'].mean():.4f}%")
    print(f"Std deviation: {df_filtered['price_diff_pct'].std():.4f}%")
    print(f"Min difference: {df_filtered['price_diff_pct'].min():.4f}%")
    print(f"Max difference: {df_filtered['price_diff_pct'].max():.4f}%")
    print(f"Median difference: {df_filtered['price_diff_pct'].median():.4f}%")
    
    print("\nPercentiles (absolute difference):")
    for p in [50, 75, 90, 95, 99, 99.9]:
        val = df_filtered['abs_price_diff_pct'].quantile(p/100)
        print(f"  {p}th percentile: {val:.4f}%")
    
    # Analyze profitable opportunities at different thresholds
    print("\n" + "="*70)
    print("PROFITABLE OPPORTUNITIES")
    print("="*70)
    print(f"Trading costs: {total_cost}%")
    
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    print("\nOpportunities at different profit thresholds:")
    print(f"{'Threshold':<12} {'Count':<10} {'Percentage':<12} {'Profitable?'}")
    print("-" * 50)
    
    for threshold in thresholds:
        count = (df_filtered['abs_price_diff_pct'] >= threshold).sum()
        pct = (count / len(df_filtered)) * 100
        profitable = "YES" if threshold > total_cost else "NO (fees too high)"
        print(f"{threshold}%{'':<9} {count:<10} {pct:<11.2f}% {profitable}")
    
    # Find best opportunities
    print("\n" + "="*70)
    print("TOP 20 ARBITRAGE OPPORTUNITIES")
    print("="*70)
    
    # Sort by absolute price difference
    top_opportunities = df_filtered.nlargest(20, 'abs_price_diff_pct')
    
    print(f"\n{'Time':<20} {exchange1:<12} {exchange2:<12} {'Diff %':<10} {'Profit/BTC':<12} {'Direction'}")
    print("-" * 100)
    
    for idx, row in top_opportunities.iterrows():
        direction = f"SELL {exchange1}, BUY {exchange2}" if row['price_diff'] > 0 else f"SELL {exchange2}, BUY {exchange1}"
        profit = abs(row['price_diff']) - (row[col2_close] * total_cost / 100)
        print(f"{str(row['time']):<20} ${row[col1_close]:<11.2f} ${row[col2_close]:<11.2f} {row['abs_price_diff_pct']:<9.4f}% ${profit:<11.2f} {direction}")
    
    # Calculate potential profits
    print("\n" + "="*70)
    print("PROFIT POTENTIAL ANALYSIS")
    print("="*70)
    
    # Only consider opportunities above cost threshold
    profitable_threshold = total_cost + 0.5  # Add 0.5% minimum profit
    profitable_ops = df_filtered[df_filtered['abs_price_diff_pct'] >= profitable_threshold].copy()
    
    if len(profitable_ops) > 0:
        profitable_ops['profit_per_btc'] = (profitable_ops['abs_price_diff_pct'] - total_cost) * profitable_ops[col2_close] / 100
        
        print(f"\nProfitable opportunities (>{profitable_threshold}% difference): {len(profitable_ops)}")
        print(f"Percentage of all data: {len(profitable_ops)/len(df_filtered)*100:.2f}%")
        print(f"\nIf trading 1 BTC per opportunity:")
        print(f"  Total potential profit: ${profitable_ops['profit_per_btc'].sum():.2f}")
        print(f"  Average profit per trade: ${profitable_ops['profit_per_btc'].mean():.2f}")
        print(f"  Best single trade: ${profitable_ops['profit_per_btc'].max():.2f}")
        print(f"  Worst profitable trade: ${profitable_ops['profit_per_btc'].min():.2f}")
    else:
        print(f"\nNo profitable opportunities found (>{profitable_threshold}% difference)")
        print("This means arbitrage is not possible with these exchanges after fees.")
    
    # Plot price differences over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Price difference over time
    ax1.plot(df_filtered['time'], df_filtered['price_diff_pct'], linewidth=0.5, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=total_cost, color='red', linestyle='--', linewidth=1, label=f'Cost threshold ({total_cost}%)')
    ax1.axhline(y=-total_cost, color='red', linestyle='--', linewidth=1)
    ax1.fill_between(df_filtered['time'], -total_cost, total_cost, alpha=0.2, color='red', label='Unprofitable zone')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price Difference (%)')
    ax1.set_title(f'{exchange1} vs {exchange2} Price Difference Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of price differences
    ax2.hist(df_filtered['price_diff_pct'], bins=100, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=total_cost, color='red', linestyle='--', linewidth=2, label=f'Cost threshold (±{total_cost}%)')
    ax2.axvline(x=-total_cost, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Price Difference (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Price Differences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path(__file__).parent / f'arbitrage_real_{exchange1}_{exchange2}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_file}")
    plt.show()
    
    return df_filtered

def main():
    # Load data
    csv_file = Path(__file__).parent.parent / 'data_retrieve' / 'combined_BTCUSD_data.csv'
    
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        return
    
    print("Loading data...")
    df = load_data(csv_file)
    
    # Analyze BINANCE vs BITFINEX
    df_filtered = analyze_arbitrage(df, 'BINANCE', 'BITFINEX')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nThis shows the REAL arbitrage opportunities in your data.")
    print("Next step: Build a model to predict WHEN these opportunities will occur.")

if __name__ == "__main__":
    main()