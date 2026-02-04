import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data(csv_file):
    """Load the combined cryptocurrency data from CSV"""
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    return df

def plot_arbitrage_high_prices(df, exchange1='BINANCE', exchange2='BITFINEX'):
    """
    Plot high prices from two exchanges to visualize arbitrage opportunities
    
    Arbitrage Strategy:
    - If Exchange A price > Exchange B price: SELL on A, BUY on B
    - You keep your BTC and make profit = (Price_A - Price_B) * amount
    
    Parameters:
    -----------
    df : DataFrame
        Combined data with all exchanges
    exchange1 : str
        First exchange name (default: BINANCE)
    exchange2 : str
        Second exchange name (default: BITFINEX)
    """
    # Get the high price columns
    col1 = f'{exchange1}:high'
    col2 = f'{exchange2}:high'
    
    # Filter out rows where both exchanges have data
    df_filtered = df[['time', col1, col2]].dropna()
    
    print(f"Data points with both {exchange1} and {exchange2}: {len(df_filtered)}")
    
    # Calculate price differences and arbitrage opportunities
    df_filtered['price_diff'] = df_filtered[col1] - df_filtered[col2]
    df_filtered['price_diff_pct'] = (df_filtered['price_diff'] / df_filtered[col2]) * 100
    df_filtered['profit_per_btc'] = abs(df_filtered['price_diff'])
    
    # Determine arbitrage direction
    # Positive: Binance higher -> Sell on Binance, Buy on Bitfinex
    # Negative: Bitfinex higher -> Sell on Bitfinex, Buy on Binance
    df_filtered['arbitrage_direction'] = df_filtered['price_diff'].apply(
        lambda x: f'SELL {exchange1}, BUY {exchange2}' if x > 0 else f'SELL {exchange2}, BUY {exchange1}'
    )
    
    # Create the scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Price comparison scatter
    colors = ['red' if x > 0 else 'green' for x in df_filtered['price_diff']]
    ax1.scatter(df_filtered[col1], df_filtered[col2], alpha=0.5, s=10, c=colors)
    
    # Add a diagonal line (y=x) representing perfect price parity
    min_price = min(df_filtered[col1].min(), df_filtered[col2].min())
    max_price = max(df_filtered[col1].max(), df_filtered[col2].max())
    ax1.plot([min_price, max_price], [min_price, max_price], 
             'b--', linewidth=2, label='Perfect Parity (y=x)', alpha=0.7)
    
    # Labels and title
    ax1.set_xlabel(f'{exchange1} High Price (USD)', fontsize=12)
    ax1.set_ylabel(f'{exchange2} High Price (USD)', fontsize=12)
    ax1.set_title(f'Price Comparison: {exchange1} vs {exchange2}', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text showing arbitrage regions
    ax1.text(0.05, 0.95, f'{exchange1} higher\nSELL on {exchange1}\nBUY on {exchange2}', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax1.text(0.95, 0.05, f'{exchange2} higher\nSELL on {exchange2}\nBUY on {exchange1}', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Plot 2: Profit per BTC over time
    ax2.plot(df_filtered['time'], df_filtered['profit_per_btc'], 
             linewidth=1, alpha=0.7, color='purple')
    ax2.fill_between(df_filtered['time'], 0, df_filtered['profit_per_btc'], 
                      alpha=0.3, color='purple')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Arbitrage Profit per BTC (USD)', fontsize=12)
    ax2.set_title('Arbitrage Profit Potential Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Rotate x-axis labels
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(__file__).parent / f'arbitrage_{exchange1}_vs_{exchange2}_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    plt.show()
    
    # Calculate and print detailed statistics
    print("\n" + "="*70)
    print("ARBITRAGE STATISTICS")
    print("="*70)
    
    print(f"\nPrice Difference ({exchange1} - {exchange2}):")
    print(f"  Mean: ${df_filtered['price_diff'].mean():.2f} ({df_filtered['price_diff_pct'].mean():.4f}%)")
    print(f"  Std Dev: ${df_filtered['price_diff'].std():.2f} ({df_filtered['price_diff_pct'].std():.4f}%)")
    print(f"  Min: ${df_filtered['price_diff'].min():.2f} ({df_filtered['price_diff_pct'].min():.4f}%)")
    print(f"  Max: ${df_filtered['price_diff'].max():.2f} ({df_filtered['price_diff_pct'].max():.4f}%)")
    
    print(f"\nProfit Per BTC (absolute difference):")
    print(f"  Mean: ${df_filtered['profit_per_btc'].mean():.2f}")
    print(f"  Median: ${df_filtered['profit_per_btc'].median():.2f}")
    print(f"  Max: ${df_filtered['profit_per_btc'].max():.2f}")
    
    # Count profitable opportunities (after accounting for typical 0.1% trading fee on each side)
    trading_fee_pct = 0.2  # 0.1% buy + 0.1% sell = 0.2% total
    threshold_pct = trading_fee_pct
    
    profitable_trades = df_filtered[abs(df_filtered['price_diff_pct']) > threshold_pct]
    
    print(f"\nArbitrage Opportunities (>{threshold_pct}% difference to cover fees):")
    print(f"  Count: {len(profitable_trades)}")
    print(f"  Percentage of total: {(len(profitable_trades)/len(df_filtered)*100):.2f}%")
    
    if len(profitable_trades) > 0:
        print(f"\nProfitable Trade Statistics:")
        print(f"  Average profit per BTC: ${profitable_trades['profit_per_btc'].mean():.2f}")
        print(f"  Average profit %: {abs(profitable_trades['price_diff_pct']).mean():.4f}%")
        print(f"  Best opportunity: ${profitable_trades['profit_per_btc'].max():.2f}")
        
        # Show top 5 opportunities
        print(f"\nTop 5 Arbitrage Opportunities:")
        top_5 = profitable_trades.nlargest(5, 'profit_per_btc')[['time', col1, col2, 'profit_per_btc', 'price_diff_pct', 'arbitrage_direction']]
        for idx, row in top_5.iterrows():
            print(f"  {row['time']}: ${row['profit_per_btc']:.2f} profit ({abs(row['price_diff_pct']):.3f}%) - {row['arbitrage_direction']}")
        
        # Calculate potential total profit if you traded on every opportunity with 1 BTC
        total_potential_profit = profitable_trades['profit_per_btc'].sum()
        print(f"\nTotal Potential Profit (1 BTC per opportunity): ${total_potential_profit:.2f}")
        print(f"  Average profit per opportunity: ${total_potential_profit/len(profitable_trades):.2f}")
    
    return df_filtered

def plot_price_over_time(df, exchanges=['BINANCE', 'BITFINEX', 'COINBASE']):
    """Plot price comparison over time for multiple exchanges"""
    plt.figure(figsize=(15, 8))
    
    for exchange in exchanges:
        col = f'{exchange}:high'
        if col in df.columns:
            df_clean = df[['time', col]].dropna()
            plt.plot(df_clean['time'], df_clean[col], label=exchange, alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('High Price (USD)', fontsize=12)
    plt.title('BTC/USD High Prices Across Exchanges', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'prices_over_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Time series plot saved to: {output_file}")
    plt.show()

def main():
    print("="*70)
    print("CRYPTOCURRENCY ARBITRAGE VISUALIZATION")
    print("="*70)
    print("\nArbitrage Strategy:")
    print("  If you have 1 BTC:")
    print("  - When Exchange A > Exchange B: SELL on A, BUY on B")
    print("  - Profit = (Price_A - Price_B) × 1 BTC")
    print("  - You keep your 1 BTC + make profit in USD")
    print("="*70)
    
    # Look for CSV file in data_retrieve directory
    csv_file = Path(__file__).parent.parent / 'data_retrieve' / 'combined_BTCUSD_data.csv'
    
    if not csv_file.exists():
        print(f"\nError: CSV file not found at {csv_file}")
        print("Please run data_retrieve.py first to generate the data.")
        return
    
    print(f"\nLoading data from: {csv_file}")
    df = load_data(csv_file)
    
    print(f"Total records: {len(df)}")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Plot Binance vs Bitfinex arbitrage
    print("\n" + "="*70)
    print("Analyzing BINANCE vs BITFINEX Arbitrage")
    print("="*70)
    df_filtered = plot_arbitrage_high_prices(df, 'BINANCE', 'BITFINEX')
    
    # Plot prices over time
    print("\n" + "="*70)
    print("Plotting Price Trends")
    print("="*70)
    plot_price_over_time(df)

if __name__ == "__main__":
    main()