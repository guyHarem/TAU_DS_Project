import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import sys


# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

TRADING_COST_PCT = 0.5
SAFETY_MARGIN_PCT = 0.1
REAL_OPPORTUNITY_THRESHOLD = TRADING_COST_PCT + SAFETY_MARGIN_PCT


# Load the combined CSV files - mode 1 only!
data_path = '../data/raw_data'
btcusd_data = pd.read_csv(f'{data_path}/combined_BTCUSD_data.csv')
ethusd_data = pd.read_csv(f'{data_path}/combined_ETHUSD_data.csv')
dogeusd_data = pd.read_csv(f'{data_path}/combined_DOGEUSD_data.csv')
linkusd_data = pd.read_csv(f'{data_path}/combined_LINKUSD_data.csv')
solusd_data = pd.read_csv(f'{data_path}/combined_SOLUSD_data.csv')
xrpusd_data = pd.read_csv(f'{data_path}/combined_XRPUSD_data.csv')

data_frames = [btcusd_data, ethusd_data, dogeusd_data, linkusd_data, solusd_data, xrpusd_data]
exchanges = ["BINANCE","BITFINEX","COINBASE","GATEIO","MEXC","KRAKEN"]
original_features = ["high","low","open","close","volume"]

### DATA CLEANING - DO WE WANT TO DISCARD EVERY LINE THAT IS NOT FULL ALREADY HERE ?? ###

# Change time string from CSV to time type
for df in data_frames:
    df['time'] = pd.to_datetime(df['time'])

## FEATURE ENGINEREEING ##

def add_close_spread(df):
    close_cols = [f"{ex}:close" for ex in exchanges if f"{ex}:close" in df.columns]
    
    # min max close identification
    df['min_close'] = df[close_cols].min(axis=1, skipna=True)
    df['max_close'] = df[close_cols].max(axis=1, skipna=True)
    df['buy_exchange'] = df[close_cols].idxmin(axis=1, skipna=True).str.split(':').str[0]
    df['sell_exchange'] = df[close_cols].idxmax(axis=1, skipna=True).str.split(':').str[0]
    
    # spread close calculation
    df['spread_close_absolute'] = df['max_close'] - df['min_close']
    df['spread_close_pct'] = (df['spread_close_absolute'] / df['min_close']) * 100
    
    # Opportunity flags (from section 11)
    df['is_opportunity'] = (df['spread_close_pct'] >= TRADING_COST_PCT).astype(int)
    df['is_opportunity_flag'] = df['is_opportunity']  # Alias for backward compatibility
    df['is_real_opportunity'] = (df['spread_close_pct'] >= REAL_OPPORTUNITY_THRESHOLD).astype(int)
    
    # data quality check
    df['num_exchanges_available'] = df[close_cols].notna().sum(axis=1)
  
def add_volume_features(df):
    # Need .apply() because each row uses a DIFFERENT column
    df['volume_buy_exchange'] = df.apply(
        lambda row: row[f"{row['buy_exchange']}:volume"], 
        axis=1
    )
    df['volume_sell_exchange'] = df.apply(
        lambda row: row[f"{row['sell_exchange']}:volume"], 
        axis=1
    )
    
    # Now we can use vectorized operations
    df['min_volume'] = df[['volume_buy_exchange', 'volume_sell_exchange']].min(axis=1, skipna=True)
    df['volume_ratio'] = df['volume_sell_exchange'] / df['volume_buy_exchange']
    
def add_high_low_spread(df):
    high_cols = [f"{ex}:high" for ex in exchanges if f"{ex}:high" in df.columns]
    low_cols = [f"{ex}:low" for ex in exchanges if f"{ex}:low" in df.columns]
    # high low identification
    df['max_high'] = df[high_cols].max(axis=1, skipna=True)
    df['min_low'] = df[low_cols].min(axis=1,skipna= True)
    df['high_exchange'] = df[high_cols].idxmax(axis=1, skipna=True).str.split(':').str[0]
    df['low_exchange'] = df[low_cols].idxmin(axis=1, skipna=True).str.split(':').str[0]
    
    # high low spread calculation
    df['spread_highlow_absolute'] = df['max_high'] - df['min_low']
    df['spread_highlow_pct'] = (df['spread_highlow_absolute'] / df['min_low']) * 100
    df['opportunity_gap'] = df['spread_highlow_pct'] - df['spread_close_pct']

def add_time_features(df): # Section 2
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df[f'overlap_hours'] = df[f'hour'].apply(lambda x: 1 if 19 <= x <= 21 else 0)  # Example: active trading hours
    return df

def add_volatility_features(df): # Section 3
    for exchange in exchanges:
        df[f'{exchange}_volatility'] = (df[f'{exchange}:high'] - df[f'{exchange}:low']) / df[f'{exchange}:close'] * 100
    df[f'volatility_avg'] = df[[f'{exchange}_volatility' for exchange in exchanges]].mean(axis=1)
    df[f'volatility_max'] = df[[f'{exchange}_volatility' for exchange in exchanges]].max(axis=1)
    df[f'volatility_min'] = df[[f'{exchange}_volatility' for exchange in exchanges]].min(axis=1) # Maybe delete?
    df['price_position_buy_exchange'] = df.apply(
        lambda row: (
            (row[f"{row['buy_exchange']}:close"] - row[f"{row['buy_exchange']}:low"]) /
            (row[f"{row['buy_exchange']}:high"] - row[f"{row['buy_exchange']}:low"])
        ) if not np.isclose(row[f"{row['buy_exchange']}:high"], row[f"{row['buy_exchange']}:low"]) else np.nan,
        axis=1
    )
    df['price_position_sell_exchange'] = df.apply(
        lambda row: (
            (row[f"{row['sell_exchange']}:close"] - row[f"{row['sell_exchange']}:low"]) /
            (row[f"{row['sell_exchange']}:high"] - row[f"{row['sell_exchange']}:low"])
        ) if not np.isclose(row[f"{row['sell_exchange']}:high"], row[f"{row['sell_exchange']}:low"]) else np.nan,
        axis=1
    )

def add_price_change_features(df): #Section 4
    for exchange in exchanges:
        df[f'{exchange}_price_change'] = (df[f'{exchange}:close'] - df[f'{exchange}:open']) / df[f'{exchange}:open'] * 100
    df['price_change_buy_exchange'] = df.apply(
        lambda row: row[f"{row['buy_exchange']}_price_change"],
        axis=1
    )
    df['price_change_sell_exchange'] = df.apply(
        lambda row: row[f"{row['sell_exchange']}_price_change"],
        axis=1
    )

def add_moving_averages(df, windows=[5, 15, 30]): # Section 5
    for window in windows:
        df[f'spread_ma_{window}'] = df[f'spread_close_pct'].rolling(window=window).mean()
        df[f'volume_ma_buy_{window}'] = df[f'volume_buy_exchange'].rolling(window=window).mean()
        df[f'volume_ma_sell_{window}'] = df[f'volume_sell_exchange'].rolling(window=window).mean()
        df[f'spread_ema_{window}'] = df[f'spread_close_pct'].ewm(span=window, adjust=False).mean()

def add_bollinger_bands(df, windows=[5, 15, 30], num_std=2): # Section 6
    for window in windows:
        df[f'spread_bb_ma_{window}'] = df[f'spread_close_pct'].rolling(window=window).mean()
        df[f'spread_bb_std_{window}'] = df[f'spread_close_pct'].rolling(window=window).std()
        df[f'spread_bb_upper_{window}'] = df[f'spread_bb_ma_{window}'] + (df[f'spread_bb_std_{window}'] * num_std)
        df[f'spread_bb_lower_{window}'] = df[f'spread_bb_ma_{window}'] - (df[f'spread_bb_std_{window}'] * num_std)
        
        lower = df[f'spread_bb_lower_{window}']
        upper = df[f'spread_bb_upper_{window}']
        denominator = upper - lower
        df[f'spread_bb_position_{window}'] = np.where(
            np.isclose(denominator, 0, 1e-9),
            np.nan,
            (df['spread_close_pct'] - lower) / denominator
        )
        df[f'spread_bb_position_{window}'].replace([np.inf, -np.inf], np.nan, inplace=True)

def add_rolling_stats(df, windows=[5, 10, 30]):  # Section 7
    """Add rolling statistical features"""
    
    for window in windows:
        # Spread rolling statistics
        df[f'spread_rolling_std_{window}'] = df['spread_close_pct'].rolling(window=window).std()
        df[f'spread_rolling_max_{window}'] = df['spread_close_pct'].rolling(window=window).max()
        df[f'spread_rolling_min_{window}'] = df['spread_close_pct'].rolling(window=window).min()
        
        # Volume rolling statistics
        df[f'volume_buy_rolling_std_{window}'] = df['volume_buy_exchange'].rolling(window=window).std()
        df[f'volume_sell_rolling_std_{window}'] = df['volume_sell_exchange'].rolling(window=window).std()
        
        # Opportunity counts - BOTH versions
        df[f'opportunities_in_last_{window}'] = df['is_opportunity'].rolling(window=window).sum()
        df[f'real_opportunities_in_last_{window}'] = df['is_real_opportunity'].rolling(window=window).sum()
        
        # Spread range
        df[f'spread_range_{window}'] = (df['spread_close_pct'].rolling(window=window).max() - 
                                        df['spread_close_pct'].rolling(window=window).min())
        
        # Z-score
        rolling_mean = df['spread_close_pct'].rolling(window=window).mean()
        rolling_std = df['spread_close_pct'].rolling(window=window).std()
        df[f'spread_zscore_{window}'] = np.where(
            np.isclose(rolling_std, 0, 1e-9),
            np.nan,
            (df['spread_close_pct'] - rolling_mean) / rolling_std
        )
        df[f'spread_zscore_{window}'].replace([np.inf, -np.inf], np.nan, inplace=True)

def add_rate_change_features(df): # Section 8
    df[f'spread_rate_change'] = df[f'spread_close_pct'] - df[f'spread_close_pct'].shift(1)
    
    df[f'spread_rate_change_pct'] = np.where(
        np.isclose(df[f'spread_close_pct'].shift(1), 0, 1e-9),
        np.nan,
        df['spread_rate_change'] / df[f'spread_close_pct'].shift(1) * 100
    )
    df[f'spread_rate_change_pct'].replace([np.inf, -np.inf], np.nan, inplace=True)

    df[f'spread_rate_acceleration'] = df[f'spread_rate_change'] - df[f'spread_rate_change'].shift(1)

def add_cross_ex_price_ratio(df): # Section 9
    """Add cross-exchange price ratio features (Section 9)"""
    # Price ratio between buy and sell exchanges
    df['price_ratio_buy_sell'] = df.apply(
        lambda row: row[f"{row['sell_exchange']}:close"] / row[f"{row['buy_exchange']}:close"],
        axis=1
    )
    # For each pair of exchanges, calculate price ratios
    for i, ex1 in enumerate(exchanges):
        for ex2 in exchanges[i+1:]:  # Avoid duplicate pairs
            close1 = f"{ex1}:close"
            close2 = f"{ex2}:close"
            
            if close1 in df.columns and close2 in df.columns:
                df[f'price_ratio_{ex1}_{ex2}'] = df[close2] / df[close1]
    
    # Average price ratio across all exchange pairs
    ratio_cols = [col for col in df.columns if col.startswith('price_ratio_') and col != 'price_ratio_buy_sell']
    if ratio_cols:
        df['avg_price_ratio'] = df[ratio_cols].mean(axis=1)
        df['max_price_ratio'] = df[ratio_cols].max(axis=1)
        df['min_price_ratio'] = df[ratio_cols].min(axis=1)
        df['price_ratio_std'] = df[ratio_cols].std(axis=1)

def add_lag_features(df, lags=[1, 5, 10, 30]):  # Section 10
    """Add lag features for time-series prediction"""
    
    for lag in lags:
        # Spread lags
        df[f'spread_lag_{lag}'] = df['spread_close_pct'].shift(lag)
        
        # Volume lags
        df[f'volume_buy_lag_{lag}'] = df['volume_buy_exchange'].shift(lag)
        df[f'volume_sell_lag_{lag}'] = df['volume_sell_exchange'].shift(lag)
        df[f'min_volume_lag_{lag}'] = df['min_volume'].shift(lag)
        
        # Opportunity flag lags - BOTH versions
        df[f'is_opportunity_lag_{lag}'] = df['is_opportunity'].shift(lag)
        df[f'is_real_opportunity_lag_{lag}'] = df['is_real_opportunity'].shift(lag)
        
        # Price change lags
        df[f'price_change_buy_lag_{lag}'] = df['price_change_buy_exchange'].shift(lag)
        df[f'price_change_sell_lag_{lag}'] = df['price_change_sell_exchange'].shift(lag)
        
        # Volatility lags
        df[f'volatility_avg_lag_{lag}'] = df['volatility_avg'].shift(lag)
    
    # Categorical lags (exchange names)
    df['buy_exchange_lag_1'] = df['buy_exchange'].shift(1)
    df['sell_exchange_lag_1'] = df['sell_exchange'].shift(1)
    
    # Diff features (change from lag)
    df['spread_diff_from_lag_1'] = df['spread_close_pct'] - df['spread_lag_1']
    df['spread_diff_from_lag_5'] = df['spread_close_pct'] - df['spread_lag_5']
    df['volume_diff_from_lag_1'] = df['min_volume'] - df['min_volume_lag_1']

def save_featured_data():
    
    print("\n=== SAVING FEATURED DATA ===\n")
    
    featured_data_path = '../data/featured_data'
       
    datasets = {
        'BTCUSD': btcusd_data,
        'ETHUSD': ethusd_data,
        'DOGEUSD': dogeusd_data,
        'LINKUSD': linkusd_data,
        'SOLUSD': solusd_data,
        'XRPUSD': xrpusd_data
    }
    for name, df in datasets.items():
        output_file = f'{featured_data_path}/featured_{name}_data.csv'
        df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file} ({len(df)} rows, {len(df.columns)} columns)")
    
    print("\n🎉 All featured data saved successfully!")
    

def load_featured_data():
    """Load all featured datasets"""
    print("=== LOADING FEATURED DATA ===\n")
    
    featured_data_path = '../data/featured_data'
    
    datasets = {}
    crypto_names = ['BTCUSD', 'ETHUSD', 'DOGEUSD', 'LINKUSD', 'SOLUSD', 'XRPUSD']
    
    for name in crypto_names:
        file_path = f'{featured_data_path}/featured_{name}_data.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df['time'])
            datasets[name] = df
            print(f"✅ Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
        else:
            print(f"❌ File not found: {file_path}")
    
    print(f"\n✅ Loaded {len(datasets)} datasets\n")
    
    return datasets


def analyze_opportunity_frequency(datasets):
    """Phase 2: Deep Opportunity Analysis"""
    print("\n" + "="*60)
    print("PHASE 2: OPPORTUNITY FREQUENCY ANALYSIS")
    print("="*60 + "\n")
    
    results = {}
    
    for name, df in datasets.items():
        print(f"\n--- {name} Analysis ---")
        
        # Basic statistics
        total_rows = len(df)
        total_opportunities = df['is_opportunity'].sum()
        real_opportunities = df['is_real_opportunity'].sum()
        
        opportunity_pct = (total_opportunities / total_rows) * 100
        real_opportunity_pct = (real_opportunities / total_rows) * 100
        
        print(f"Total minutes analyzed: {total_rows}")
        print(f"Opportunities (≥0.50%): {total_opportunities} ({opportunity_pct:.2f}%)")
        print(f"Real opportunities (≥0.60%): {real_opportunities} ({real_opportunity_pct:.2f}%)")
        
        # Opportunity duration analysis
        df['opportunity_group'] = (df['is_real_opportunity'] != df['is_real_opportunity'].shift()).cumsum()
        opportunity_durations = df[df['is_real_opportunity'] == 1].groupby('opportunity_group').size()
        
        if len(opportunity_durations) > 0:
            print(f"\nOpportunity Duration Statistics:")
            print(f"  Average duration: {opportunity_durations.mean():.2f} minutes")
            print(f"  Median duration: {opportunity_durations.median():.0f} minutes")
            print(f"  Max duration: {opportunity_durations.max():.0f} minutes")
            print(f"  Total opportunity events: {len(opportunity_durations)}")
        
        # Average spreads
        avg_spread_all = df['spread_close_pct'].mean()
        avg_spread_opportunity = df[df['is_opportunity'] == 1]['spread_close_pct'].mean()
        avg_spread_real = df[df['is_real_opportunity'] == 1]['spread_close_pct'].mean()
        
        print(f"\nAverage Spreads:")
        print(f"  All times: {avg_spread_all:.4f}%")
        print(f"  During opportunities: {avg_spread_opportunity:.4f}%")
        print(f"  During real opportunities: {avg_spread_real:.4f}%")
        
        # Store results
        results[name] = {
            'total_minutes': total_rows,
            'opportunities': total_opportunities,
            'real_opportunities': real_opportunities,
            'opportunity_pct': opportunity_pct,
            'real_opportunity_pct': real_opportunity_pct,
            'avg_duration': opportunity_durations.mean() if len(opportunity_durations) > 0 else 0,
            'avg_spread_all': avg_spread_all,
            'avg_spread_opportunity': avg_spread_opportunity,
            'avg_spread_real': avg_spread_real
        }
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Cross-Crypto Comparison")
    print("="*60)
    
    summary_df = pd.DataFrame(results).T
    print(summary_df[['opportunity_pct', 'real_opportunity_pct', 'avg_duration', 'avg_spread_real']])
    
    return results


def analyze_temporal_patterns(datasets):
    """Phase 3: When do opportunities occur?"""
    print("\n" + "="*60)
    print("PHASE 3: TEMPORAL PATTERN ANALYSIS")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Temporal Patterns ---")
        
        # Hourly analysis
        hourly_opportunities = df.groupby('hour')['is_real_opportunity'].agg(['sum', 'mean', 'count'])
        hourly_opportunities['opportunity_rate'] = (hourly_opportunities['sum'] / hourly_opportunities['count']) * 100
        
        print("\nTop 5 Hours for Opportunities:")
        top_hours = hourly_opportunities.nlargest(5, 'opportunity_rate')
        for hour, row in top_hours.iterrows():
            print(f"  Hour {hour:02d}:00 - {row['opportunity_rate']:.2f}% ({int(row['sum'])} opportunities)")
        
        # Day of week analysis
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_opportunities = df.groupby('day_of_week')['is_real_opportunity'].agg(['sum', 'mean', 'count'])
        daily_opportunities['opportunity_rate'] = (daily_opportunities['sum'] / daily_opportunities['count']) * 100
        daily_opportunities['day_name'] = [day_names[i] for i in daily_opportunities.index]
        
        print("\nOpportunity Rate by Day of Week:")
        for idx, row in daily_opportunities.iterrows():
            print(f"  {row['day_name']}: {row['opportunity_rate']:.2f}% ({int(row['sum'])} opportunities)")
        
        # Weekend vs Weekday
        weekend_rate = df[df['is_weekend'] == 1]['is_real_opportunity'].mean() * 100
        weekday_rate = df[df['is_weekend'] == 0]['is_real_opportunity'].mean() * 100
        
        print(f"\nWeekend vs Weekday:")
        print(f"  Weekend opportunity rate: {weekend_rate:.2f}%")
        print(f"  Weekday opportunity rate: {weekday_rate:.2f}%")
        print(f"  Difference: {weekend_rate - weekday_rate:+.2f}%")
        
        # Market overlap hours
        overlap_rate = df[df['overlap_hours'] == 1]['is_real_opportunity'].mean() * 100
        non_overlap_rate = df[df['overlap_hours'] == 0]['is_real_opportunity'].mean() * 100
        
        print(f"\nMarket Overlap Hours (19:00-21:00 UTC):")
        print(f"  During overlap: {overlap_rate:.2f}%")
        print(f"  Outside overlap: {non_overlap_rate:.2f}%")
        print(f"  Difference: {overlap_rate - non_overlap_rate:+.2f}%")


def analyze_exchange_patterns(datasets):
    """Phase 4: Which exchanges are most profitable?"""
    print("\n" + "="*60)
    print("PHASE 4: EXCHANGE PATTERN ANALYSIS")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Exchange Analysis ---")
        
        # Most common exchange pairs
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            df_opportunities['exchange_pair'] = df_opportunities['buy_exchange'] + ' → ' + df_opportunities['sell_exchange']
            pair_counts = df_opportunities['exchange_pair'].value_counts()
            
            print("\nTop 10 Most Profitable Exchange Pairs:")
            for pair, count in pair_counts.head(10).items():
                pct = (count / len(df_opportunities)) * 100
                avg_spread = df_opportunities[df_opportunities['exchange_pair'] == pair]['spread_close_pct'].mean()
                print(f"  {pair}: {count} times ({pct:.1f}%) - Avg spread: {avg_spread:.4f}%")
            
            # Buy exchange frequency
            print("\nMost Common Buy Exchanges (cheapest):")
            buy_counts = df_opportunities['buy_exchange'].value_counts()
            for exchange, count in buy_counts.items():
                pct = (count / len(df_opportunities)) * 100
                print(f"  {exchange}: {count} times ({pct:.1f}%)")
            
            # Sell exchange frequency
            print("\nMost Common Sell Exchanges (most expensive):")
            sell_counts = df_opportunities['sell_exchange'].value_counts()
            for exchange, count in sell_counts.items():
                pct = (count / len(df_opportunities)) * 100
                print(f"  {exchange}: {count} times ({pct:.1f}%)")


def analyze_volume_liquidity(datasets):
    """Phase 4b: Volume and liquidity analysis"""
    print("\n" + "="*60)
    print("PHASE 4B: VOLUME & LIQUIDITY ANALYSIS")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Volume Analysis ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            print(f"\nVolume Statistics During Opportunities:")
            print(f"  Average min_volume: {df_opportunities['min_volume'].mean():.2f}")
            print(f"  Median min_volume: {df_opportunities['min_volume'].median():.2f}")
            print(f"  25th percentile: {df_opportunities['min_volume'].quantile(0.25):.2f}")
            print(f"  75th percentile: {df_opportunities['min_volume'].quantile(0.75):.2f}")
            
            # Volume sufficiency
            volume_thresholds = [10, 50, 100, 500]
            print(f"\nOpportunities by Volume Threshold:")
            for threshold in volume_thresholds:
                count = (df_opportunities['min_volume'] >= threshold).sum()
                pct = (count / len(df_opportunities)) * 100
                print(f"  ≥ {threshold}: {count} opportunities ({pct:.1f}%)")
            
            # Volume ratio analysis
            print(f"\nVolume Ratio (sell/buy) Statistics:")
            print(f"  Average ratio: {df_opportunities['volume_ratio'].mean():.3f}")
            print(f"  Median ratio: {df_opportunities['volume_ratio'].median():.3f}")


def analyze_risk_factors(datasets):
    """Phase 6: Risk assessment during opportunities"""
    print("\n" + "="*60)
    print("PHASE 6: RISK FACTOR ANALYSIS")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Risk Analysis ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            # Volatility during opportunities
            print(f"\nVolatility During Opportunities:")
            print(f"  Average volatility: {df_opportunities['volatility_avg'].mean():.4f}%")
            print(f"  Median volatility: {df_opportunities['volatility_avg'].median():.4f}%")
            print(f"  Max volatility: {df_opportunities['volatility_avg'].max():.4f}%")
            
            # Opportunity gap (high-low vs close spread)
            print(f"\nOpportunity Gap Analysis:")
            print(f"  Average gap: {df_opportunities['opportunity_gap'].mean():.4f}%")
            print(f"  Median gap: {df_opportunities['opportunity_gap'].median():.4f}%")
            
            # High gap = less realistic to execute at close prices
            high_gap_pct = (df_opportunities['opportunity_gap'] > 0.1).sum() / len(df_opportunities) * 100
            print(f"  Opportunities with gap >0.1%: {high_gap_pct:.1f}%")


def estimate_profitability(datasets):
    """Phase 7: Profitability estimation"""
    print("\n" + "="*60)
    print("PHASE 7: PROFITABILITY ESTIMATION")
    print("="*60 + "\n")
    
    TRADE_AMOUNT_USD = 1000  # Assume $1000 per trade
    
    for name, df in datasets.items():
        print(f"\n--- {name} Profit Estimation ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            # Calculate profit per opportunity
            df_opportunities['profit_pct'] = df_opportunities['spread_close_pct'] - TRADING_COST_PCT
            df_opportunities['profit_usd'] = (df_opportunities['profit_pct'] / 100) * TRADE_AMOUNT_USD
            
            total_opportunities = len(df_opportunities)
            total_profit = df_opportunities['profit_usd'].sum()
            avg_profit_per_trade = df_opportunities['profit_usd'].mean()
            
            print(f"\nAssuming ${TRADE_AMOUNT_USD} per trade:")
            print(f"  Total real opportunities: {total_opportunities}")
            print(f"  Total potential profit: ${total_profit:.2f}")
            print(f"  Average profit per trade: ${avg_profit_per_trade:.2f}")
            print(f"  Profit per hour (if traded all): ${total_profit / (len(df) / 60):.2f}")
            
            # Conservative estimate (only high volume opportunities)
            df_high_volume = df_opportunities[df_opportunities['min_volume'] >= 50]
            if len(df_high_volume) > 0:
                conservative_profit = df_high_volume['profit_usd'].sum()
                print(f"\nConservative Estimate (volume ≥ 50):")
                print(f"  Opportunities: {len(df_high_volume)}")
                print(f"  Total profit: ${conservative_profit:.2f}")
                print(f"  Average profit per trade: ${df_high_volume['profit_usd'].mean():.2f}")


def run_full_analysis():
    """Run all analysis phases"""
    print("\n" + "="*70)
    print("COMPREHENSIVE ARBITRAGE OPPORTUNITY ANALYSIS")
    print("="*70)
    
    # Load data
    datasets = load_featured_data()
    
    if not datasets:
        print("❌ No featured data found. Run feature engineering first!")
        return
    
    # Run all analysis phases
    analyze_opportunity_frequency(datasets)
    analyze_temporal_patterns(datasets)
    analyze_exchange_patterns(datasets)
    analyze_volume_liquidity(datasets)
    analyze_risk_factors(datasets)
    estimate_profitability(datasets)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70 + "\n")


def main():
    
    print("DATA ANALYZER\n")
    print("Choose what you want to do:\n")
    print("1. Add features to raw data (ADD)\n")
    print("2. Run Analyzer on featured data (ANALYZE)\n")
    user_option = input("ADD or ANALYZE? ").strip().upper()
    
    if user_option == "ADD":
        print("\n=== ADDING FEATURES ===\n")
        
        for df in data_frames:
            add_close_spread(df)
            add_volume_features(df)
            add_high_low_spread(df)
            add_time_features(df)
            # add_volatility_features(df)  # Commented out due to division by zero
            add_price_change_features(df)
            add_moving_averages(df)
            add_bollinger_bands(df)
            add_rolling_stats(df)
            add_rate_change_features(df) 
            add_cross_ex_price_ratio(df)
            # add_lag_features(df)  # Commented out
        
        print("✅ Features added!\n")
        save_featured_data()
        
    elif user_option == "ANALYZE":
        
        datasets = load_featured_data()
        
        if not datasets:
            print("❌ No featured data found. Run feature engineering first!")
            return
    
        # Run all analysis phases
        analyze_opportunity_frequency(datasets)
        analyze_temporal_patterns(datasets)
        analyze_exchange_patterns(datasets)
        analyze_volume_liquidity(datasets)
        analyze_risk_factors(datasets)
        estimate_profitability(datasets)
            
    else:
        print("❌ Invalid option! Please enter 'ADD' or 'ANALYZE'")


if __name__ == "__main__":
    main()

















