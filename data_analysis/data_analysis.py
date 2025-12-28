import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

TRADING_COST_PCT = 0.2
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

## FEATURE ENGINEERING ##

def add_close_spread(df):  # L2, from L1 close prices
    """
    LAYER 2: Core Spread Features
    Uses: L1 close prices from all exchanges
    Creates: min_close, max_close, buy_exchange, sell_exchange, spread_close_pct, 
             is_opportunity, is_real_opportunity, num_exchanges_available
    
    Identifies arbitrage opportunities by finding min/max close prices across all exchanges.
    This is the foundation - all other arbitrage features depend on spread_close_pct.
    Uses ALL exchanges (skipna=True) to handle partial data.
    """
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
  
def add_volume_features(df):  # L3, from L2 buy/sell_exchange, from L1 buy/sell volume
    """
    LAYER 3: Volume Features  
    Uses: L2 buy_exchange + sell_exchange, L1 volume from those exchanges
    Creates: volume_buy_exchange, volume_sell_exchange, min_volume, volume_ratio
    
    Extracts volume from the identified buy/sell exchanges to assess executability.
    Uses ONLY BUY and SELL exchanges (2 exchanges).
    If buy or sell exchange volume is NaN, all features become NaN.
    """
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
    
def add_high_low_spread(df):  # L2, from L1 high/low prices
    """
    LAYER 2: High-Low Spread Features
    Uses: L1 high and low prices from all exchanges
    Creates: max_high, min_low, spread_highlow_pct, opportunity_gap
    
    Calculates theoretical maximum arbitrage if perfect timing was achieved.
    Uses ALL exchanges (skipna=True). Depends on spread_close_pct from add_close_spread.
    """
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

def add_time_features(df):  # L2, from L1 time
    """
    LAYER 2: Time Features
    Uses: L1 time column only
    Creates: hour, minute, day_of_week, is_weekend, overlap_hours
    
    Extracts time-based features to capture cyclical patterns.
    No exchange data needed - time always available.
    """
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df[f'overlap_hours'] = df[f'hour'].apply(lambda x: 1 if 19 <= x <= 21 else 0)  # Example: active trading hours

def add_volatility_features(df):  # L3, from L2 buy/sell_exchange, from L1 high/low/close prices
    """
    LAYER 3: Volatility Features
    Uses: L2 buy_exchange + sell_exchange, L1 high/low/close from all exchanges
    Creates: {EXCHANGE}_volatility, volatility_avg/max/min, price_position_buy/sell_exchange
    
    Measures intra-minute volatility and where close sits in high-low range.
    Uses ALL exchanges for volatility aggregates, ONLY BUY/SELL for price positions.
    """
    for exchange in exchanges:
        if f'{exchange}:high' in df.columns and f'{exchange}:low' in df.columns and f'{exchange}:close' in df.columns:
            df[f'{exchange}_volatility'] = (df[f'{exchange}:high'] - df[f'{exchange}:low']) / df[f'{exchange}:close'] * 100
    
    # Get available volatility columns
    volatility_cols = [f'{exchange}_volatility' for exchange in exchanges if f'{exchange}_volatility' in df.columns]
    
    if volatility_cols:
        df[f'volatility_avg'] = df[volatility_cols].mean(axis=1)
        df[f'volatility_max'] = df[volatility_cols].max(axis=1)
        df[f'volatility_min'] = df[volatility_cols].min(axis=1)
    
    df['price_position_buy_exchange'] = df.apply(
        lambda row: (
            (row[f"{row['buy_exchange']}:close"] - row[f"{row['buy_exchange']}:low"]) /
            (row[f"{row['buy_exchange']}:high"] - row[f"{row['buy_exchange']}:low"])
        ) if (f"{row['buy_exchange']}:close" in df.columns and 
              f"{row['buy_exchange']}:low" in df.columns and 
              f"{row['buy_exchange']}:high" in df.columns and
              not np.isclose(row[f"{row['buy_exchange']}:high"], row[f"{row['buy_exchange']}:low"])) else np.nan,
        axis=1
    )
    df['price_position_buy_exchange'] = df['price_position_buy_exchange'].replace([np.inf, -np.inf], np.nan)

    df['price_position_sell_exchange'] = df.apply(
        lambda row: (
            (row[f"{row['sell_exchange']}:close"] - row[f"{row['sell_exchange']}:low"]) /
            (row[f"{row['sell_exchange']}:high"] - row[f"{row['sell_exchange']}:low"])
        ) if (f"{row['sell_exchange']}:close" in df.columns and 
              f"{row['sell_exchange']}:low" in df.columns and 
              f"{row['sell_exchange']}:high" in df.columns and
              not np.isclose(row[f"{row['sell_exchange']}:high"], row[f"{row['sell_exchange']}:low"])) else np.nan,
        axis=1
    )
    df['price_position_sell_exchange'] = df['price_position_sell_exchange'].replace([np.inf, -np.inf], np.nan)

def add_price_change_features(df):  # L3, from L2 buy/sell_exchange, from L1 close/open price
    """
    LAYER 3: Price Change Features
    Uses: L2 buy_exchange + sell_exchange, L1 close and open from all exchanges
    Creates: {EXCHANGE}_price_change, price_change_buy/sell_exchange
    
    Calculates intra-minute price change (close-open) to detect momentum.
    Uses ALL exchanges for per-exchange changes, ONLY BUY/SELL for specific features.
    """
    for exchange in exchanges:
        if f'{exchange}:close' in df.columns and f'{exchange}:open' in df.columns:
            df[f'{exchange}_price_change'] = (df[f'{exchange}:close'] - df[f'{exchange}:open']) / df[f'{exchange}:open'] * 100
    
    df['price_change_buy_exchange'] = df.apply(
        lambda row: row[f"{row['buy_exchange']}_price_change"] if f"{row['buy_exchange']}_price_change" in df.columns else np.nan,
        axis=1
    )
    df['price_change_sell_exchange'] = df.apply(
        lambda row: row[f"{row['sell_exchange']}_price_change"] if f"{row['sell_exchange']}_price_change" in df.columns else np.nan,
        axis=1
    )

def add_moving_averages(df, windows=[5, 15, 30]):  # L4, from L3 volume_buy/sell_exchange, from L2 spread_close_pct
    """
    LAYER 4: Moving Averages
    Uses: L3 volume_buy_exchange + volume_sell_exchange, L2 spread_close_pct
    Creates: spread_ma/ema_{window}, volume_ma_buy/sell_{window} for each window
    
    Calculates SMA and EMA for spread and volume to detect trends.
    ⚠️  WARNING: Volume MAs mix exchanges over time - see ROLLING_STATS_ISSUE.md
    """
    for window in windows:
        df[f'spread_ma_{window}'] = df[f'spread_close_pct'].rolling(window=window).mean()
        df[f'volume_ma_buy_{window}'] = df[f'volume_buy_exchange'].rolling(window=window).mean()
        df[f'volume_ma_sell_{window}'] = df[f'volume_sell_exchange'].rolling(window=window).mean()
        df[f'spread_ema_{window}'] = df[f'spread_close_pct'].ewm(span=window, adjust=False).mean()

def add_bollinger_bands(df, windows=[5, 15, 30], num_std=2):  # L3, from L2 spread_close_pct
    """
    LAYER 3: Bollinger Bands
    Uses: L2 spread_close_pct only
    Creates: spread_bb_ma/std/upper/lower/position_{window} for each window
    
    Detects statistical extremes in spread using Bollinger Bands.
    Note: spread_bb_ma_{window} duplicates spread_ma_{window} from add_moving_averages
    """
    for window in windows:
        df[f'spread_bb_upper_{window}'] = df[f'spread_ma_{window}'] + (df[f'spread_rolling_std_{window}'] * num_std)
        df[f'spread_bb_lower_{window}'] = df[f'spread_ma_{window}'] - (df[f'spread_rolling_std_{window}'] * num_std)
        
        lower = df[f'spread_bb_lower_{window}']
        upper = df[f'spread_bb_upper_{window}']
        denominator = upper - lower
        df[f'spread_bb_position_{window}'] = np.where(
            np.isclose(denominator, 0, 1e-9),
            np.nan,
            (df['spread_close_pct'] - lower) / denominator
        )
        df[f'spread_bb_position_{window}'] = df[f'spread_bb_position_{window}'].replace([np.inf, -np.inf], np.nan, inplace=True)

def add_rolling_stats(df, windows=[5, 15, 30]):  # L4, from L3 volume_buy/sell_exchange, from L2 spread_close_pct + is_opportunity + is_real_opportunity
    """
    LAYER 4: Rolling Statistics
    Uses: L3 volume_buy/sell_exchange, L2 spread_close_pct + is_opportunity + is_real_opportunity
    Creates: spread_rolling_std/max/min/range/zscore, volume_rolling_std, opportunities_in_last_{window}
    
    Calculates rolling statistics for spread stability and opportunity clustering.
    ⚠️  WARNING: Volume stats mix exchanges over time - see ROLLING_STATS_ISSUE.md
    """
    
    for window in windows:
        # Spread rolling statistics
        df[f'spread_rolling_std_{window}'] = df[f'spread_close_pct'].rolling(window=window).std()
        df[f'spread_rolling_max_{window}'] = df[f'spread_close_pct'].rolling(window=window).max()
        df[f'spread_rolling_min_{window}'] = df[f'spread_close_pct'].rolling(window=window).min()
        
        # Volume rolling statistics
        df[f'volume_buy_rolling_std_{window}'] = df[f'volume_buy_exchange'].rolling(window=window).std()
        df[f'volume_sell_rolling_std_{window}'] = df[f'volume_sell_exchange'].rolling(window=window).std()
        
        # Opportunity counts - BOTH versions
        df[f'opportunities_in_last_{window}'] = df[f'is_opportunity'].rolling(window=window).sum()
        df[f'real_opportunities_in_last_{window}'] = df[f'is_real_opportunity'].rolling(window=window).sum()
        
        # Spread range
        df[f'spread_range_{window}'] = (df[f'spread_close_pct'].rolling(window=window).max() - 
                                        df[f'spread_close_pct'].rolling(window=window).min())
        
        # Z-score
        rolling_mean = df[f'spread_close_pct'].rolling(window=window).mean()
        rolling_std = df[f'spread_close_pct'].rolling(window=window).std()
        df[f'spread_zscore_{window}'] = np.where(
            np.isclose(rolling_std, 0, 1e-9),
            np.nan,
            (df['spread_close_pct'] - rolling_mean) / rolling_std
        )
        df[f'spread_zscore_{window}'] = df[f'spread_zscore_{window}'].replace([np.inf, -np.inf], np.nan, inplace=True)

def add_rate_change_features(df):  # L3, from L2 spread_close_pct
    """
    LAYER 3: Rate Change Features
    Uses: L2 spread_close_pct only
    Creates: spread_rate_change, spread_rate_change_pct, spread_rate_acceleration
    
    Calculates first and second derivatives of spread (momentum and acceleration).
    """
    df[f'spread_rate_change'] = df[f'spread_close_pct'] - df[f'spread_close_pct'].shift(1)
    
    df[f'spread_rate_change_pct'] = np.where(
        np.isclose(df[f'spread_close_pct'].shift(1), 0, 1e-9),
        np.nan,
        df['spread_rate_change'] / df[f'spread_close_pct'].shift(1) * 100
    )
    df[f'spread_rate_change_pct'] = df[f'spread_rate_change_pct'].replace([np.inf, -np.inf], np.nan, inplace=True)

    df[f'spread_rate_acceleration'] = df[f'spread_rate_change'] - df[f'spread_rate_change'].shift(1)

def add_cross_ex_price_ratio(df):  # L3, from L2 buy/sell_exchange, from L1 close prices
    """
    LAYER 3: Cross-Exchange Price Ratios
    Uses: L2 buy_exchange + sell_exchange, L1 close prices from all exchanges
    Creates: price_ratio_buy_sell, price_ratio_{EX1}_{EX2} for all pairs, avg/max/min/std aggregates
    
    Calculates price ratios between all exchange pairs to measure market fragmentation.
    Ratios complement spreads by being price-normalized.
    """
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

def add_lag_features(df, lags=[1, 5, 10, 30]):  # L4, from L3 volume/price_change/volatility, from L2 spread/buy_sell_exchange/opportunities
    """
    LAYER 4: Lag Features
    Uses: L3 volume_buy/sell_exchange + price_change_buy/sell_exchange + volatility_avg,
          L2 spread_close_pct + buy/sell_exchange + is_opportunity + is_real_opportunity
    Creates: Lagged versions of all above features (spread_lag, volume_lag, opportunity_lag, etc.)
    
    Creates time-lagged versions of key features for time-series prediction.
    Essential for ML models to learn temporal dependencies.
    """
    
    for lag in lags:
        # Spread lags
        df[f'spread_lag_{lag}'] = df[f'spread_close_pct'].shift(lag)
        
        # Volume lags
        df[f'volume_buy_lag_{lag}'] = df[f'volume_buy_exchange'].shift(lag)
        df[f'volume_sell_lag_{lag}'] = df[f'volume_sell_exchange'].shift(lag)
        df[f'min_volume_lag_{lag}'] = df[f'min_volume'].shift(lag)
        
        # Opportunity flag lags - BOTH versions
        df[f'is_opportunity_lag_{lag}'] = df[f'is_opportunity'].shift(lag)
        df[f'is_real_opportunity_lag_{lag}'] = df[f'is_real_opportunity'].shift(lag)
        
        # Price change lags
        df[f'price_change_buy_lag_{lag}'] = df[f'price_change_buy_exchange'].shift(lag)
        df[f'price_change_sell_lag_{lag}'] = df[f'price_change_sell_exchange'].shift(lag)
        
        # Volatility lags
        df[f'volatility_avg_lag_{lag}'] = df[f'volatility_avg'].shift(lag)
    
    # Categorical lags (exchange names)
    df[f'buy_exchange_lag_1'] = df[f'buy_exchange'].shift(1)
    df[f'sell_exchange_lag_1'] = df[f'sell_exchange'].shift(1)
    
    # Diff features (change from lag)
    diff_lags = [1, 5]
    for lag in diff_lags:
        df[f'spread_diff_from_lag_{lag}'] = df[f'spread_close_pct'] - df[f'spread_lag_{lag}']
        df[f'volume_diff_from_lag_{lag}'] = df[f'min_volume'] - df[f'min_volume_lag_{lag}']


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
    # crypto_names = ['BTCUSD', 'ETHUSD', 'DOGEUSD', 'LINKUSD', 'SOLUSD', 'XRPUSD']
    crypto_names = ['BTCUSD', 'ETHUSD', 'LINKUSD', 'SOLUSD', 'XRPUSD']
    
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
        print(f"Opportunities (≥{TRADING_COST_PCT}%): {total_opportunities} ({opportunity_pct:.2f}%)")
        print(f"Real opportunities (≥{REAL_OPPORTUNITY_THRESHOLD}%): {real_opportunities} ({real_opportunity_pct:.2f}%)")
        
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

def analyze_momentum_indicators(datasets):
    """Phase 8: Momentum and trend indicators analysis"""
    print("\n" + "="*60)
    print("PHASE 8: MOMENTUM INDICATORS ANALYSIS")
    print("="*60 + "\n")
    
    windows = [5, 15, 30]
    
    for name, df in datasets.items():
        print(f"\n--- {name} Momentum Analysis ---")
        
        # Skip NaN rows for rolling features
        df_valid = df.dropna(subset=['spread_ma_5', 'spread_ema_5'])
        df_opportunities = df_valid[df_valid['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            print("\n1. Moving Average Patterns:")
            for window in windows:
                ma_col = f'spread_ma_{window}'
                ema_col = f'spread_ema_{window}'
                
                # Compare current spread vs MA during opportunities
                above_ma = (df_opportunities['spread_close_pct'] > df_opportunities[ma_col]).mean() * 100
                avg_distance_from_ma = (df_opportunities['spread_close_pct'] - df_opportunities[ma_col]).mean()
                
                print(f"\n  Window {window}:")
                print(f"    Opportunities above MA: {above_ma:.1f}%")
                print(f"    Avg distance from MA: {avg_distance_from_ma:.4f}%")
                print(f"    Avg MA during opportunities: {df_opportunities[ma_col].mean():.4f}%")
                print(f"    Avg EMA during opportunities: {df_opportunities[ema_col].mean():.4f}%")
            
            print("\n2. Rate of Change Analysis:")
            if 'spread_rate_change' in df_opportunities.columns:
                print(f"  Average rate of change: {df_opportunities['spread_rate_change'].mean():.4f}%")
                print(f"  Median rate of change: {df_opportunities['spread_rate_change'].median():.4f}%")
                
                # Positive vs negative rate of change
                positive_rate = (df_opportunities['spread_rate_change'] > 0).mean() * 100
                print(f"  Opportunities with positive rate: {positive_rate:.1f}%")
                
                if 'spread_rate_acceleration' in df_opportunities.columns:
                    print(f"  Average acceleration: {df_opportunities['spread_rate_acceleration'].mean():.4f}%")
                    positive_accel = (df_opportunities['spread_rate_acceleration'] > 0).mean() * 100
                    print(f"  Opportunities with positive acceleration: {positive_accel:.1f}%")
            
            print("\n3. Price Change Patterns:")
            if 'price_change_buy_exchange' in df_opportunities.columns:
                print(f"  Avg price change (buy exchange): {df_opportunities['price_change_buy_exchange'].mean():.4f}%")
                print(f"  Avg price change (sell exchange): {df_opportunities['price_change_sell_exchange'].mean():.4f}%")
                
                # Divergence in price movements
                divergence = df_opportunities['price_change_sell_exchange'] - df_opportunities['price_change_buy_exchange']
                print(f"  Avg price divergence: {divergence.mean():.4f}%")

def analyze_bollinger_patterns(datasets):
    """Phase 9: Bollinger Bands analysis"""
    print("\n" + "="*60)
    print("PHASE 9: BOLLINGER BANDS ANALYSIS")
    print("="*60 + "\n")
    
    windows = [5, 15, 30]
    
    for name, df in datasets.items():
        print(f"\n--- {name} Bollinger Bands Analysis ---")
        
        df_valid = df.dropna(subset=['spread_ma_5', 'spread_bb_position_5'])
        df_opportunities = df_valid[df_valid['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            print("\nBollinger Band Position Analysis:")
            for window in windows:
                position_col = f'spread_bb_position_{window}'
                upper_col = f'spread_bb_upper_{window}'
                lower_col = f'spread_bb_lower_{window}'
                
                if position_col in df_opportunities.columns:
                    avg_position = df_opportunities[position_col].mean()
                    median_position = df_opportunities[position_col].median()
                    
                    # Band breakouts
                    above_upper = (df_opportunities['spread_close_pct'] > df_opportunities[upper_col]).mean() * 100
                    below_lower = (df_opportunities['spread_close_pct'] < df_opportunities[lower_col]).mean() * 100
                    
                    print(f"\n  Window {window}:")
                    print(f"    Avg BB position: {avg_position:.3f} (0=lower band, 1=upper band)")
                    print(f"    Median BB position: {median_position:.3f}")
                    print(f"    Opportunities above upper band: {above_upper:.1f}%")
                    print(f"    Opportunities below lower band: {below_lower:.1f}%")
                    
                    # Band width during opportunities
                    band_width = df_opportunities[upper_col] - df_opportunities[lower_col]
                    print(f"    Avg band width: {band_width.mean():.4f}%")

def analyze_persistence_patterns(datasets):
    """Phase 10: Persistence and lag pattern analysis"""
    print("\n" + "="*60)
    print("PHASE 10: PERSISTENCE & LAG PATTERNS ANALYSIS")
    print("="*60 + "\n")
    
    lags = [1, 5, 10, 30]
    windows = [5, 15, 30]
    
    for name, df in datasets.items():
        print(f"\n--- {name} Persistence Analysis ---")
        
        print("\n1. Opportunity Persistence (Autocorrelation):")
        for lag in lags:
            lag_col = f'is_real_opportunity_lag_{lag}'
            if lag_col in df.columns:
                df_valid = df.dropna(subset=[lag_col])
                
                # Conditional probability: P(opportunity now | opportunity lag_N ago)
                had_opportunity_before = df_valid[df_valid[lag_col] == 1]
                if len(had_opportunity_before) > 0:
                    prob_now = had_opportunity_before['is_real_opportunity'].mean() * 100
                    baseline = df_valid['is_real_opportunity'].mean() * 100
                    
                    print(f"  Lag {lag}: {prob_now:.2f}% (baseline: {baseline:.2f}%) - " + 
                          f"{'HIGHER' if prob_now > baseline else 'LOWER'} than baseline")
        
        print("\n2. Rolling Opportunity Counts:")
        df_opportunities = df[df['is_real_opportunity'] == 1]
        if not len(df_opportunities):
            print("  No real opportunities found.")
        else:
            for window in windows:
                count_col = f'real_opportunities_in_last_{window}'
                if count_col in df_opportunities.columns:
                    avg_count = df_opportunities[count_col].mean()
                    max_count = df_opportunities[count_col].max()
                    print(f"  Last {window} min: Avg={avg_count:.2f}, Max={int(max_count)}")
        
        print("\n3. Spread Lag Correlation:")
        for lag in [1, 5, 10]:
            lag_col = f'spread_lag_{lag}'
            if lag_col in df.columns:
                df_valid = df.dropna(subset=[lag_col, 'spread_close_pct'])
                correlation = df_valid['spread_close_pct'].corr(df_valid[lag_col])
                print(f"  Spread vs {lag}-min lag: {correlation:.4f}")
        
        print("\n4. Spread Change Patterns:")
        for lag in [1, 5]:
            diff_col = f'spread_diff_from_lag_{lag}'
            if diff_col in df.columns:
                df_opportunities = df[df['is_real_opportunity'] == 1]
                df_opportunities_valid = df_opportunities.dropna(subset=[diff_col])
                
                if len(df_opportunities_valid) > 0:
                    avg_diff = df_opportunities_valid[diff_col].mean()
                    increasing = (df_opportunities_valid[diff_col] > 0).mean() * 100
                    print(f"  {lag}-min diff during opportunities: {avg_diff:.4f}% " +
                          f"({increasing:.1f}% increasing)")

def analyze_rolling_statistics(datasets):
    """Phase 11: Rolling statistics analysis"""
    print("\n" + "="*60)
    print("PHASE 11: ROLLING STATISTICS ANALYSIS")
    print("="*60 + "\n")
    
    windows = [5, 15, 30]
    
    for name, df in datasets.items():
        print(f"\n--- {name} Rolling Stats Analysis ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        print("\n1. Spread Volatility (Rolling Std):")
        for window in windows:
            std_col = f'spread_rolling_std_{window}'
            if std_col in df_opportunities.columns:
                df_valid = df_opportunities.dropna(subset=[std_col])
                if len(df_valid) > 0:
                    avg_std = df_valid[std_col].mean()
                    median_std = df_valid[std_col].median()
                    print(f"  Window {window}: Avg={avg_std:.4f}%, Median={median_std:.4f}%")
        
        print("\n2. Spread Range (Max - Min):")
        for window in windows:
            range_col = f'spread_range_{window}'
            if range_col in df_opportunities.columns:
                df_valid = df_opportunities.dropna(subset=[range_col])
                if len(df_valid) > 0:
                    avg_range = df_valid[range_col].mean()
                    print(f"  Window {window}: Avg range={avg_range:.4f}%")
        
        print("\n3. Z-Score Analysis:")
        for window in windows:
            zscore_col = f'spread_zscore_{window}'
            if zscore_col in df_opportunities.columns:
                df_valid = df_opportunities.dropna(subset=[zscore_col])
                if len(df_valid) > 0:
                    avg_zscore = df_valid[zscore_col].mean()
                    median_zscore = df_valid[zscore_col].median()
                    
                    # High z-score means unusual spread
                    high_zscore = (df_valid[zscore_col].abs() > 2).mean() * 100
                    
                    print(f"  Window {window}:")
                    print(f"    Avg z-score: {avg_zscore:.3f}, Median: {median_zscore:.3f}")
                    print(f"    Opportunities with |z-score| > 2: {high_zscore:.1f}%")
        
        print("\n4. Volume Rolling Statistics:")
        for window in [5, 15]:
            buy_std_col = f'volume_buy_rolling_std_{window}'
            sell_std_col = f'volume_sell_rolling_std_{window}'
            
            if buy_std_col in df_opportunities.columns:
                df_valid = df_opportunities.dropna(subset=[buy_std_col, sell_std_col])
                if len(df_valid) > 0:
                    avg_buy_std = df_valid[buy_std_col].mean()
                    avg_sell_std = df_valid[sell_std_col].mean()
                    print(f"  Window {window}: Buy Std={avg_buy_std:.2f}, Sell Std={avg_sell_std:.2f}")

def analyze_feature_correlations(datasets):
    """Phase 12: Feature correlation with opportunities"""
    print("\n" + "="*60)
    print("PHASE 12: FEATURE CORRELATION ANALYSIS")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Correlation Analysis ---")
        
        # Select numeric features only
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target variables and raw exchange data
        exclude_patterns = ['is_opportunity', 'is_real_opportunity', ':close', ':open', ':high', ':low', ':volume',
                           'min_close', 'max_close', 'min_low', 'max_high', 'time']
        
        features = [col for col in numeric_features if not any(pattern in col for pattern in exclude_patterns)]
        
        # Calculate correlations with target
        if 'is_real_opportunity' in df.columns and len(features) > 0:
            correlations = {}
            for feature in features:
                df_valid = df.dropna(subset=[feature, 'is_real_opportunity'])
                if len(df_valid) > 1:  # Need enough data
                    corr = df_valid[feature].corr(df_valid['is_real_opportunity'])
                    if not np.isnan(corr):
                        correlations[feature] = corr
            
            # Sort by absolute correlation
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print("\nTop 15 Features Correlated with Opportunities:")
            for i, (feature, corr) in enumerate(sorted_corr[:15], 1):
                print(f"  {i:2d}. {feature:40s}: {corr:+.4f}")
            
            print("\nBottom 10 (Least Correlated):")
            for i, (feature, corr) in enumerate(sorted_corr[-10:], 1):
                print(f"  {i:2d}. {feature:40s}: {corr:+.4f}")

def analyze_cross_exchange_ratios(datasets):
    """Phase 13: Cross-exchange price ratio analysis"""
    print("\n" + "="*60)
    print("PHASE 13: CROSS-EXCHANGE PRICE RATIO ANALYSIS")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Price Ratio Analysis ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if 'price_ratio_buy_sell' in df_opportunities.columns:
            df_valid = df_opportunities.dropna(subset=['price_ratio_buy_sell'])
            
            if len(df_valid) > 0:
                print("\n1. Buy-Sell Price Ratio:")
                avg_ratio = df_valid['price_ratio_buy_sell'].mean()
                median_ratio = df_valid['price_ratio_buy_sell'].median()
                min_ratio = df_valid['price_ratio_buy_sell'].min()
                max_ratio = df_valid['price_ratio_buy_sell'].max()
                
                print(f"  Average: {avg_ratio:.6f}")
                print(f"  Median: {median_ratio:.6f}")
                print(f"  Range: {min_ratio:.6f} - {max_ratio:.6f}")
                print(f"  Spread % = (ratio - 1) * 100: {(avg_ratio - 1) * 100:.4f}%")
        
        if 'avg_price_ratio' in df_opportunities.columns:
            df_valid = df_opportunities.dropna(subset=['avg_price_ratio', 'max_price_ratio', 
                                                       'min_price_ratio', 'price_ratio_std'])
            
            if len(df_valid) > 0:
                print("\n2. Cross-Exchange Ratio Statistics:")
                print(f"  Avg of all ratios: {df_valid['avg_price_ratio'].mean():.6f}")
                print(f"  Avg max ratio: {df_valid['max_price_ratio'].mean():.6f}")
                print(f"  Avg min ratio: {df_valid['min_price_ratio'].mean():.6f}")
                print(f"  Avg std of ratios: {df_valid['price_ratio_std'].mean():.6f}")
                
                # High std = high price dispersion across exchanges
                high_dispersion = (df_valid['price_ratio_std'] > df_valid['price_ratio_std'].median()).mean() * 100
                print(f"  Opportunities with high price dispersion: {high_dispersion:.1f}%")

def analyze_price_position(datasets):
    """Phase 14: Price position within high-low range"""
    print("\n" + "="*60)
    print("PHASE 14: PRICE POSITION ANALYSIS")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Price Position Analysis ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if 'price_position_buy_exchange' in df_opportunities.columns:
            df_valid = df_opportunities.dropna(subset=['price_position_buy_exchange', 
                                                       'price_position_sell_exchange'])
            
            if len(df_valid) > 0:
                buy_position = df_valid['price_position_buy_exchange'].mean()
                sell_position = df_valid['price_position_sell_exchange'].mean()
                
                print("\nPrice Position (0=low of day, 1=high of day):")
                print(f"  Buy exchange avg position: {buy_position:.3f}")
                print(f"  Sell exchange avg position: {sell_position:.3f}")
                print(f"  Position difference: {sell_position - buy_position:.3f}")
                
                # Categorize positions
                buy_near_low = (df_valid['price_position_buy_exchange'] < 0.3).mean() * 100
                buy_near_high = (df_valid['price_position_buy_exchange'] > 0.7).mean() * 100
                sell_near_low = (df_valid['price_position_sell_exchange'] < 0.3).mean() * 100
                sell_near_high = (df_valid['price_position_sell_exchange'] > 0.7).mean() * 100
                
                print(f"\n  Buy exchange near daily low (<0.3): {buy_near_low:.1f}%")
                print(f"  Buy exchange near daily high (>0.7): {buy_near_high:.1f}%")
                print(f"  Sell exchange near daily low (<0.3): {sell_near_low:.1f}%")
                print(f"  Sell exchange near daily high (>0.7): {sell_near_high:.1f}%")


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
            add_volatility_features(df)
            add_price_change_features(df)
            add_rolling_stats(df)
            add_moving_averages(df)
            add_bollinger_bands(df)
            add_rate_change_features(df) 
            add_cross_ex_price_ratio(df)
            add_lag_features(df) 
        
        print("✅ Features added!\n")
        save_featured_data()
        
    elif user_option == "ANALYZE":
        
        datasets = load_featured_data()
        
        if not datasets:
            print("❌ No featured data found. Run feature engineering first!")
            return
    
        # Run all analysis phases
        # print("\n=== BASIC ANALYSIS ===")
        # analyze_opportunity_frequency(datasets) #is_opportunity, is_real_oppt, spread_close_pct
        # analyze_temporal_patterns(datasets) # day_of_week, is_weekend, overlap_hours, is_real_oppt
        # analyze_exchange_patterns(datasets) # is_real_oppt, buy_exchange, sell_exchange, spread_close_pct
        # analyze_volume_liquidity(datasets) # is_real_oppt, min_volume, volume_ratio
        # analyze_risk_factors(datasets) # is_real_oppt, volatility_avg, oppt_gap
        # estimate_profitability(datasets) #is_real_oppt, spread_close_pct
        
        # print("\n\n=== ADVANCED FEATURE ANALYSIS ===")
        # analyze_momentum_indicators(datasets) # MAs, EMAs, rate_change, price_change
        # analyze_bollinger_patterns(datasets) # BB position, bands, breakouts
        # analyze_persistence_patterns(datasets) # lag features, rolling counts, autocorrelation
        # analyze_rolling_statistics(datasets) # rolling std, range, z-scores
        analyze_feature_correlations(datasets) # correlation with target
        # analyze_cross_exchange_ratios(datasets) # price ratios across exchanges
        # analyze_price_position(datasets) # price position in high-low range
            
    else:
        print("❌ Invalid option! Please enter 'ADD' or 'ANALYZE'")


if __name__ == "__main__":
    main()
