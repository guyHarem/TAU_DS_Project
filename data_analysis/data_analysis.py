import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
        lambda row: (row[f"{row['buy_exchange']}:close"] - row[f"{row['buy_exchange']}:low"]) / 
                    (row[f"{row['buy_exchange']}:high"] - row[f"{row['buy_exchange']}:low"]),
        axis=1
    )
    df['price_position_sell_exchange'] = df.apply(
        lambda row: (row[f"{row['sell_exchange']}:close"] - row[f"{row['sell_exchange']}:low"]) / 
                    (row[f"{row['sell_exchange']}:high"] - row[f"{row['sell_exchange']}:low"]),
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
        df[f'spread_bb_position_{window}'] = (df['spread_close_pct'] - df[f'spread_bb_lower_{window}']) / (df[f'spread_bb_upper_{window}'] - df[f'spread_bb_lower_{window}'])

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
        df[f'spread_zscore_{window}'] = (df['spread_close_pct'] - rolling_mean) / rolling_std

def add_rate_change_features(df): # Section 8
    df[f'spread_rate_change'] = df[f'spread_close_pct'] - df[f'spread_close_pct'].shift(1)
    df[f'spread_rate_change_pct'] = df['spread_rate_change'] / df[f'spread_close_pct'].shift(1) * 100
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


def main():
    print("=== ADDING FEATURES ===\n")
    
    for df in data_frames:
        add_close_spread(df)
        add_volume_features(df)
        add_high_low_spread(df)
        add_time_features(df)
        # add_volatility_features(df)
        add_price_change_features(df)
        add_moving_averages(df)
        add_bollinger_bands(df)
        add_rolling_stats(df)
        add_rate_change_features(df) 
        add_cross_ex_price_ratio(df)
        # add_lag_features(df)
    
    print("✅ Features added!\n")
    
    # Verify opportunity flags
    print("BTC Opportunity Statistics:")
    print(f"Total opportunities (≥ 0.50%): {btcusd_data['is_opportunity'].sum()}")
    print(f"Real opportunities (≥ 0.60%): {btcusd_data['is_real_opportunity'].sum()}")
    print(f"Percentage with opportunity: {btcusd_data['is_opportunity'].mean() * 100:.2f}%")
    print(f"Percentage with real opportunity: {btcusd_data['is_real_opportunity'].mean() * 100:.2f}%")
    
    # Verify rolling stats
    print("\nBTC Rolling Opportunity Counts (5-min window):")
    print(btcusd_data[['time', 'is_opportunity', 'is_real_opportunity', 
                       'opportunities_in_last_5', 'real_opportunities_in_last_5']].head(10))
    
    # Save featured data
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


if __name__ == "__main__":
    main()

















