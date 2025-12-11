import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRADING_COST_PCT = 0.5
SAFETY_MARGIN_PCT = 0.1
REAL_OPPORTUNITY_THRESHOLD = TRADING_COST_PCT + SAFETY_MARGIN_PCT


# Load the combined CSV files - mode 1 only!
data_path = '../data'
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

exchanges = ["BINANCE","BITFINEX","COINBASE","GATEIO","MEXC","KRAKEN"]

original_features = ["high","low","open","close","volume"]


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
    
    # is oppurtinity flag
    df['is_opportunity_flag'] = (df['spread_close_pct'] >= REAL_OPPORTUNITY_THRESHOLD).astype(int)
    
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





def main():
    print("=== ADDING FEATURES ===\n")
    
    for df in data_frames:
        add_close_spread(df)
        add_volume_features(df)
        add_high_low_spread(df)
        add_time_features(df)
        add_volatility_features(df)
        add_price_change_features(df)
        add_moving_averages(df)
        
        
    
    print("✅ Done!\n")
    
    # Verify close spreads
    print("BTC Close Spread Sample:")
    print(btcusd_data[['time', 'spread_close_pct', 'buy_exchange', 'sell_exchange']].head())
    
    # Verify high-low spreads
    print("\nBTC High-Low Spread Sample:")
    print(btcusd_data[['time', 'spread_highlow_pct', 'opportunity_gap', 'high_exchange', 'low_exchange']].head())
    
    # Statistics
    print("\nOpportunity Gap Statistics:")
    print(btcusd_data['opportunity_gap'].describe())


if __name__ == "__main__":
    main()

















