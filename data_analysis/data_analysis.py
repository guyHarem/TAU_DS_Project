import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the combined CSV files - mode 1 only!
data_path = '../data'
btcusd_data = pd.read_csv(f'{data_path}/combined_BTCUSD_data.csv')
ethusd_data = pd.read_csv(f'{data_path}/combined_ETHUSD_data.csv')
dogeusd_data = pd.read_csv(f'{data_path}/combined_DOGEUSD_data.csv')
linkusd_data = pd.read_csv(f'{data_path}/combined_LINKUSD_data.csv')
solusd_data = pd.read_csv(f'{data_path}/combined_SOLUSD_data.csv')
xrpusd_data = pd.read_csv(f'{data_path}/combined_XRPUSD_data.csv')

data_frames = [btcusd_data, ethusd_data, dogeusd_data, linkusd_data, solusd_data, xrpusd_data]


### DATA CLEANING - DO WE WANT TO DISCARD EVERY LINE THAT IS NOT FULL ALREADY HERE ?? ###

# Change time string from CSV to time type
for df in data_frames:
    df['time'] = pd.to_datetime(df['time'])

exchanges = ["BINANCE","BITFINEX","COINBASE","GATEIO","MEXC","KRAKEN"]

original_features = ["high","low","open","close","volume"]


## FEATURE ENGINEREEING ##

def add_close_spread(df, exchanges):
    return

def add_time_features(df): # Section 2
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df[f'overlap_hours'] = df[f'hour'].apply(lambda x: 1 if 19 <= x <= 21 else 0)  # Example: active trading hours
    return df

def add_volatility_features(df, exchanges): # Section 3
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
    return df

def price_change_features(df, exchanges): #Section 4
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
    return df

def add_moving_averages(df, windows=[5, 15, 30]): # Section 5
    for window in windows:
        df[f'spread_ma_{window}'] = df[f'spread_close_pct'].rolling(window=window).mean()
        df[f'volume_ma_buy_{window}'] = df[f'volume_buy_exchange'].rolling(window=window).mean()
        df[f'volume_ma_sell_{window}'] = df[f'volume_sell_exchange'].rolling(window=window).mean()
        df[f'spread_ema_{window}'] = df[f'spread_close_pct'].ewm(span=window, adjust=False).mean()
    return df
    



# Close spread #

for line in btcusd_data: ## add more dfs later
    print("hekko")


    
    







def main():
    print("hello world")

if __name__ == "__main__":
    main()
   
















