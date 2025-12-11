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
    for
    
    
    
    
    



# Close spread #

for line in btcusd_data: ## add more dfs later
    print("hekko")


    
    







def main():
    print("hello world")

if __name__ == "__main__":
    main()
   
















