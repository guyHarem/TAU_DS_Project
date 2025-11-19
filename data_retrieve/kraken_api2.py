import requests
import pandas as pd

# Define Kraken API endpoint for historical data (OHLC)
url = "https://api.kraken.com/0/public/OHLC"

# Set parameters (pair = BTCUSD, interval = 3600s = 1 hour)
params = {
    'pair': 'XBTUSD',  # BTC to USD
    'interval': 60  # 1 hour
}

# Fetch data
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    
    # Kraken returns data in a different structure, need to parse it
    ohlc_data = data['result']['XXBTZUSD']
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                          'close_time', 'quote_asset_volume', 'number_of_trades', 
                                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Save to CSV
    df.to_csv('kraken_btc_usd_1h.csv', index=False)
    print("Data saved to kraken_btc_usd_1h.csv")
else:
    print("Error fetching data from Kraken API")