import requests
import pandas as pd
import sys
from datetime import datetime, timedelta

def fetch_data(currency, start_date, end_date):
    # Parse currency pair - Kraken uses specific naming conventions
    base, quote = currency.split('/')
    
    # Kraken pair mapping
    if base == "BTC" and quote == "USD":
        pair = "XXBTZUSD"
    elif base == "ETH" and quote == "USD":
        pair = "XETHZUSD"
    else:
        # Try generic format
        if base == "BTC":
            base = "XBT"
        pair = f"X{base}Z{quote}"
    
    # Convert to seconds timestamp
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
    start_s = int(start_dt.timestamp())
    
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        'pair': pair,
        'interval': 1,  # 1 minute
        'since': start_s
    }
    
    print(f"Kraken request - Pair: {pair}, Since: {start_s} ({start_date})")
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    print(f"Kraken response keys: {data.keys()}")
    
    # Check for errors in response
    if 'error' in data and data['error']:
        error_msg = data['error'][0] if data['error'] else 'Unknown error'
        print(f"Kraken API Error: {error_msg}")
        
        # Return empty DataFrame instead of raising exception
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    if 'result' not in data:
        print(f"Unexpected Kraken response: {data}")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    print(f"Result keys: {data['result'].keys()}")
    
    # Get the pair key (might be different from what we sent)
    result_keys = [k for k in data['result'].keys() if k != 'last']
    if not result_keys:
        print("No pair data in result")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    pair_key = result_keys[0]
    ohlc_data = data['result'][pair_key]
    
    print(f"Pair key: {pair_key}, Data length: {len(ohlc_data)}")
    
    if not ohlc_data:
        print("Kraken returned empty data array")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    # Kraken returns 720 max, but we'll limit to match time range
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
    
    df = pd.DataFrame(
        ohlc_data,
        columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
    )
    
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Filter by end time
    df = df[df['time'] <= end_dt]
    
    # Keep only relevant columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    return df

if __name__ == "__main__":
    print("Testing Kraken data availability...\n")
    
    # Test different time ranges
    test_dates = [
        ("Recent (today)", (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"), datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("1 week ago", (datetime.now() - timedelta(days=7, hours=5)).strftime("%Y-%m-%d %H:%M"), (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M")),
        ("1 month ago", (datetime.now() - timedelta(days=30, hours=5)).strftime("%Y-%m-%d %H:%M"), (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M")),
        ("3 months ago", (datetime.now() - timedelta(days=90, hours=5)).strftime("%Y-%m-%d %H:%M"), (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M")),
        ("March 2022", "2022-03-15 00:00", "2022-03-15 05:00"),
    ]
    
    for label, start, end in test_dates:
        print(f"\n{'='*50}")
        print(f"Testing: {label}")
        print(f"{'='*50}")
        df = fetch_data("BTC/USD", start, end)
        print(f"Retrieved {len(df)} entries")
        if not df.empty:
            print(f"First timestamp: {df['time'].min()}")
            print(f"Last timestamp: {df['time'].max()}")