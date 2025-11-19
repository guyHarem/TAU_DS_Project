import requests
import pandas as pd
import sys
from datetime import datetime, timezone

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
    
    # Convert to seconds timestamp (UTC)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    start_s = int(start_dt.timestamp())
    
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        'pair': pair,
        'interval': 1,  # 1 minute
        'since': start_s
    }
    
    print(f"Requesting from Kraken API:")
    print(f"  URL: {url}")
    print(f"  Pair: {pair}")
    print(f"  Since: {start_s} ({start_date})")
    print(f"  Interval: 1 minute")
    
    resp = requests.get(url, params=params)
    print(f"  Response Status: {resp.status_code}")
    
    resp.raise_for_status()
    data = resp.json()
    
    print(f"  Response keys: {data.keys()}")
    
    # Check for errors in response
    if 'error' in data and data['error']:
        error_msg = data['error'][0] if data['error'] else 'Unknown error'
        print(f"  Kraken API Error: {error_msg}")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    if 'result' not in data:
        print(f"  No 'result' in response")
        print(f"  Full response: {data}")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    print(f"  Result keys: {data['result'].keys()}")
    
    # Get the pair key from result
    result_keys = [k for k in data['result'].keys() if k != 'last']
    if not result_keys:
        print("  No pair data in result")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    pair_key = result_keys[0]
    ohlc_data = data['result'][pair_key]
    
    print(f"  Pair key: {pair_key}")
    print(f"  Data received: {len(ohlc_data)} candles")
    
    if not ohlc_data:
        print("  Kraken returned empty data array")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    # Parse end date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    
    # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(
        ohlc_data,
        columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
    )
    
    # Convert to UTC datetime and remove timezone info
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_localize(None)
    
    # Filter by end time
    df = df[df['time'] <= end_dt]
    
    # Floor timestamps to nearest minute
    df['time'] = df['time'].dt.floor('min')
    
    # Keep only relevant columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"Kraken: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    
    return df

if __name__ == "__main__":
    print("="*70)
    print("KRAKEN API TEST - 1 HOUR OF MINUTE DATA")
    print("="*70)
    
    # Test with 1 hour of data from March 15, 2022
    test_start = "2022-03-15 01:00"
    test_end = "2022-03-15 02:00"
    
    print(f"\nTest Parameters:")
    print(f"  Currency: BTC/USD")
    print(f"  Start: {test_start} UTC")
    print(f"  End: {test_end} UTC")
    print(f"  Expected candles: 60 (1 per minute)")
    print()
    
    df = fetch_data("BTC/USD", test_start, test_end)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total entries retrieved: {len(df)}")
    
    if not df.empty:
        print(f"\nFirst 5 entries:")
        print(df.head())
        print(f"\nLast 5 entries:")
        print(df.tail())
        
        # Save to test CSV
        filename = "kraken_test_2022-03-15.csv"
        df.to_csv(filename, index=False)
        print(f"\n✓ Data saved to: {filename}")
        
        # Show some statistics
        print(f"\nData Statistics:")
        print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"  Open price range: ${float(df['open'].min()):.2f} - ${float(df['open'].max()):.2f}")
        print(f"  Close price range: ${float(df['close'].min()):.2f} - ${float(df['close'].max()):.2f}")
        print(f"  Total volume: {float(df['volume'].sum()):.6f} BTC")
    else:
        print("\n✗ No data retrieved!")
        print("\nPossible reasons:")
        print("  - Kraken doesn't have 1-minute historical data for March 2022")
        print("  - The 'since' parameter might be too old")
        print("  - Try a more recent date or larger interval (5min, 15min)")
    
    print("\n" + "="*70)