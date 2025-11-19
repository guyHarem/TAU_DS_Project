import requests
import pandas as pd
import sys
from datetime import datetime, timezone

def fetch_data(currency, start_date, end_date):
    # Parse currency pair
    base, quote = currency.split('/')
    symbol = f"t{base}{quote}"
    
    # Convert to milliseconds timestamp (UTC)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    url = f"https://api-pub.bitfinex.com/v2/candles/trade:1m:{symbol}/hist"
    params = {
        'start': start_ms,
        'end': end_ms,
        'limit': 10000  # Bitfinex allows up to 10000
    }
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    df = pd.DataFrame(
        data,
        columns=['time', 'open', 'close', 'high', 'low', 'volume']
    )
    
    # Convert to UTC datetime and remove timezone info
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
    
    # Floor timestamps to nearest minute
    df['time'] = df['time'].dt.floor('min')
    
    # Sort by time ascending
    df = df.sort_values('time')
    
    # Reorder columns to match other APIs
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"Bitfinex: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    
    return df

if __name__ == "__main__":
    if len(sys.argv) == 4:
        df = fetch_data(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"Retrieved {len(df)} entries")
        if not df.empty:
            print(df.head())
    else:
        df = fetch_data("BTC/USD", "2022-03-15 01:00", "2022-03-15 02:00")
        print(f"Retrieved {len(df)} entries")
        if not df.empty:
            print(df.head())