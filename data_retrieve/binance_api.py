import requests
import pandas as pd
import sys
from datetime import datetime, timezone

def fetch_data(currency, start_date, end_date):
    base, quote = currency.split('/')
    # Map USD to USDT for Binance
    if quote == "USD":
        quote = "USDT"
    symbol = f"{base}{quote}"
    
    # Convert to milliseconds timestamp (UTC)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",  # 1 minute
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000  # Binance allows up to 1000
    }
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    cols = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_base", "taker_quote", "ignore"
    ]
    
    df = pd.DataFrame(data, columns=cols)
    
    # Convert to UTC datetime and remove timezone info
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
    
    # Floor timestamps to nearest minute
    df["time"] = df["time"].dt.floor("min")
    
    # Keep only relevant columns
    df = df[["time", "open", "high", "low", "close", "volume"]]
    
    print(f"Binance: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    
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