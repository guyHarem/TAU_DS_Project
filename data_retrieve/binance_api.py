import requests
import pandas as pd
import sys
import time
from datetime import datetime, timezone

def fetch_data(currency, start_date, end_date):
    """
    Fetch historical 1-minute kline data from Binance for a given currency pair and time range.
    Handles pagination internally to retrieve full time range in 1000-record chunks.
    
    Args:
        currency (str): e.g. "BTC/USD"
        start_date (str): "YYYY-MM-DD HH:MM" (UTC)
        end_date (str): "YYYY-MM-DD HH:MM" (UTC)
    Returns:
        pd.DataFrame: columns = ["time", "open", "high", "low", "close", "volume"]
    """
    base, quote = currency.split('/')
    # Map USD to USDT for Binance
    if quote == "USD":
        quote = "USDT"
    
    # Convert to milliseconds timestamp (UTC)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    limit = 1000  # Binance max per request
    
    symbol = f"{base}{quote}"
    is_reversed = False
    curr_start = start_ms
    
    print(f"Binance: Fetching {symbol} from {start_date} to {end_date}")
    print(f"Binance: Using pagination (1000 records per request)\n")
    
    # Initial request to check if symbol exists
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": curr_start,
        "endTime": end_ms,
        "limit": limit
    }
    
    resp = requests.get(url, params=params)

    # If symbol not found, try reversed
    if resp.status_code != 200:
        symbol = f"{quote}{base}"
        params["symbol"] = symbol
        is_reversed = True
        resp = requests.get(url, params=params)

    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Binance: HTTP error {resp.status_code}. No data available for {base}/{quote} or reversed pair.")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    
    data = resp.json()
    if not data:
        print(f"Binance: No data returned for {base}/{quote}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    
    all_data.extend(data)
    last_time = data[-1][0]  # timestamp of last candle
    curr_start = last_time + 60_000  # Next start is last candle's time + 60 seconds
    print(f"  ✅ Request 1: {len(data)} records (up to {datetime.fromtimestamp(last_time/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')})")
    time.sleep(0.1)  # Respect rate limits

    # Paginate through remaining data
    request_count = 1
    while curr_start < end_ms:
        request_count += 1
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": curr_start,
            "endTime": end_ms,
            "limit": limit
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        if not data:
            break
        
        all_data.extend(data)
        last_time = data[-1][0]
        print(f"  ✅ Request {request_count}: {len(data)} records (up to {datetime.fromtimestamp(last_time/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')})")
        
        curr_start = last_time + 60_000
        if len(data) < limit:
            break
        
        time.sleep(0.1)  # Respect rate limits

    if not all_data:
        print(f"Binance: No data collected for {base}/{quote}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    
    cols = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_base", "taker_quote", "ignore"
    ]
    
    df = pd.DataFrame(all_data, columns=cols)
    
    # Convert to UTC datetime and remove timezone info
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
    
    # Floor timestamps to nearest minute
    df["time"] = df["time"].dt.floor("min")
    
    # Remove duplicates (shouldn't happen, but safety check)
    df = df.drop_duplicates(subset=["time"], keep="first")

    # If data is from reversed pair, invert prices
    if is_reversed:
        df[["open", "high", "low", "close"]] = 1 / df[["open", "high", "low", "close"]].astype(float)
        df["volume"] = df["volume"].astype(float) * df["close"].astype(float)
    
    # Keep only relevant columns
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    
    if len(df) > 0:
        print(f"\n✅ Binance: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    else:
        print(f"\n❌ Binance: No valid data after processing")

    return df

if __name__ == "__main__":
    if len(sys.argv) == 4:
        df = fetch_data(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"Retrieved {len(df)} entries")
        if not df.empty:
            print(df.head())
    else:
        # Test with 1 hour of data
        df = fetch_data("BTC/USD", "2022-03-15 01:00", "2022-03-15 02:00")
        print(f"Retrieved {len(df)} entries")
        if not df.empty:
            print(df.head())