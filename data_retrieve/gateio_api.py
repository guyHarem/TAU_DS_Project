import requests
import pandas as pd
import time
from datetime import datetime, timezone

def fetch_data(currency, start_date, end_date):
    """
    Fetch historical 1-minute kline data from Gate.io for a given currency pair and time range.
    Args:
        currency (str): e.g. "BTC/USD"
        start_date (str): "YYYY-MM-DD HH:MM" (UTC)
        end_date (str): "YYYY-MM-DD HH:MM" (UTC)
    Returns:
        pd.DataFrame: columns = ["time", "open", "high", "low", "close", "volume"]
    """
    # Parse currency pair for Gate.io (BTC_USDT format)
    base, quote = currency.split('/')
    if quote == "USD":
        symbol = f"{base}_USDT"
    else:
        symbol = f"{base}_{quote}"

    # Convert to seconds timestamp (UTC)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    url = "https://api.gateio.ws/api/v4/spot/candlesticks"
    all_data = []
    limit = 1000  # Gate.io max per request

    curr_start = start_ts
    while curr_start < end_ts:
        # Calculate the end for this chunk (max 1000 minutes ahead)
        chunk_end = min(curr_start + (limit - 1) * 60, end_ts)
        params = {
            "currency_pair": symbol,
            "interval": "1m",
            "from": curr_start,
            "to": chunk_end,
            "limit": limit
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        # Next start is last candle's close time + 60 seconds
        last_time = int(data[-1][0])
        curr_start = last_time + 60
        if len(data) < limit:
            break
        time.sleep(0.2)  # avoid rate limits

    if not all_data:
        print("Gate.io returned no data")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    # Gate.io returns: [timestamp, volume, close, high, low, open, turnover, is_final]
    df = pd.DataFrame(
        all_data,
        columns=["time", "volume", "close", "high", "low", "open", "turnover", "is_final"]
    )
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df["time"] = df["time"].dt.floor("min")
    df["time"] = df["time"].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df[["time", "open", "high", "low", "close", "volume"]]

    print(f"Gate.io: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        df = fetch_data(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"Retrieved {len(df)} entries")
    else:
        df = fetch_data("BTC/USD", "2025-12-01 00:00", "2025-12-01 00:05")
        print(f"Retrieved {len(df)} entries")