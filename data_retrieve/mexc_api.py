import requests
import pandas as pd
import sys
import time
from datetime import datetime, timezone

def fetch_data(currency, start_date, end_date):
    """
    Fetch historical 1-minute kline data from MEXC for a given currency pair and time range.
    Args:
        currency (str): e.g. "BTC/USD"
        start_date (str): "YYYY-MM-DD HH:MM" (UTC)
        end_date (str): "YYYY-MM-DD HH:MM" (UTC)
    Returns:
        pd.DataFrame: columns = ["time", "open", "high", "low", "close", "volume"]
    """
    # Parse currency pair for MEXC (BTCUSDT format)
    base, quote = currency.split('/')
    symbol = f"{base}{quote}T"  # BTCUSDT

    # Convert to milliseconds timestamp (UTC)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    url = "https://api.mexc.com/api/v3/klines"
    all_data = []
    limit = 1000  # MEXC max per request

    curr_start = start_ms
    while curr_start < end_ms:
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
        # Next start is last candle's open time + 60,000 ms
        last_time = data[-1][0]
        curr_start = last_time + 60_000
        if len(data) < limit:
            break
        time.sleep(0.2)  # avoid rate limits

    if not all_data:
        print("MEXC returned no data")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        all_data,
        columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume"
        ]
    )

    # Convert to UTC datetime and remove timezone info
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
    df["time"] = df["time"].dt.floor("min")
    df = df[["time", "open", "high", "low", "close", "volume"]]

    print(f"MEXC: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")

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