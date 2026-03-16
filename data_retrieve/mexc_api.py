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
    base, quote = currency.split('/')
    if quote == "USD":
        quote = "USDT"

    symbol = f"{base}{quote}"
    is_reversed = False

    # Convert to milliseconds timestamp (UTC)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    url = "https://api.mexc.com/api/v3/klines"
    all_data = []
    limit = 1000  # MEXC max per request

    curr_start = start_ms

    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": curr_start,
        "endTime": end_ms,
        "limit": limit
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"MEXC: {base}{quote} not found, trying the reverse pair.")
        symbol = f"{quote}{base}"
        params["symbol"] = symbol
        is_reversed = True
        resp = requests.get(url, params=params)

    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"MEXC: HTTP error {resp.status_code}. No data available for {base}/{quote} or reversed pair.")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    data = resp.json()
    if not data:
        print(f"MEXC: No data returned for {base}/{quote}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    
    all_data.extend(data)
    last_time = data[-1][0]
    curr_start = last_time + 60_000
    time.sleep(0.2)  # avoid rate limits

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
        print(f"MEXC: No data collected for {base}/{quote}")
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
    # Floor to minute (removes seconds)
    df["time"] = df["time"].dt.floor("min")

    if is_reversed:
        df[["open", "high", "low", "close"]] = 1 / df[["open", "high", "low", "close"]].astype(float)
        df["volume"] = df["volume"].astype(float) * df["close"].astype(float)
    
    df = df[["time", "open", "high", "low", "close", "volume"]]
    
    # Convert time to string without seconds (HH:MM format)
    df["time"] = df["time"].dt.strftime('%Y-%m-%d %H:%M')

    if len(df) > 0:
        print(f"MEXC: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    else:
        print(f"MEXC: No valid data after processing")

    return df

if __name__ == "__main__":
    if len(sys.argv) == 4:
        df = fetch_data(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"Retrieved {len(df)} entries")
    else:
        df = fetch_data("BTC/USD", "2022-03-15 01:00", "2022-03-15 02:00")
        print(f"Retrieved {len(df)} entries")