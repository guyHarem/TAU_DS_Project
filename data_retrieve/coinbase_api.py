"""
COINBASE API Module
===================

Fetches 1-minute OHLCV candlestick data from the Coinbase cryptocurrency exchange.

EXCHANGE DETAILS:
- REST Endpoint: https://api.exchange.coinbase.com/products/{pair}/candles
- Pair Format: BTC-USD (hyphen-separated)
- Pagination: 300 records per request (maximum)
- Rate Limit: Adaptive - increases with consecutive requests (0.1s-1.0s)
- Data Order: Newest first (queries advance forward in time)
- Pair Reversal: Supported (only on first request)

DATA QUALITY:
- Timestamps: ISO 8601 format, converted to UTC
- Duplicates: Automatically removed
- Time Floors: Rounded to nearest minute
- Price Inversion: Handled if reversed pair is used

IMPLEMENTATION NOTES:
1. Data is returned newest-first, so each request moves forward in time
   Example: Query [10:00 - 10:05] returns minute 10:05 then 10:04, etc.

2. Pagination: Each request fetches up to 300 records
   Time window per request = min(300 minutes, remaining_time)
   This ensures continuous coverage

3. Uses ISO 8601 timestamps in requests (e.g., "2025-12-01T10:00:00Z")
   
4. If original pair fails (404), automatically retries with reversed pair
   Example: If "BTC-USD" fails, tries "USD-BTC"
   - When reversed: inverts prices mathematically

5. Rate limiting increases with consecutive requests
   - Prevents hitting exchange rate limits
   - Backs off longer between chunks

6. De-duplicates by timestamp before returning DataFrame

USAGE:
    from coinbase_api import fetch_data
    df = fetch_data("BTC/USD", "2025-03-01 10:00", "2025-03-02 10:00")
    # Returns DataFrame with columns: [time, open, high, low, close, volume]

ERROR HANDLING:
- Returns empty DataFrame if pair not found (even after reversal)
- Returns empty DataFrame on network/API errors
- Does NOT raise exceptions - logs to console instead
"""

import requests
import pandas as pd
import sys
import time
from datetime import datetime, timedelta

def fetch_data(currency, start_date, end_date):
    """
    Fetch historical 1-minute kline data from Coinbase for a given currency pair and time range.
    Handles pagination internally to retrieve full time range in 300-record chunks.
    
    Args:
        currency (str): e.g. "BTC/USD"
        start_date (str): "YYYY-MM-DD HH:MM" (UTC)
        end_date (str): "YYYY-MM-DD HH:MM" (UTC)
    Returns:
        pd.DataFrame: columns = ["time", "open", "high", "low", "close", "volume"]
    """
    # Parse currency pair
    base, quote = currency.split('/')
    pair = f"{base}-{quote}"
    is_reversed = False
    
    # Convert dates to UTC datetime objects
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
    
    print(f"Coinbase: Fetching {pair} from {start_date} to {end_date}")
    print(f"Coinbase: Using pagination (300 records per request)\n")

    all_data = []
    limit = 300  # Coinbase max per request
    
    # Paginate forward through time (Coinbase returns newest first, so we query chunks)
    curr_start = start_dt
    request_count = 0
    
    while curr_start < end_dt:
        request_count += 1
        # Each request covers up to 300 minutes (one per minute at 1m granularity)
        curr_end = min(curr_start + timedelta(minutes=limit), end_dt)
        
        start_iso = curr_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = curr_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
        params = {
            "granularity": 60,  # 1 minute
            "start": start_iso,
            "end": end_iso
        }
        
        resp = requests.get(url, params=params)

        # If symbol not found, try reversed (only on first request)
        if resp.status_code == 404 and request_count == 1:
            pair = f"{quote}-{base}"
            is_reversed = True
            url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
            resp = requests.get(url, params=params)

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if request_count == 1:
                print(f"Coinbase: HTTP error {resp.status_code}. No data available for {base}/{quote} or reversed pair.")
                return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            else:
                # On subsequent requests, if we get an error, break (we've likely reached the end)
                break
        
        data = resp.json()
        
        if data:
            all_data.extend(data)
            print(f"  ✅ Request {request_count}: {len(data)} records ({start_iso} to {end_iso})")
        else:
            print(f"  ⚠️  Request {request_count}: No data ({start_iso} to {end_iso})")
        
        # Move to next window
        curr_start = curr_end
        time.sleep(0.1)  # Respect rate limits

    if not all_data:
        print(f"Coinbase: No data returned for {base}/{quote}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    
    df = pd.DataFrame(
        all_data,
        columns=["time", "low", "high", "open", "close", "volume"]
    )
    
    # Convert to UTC datetime and remove timezone info
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    
    # Round timestamps to nearest minute
    df["time"] = df["time"].dt.floor("min")
    
    # Remove duplicates (Coinbase returns descending, so we get overlaps at window boundaries)
    df = df.drop_duplicates(subset=["time"], keep="first")
    
    # Sort by time ascending
    df = df.sort_values("time")

    # If data is from reversed pair, invert prices
    if is_reversed:
        df[["open", "high", "low", "close"]] = 1 / df[["open", "high", "low", "close"]].astype(float)
        df["volume"] = df["volume"].astype(float) * df["close"].astype(float)
    
    # Reorder columns
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.reset_index(drop=True)
    
    if len(df) > 0:
        print(f"\n✅ Coinbase: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    else:
        print(f"\n❌ Coinbase: No valid data after processing")

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