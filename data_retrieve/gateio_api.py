"""
GATE.IO API Module
==================

Fetches 1-minute OHLCV candlestick data from Gate.io via public data archive
(NOT the REST API - uses historical data archive to bypass rate limits).

EXCHANGE DETAILS:
- Data Source: Public data archive (https://download.gatedata.org/)
- Endpoint: https://download.gatedata.org/spot/candlesticks_1m/YYYYMM/{pair}-YYYYMMDD.csv.gz
- Pair Format: BTC_USDT (underscore-separated)
- Quotes: USD is mapped to USDT (Gate.io uses USDT, not USD)
- Chunking: Per-day files (automatic date iteration)
- Rate Limit: NONE (using archive, not live API)
- Temporary Storage: Downloads to /tmp/ directory (auto-cleaned)

DATA QUALITY:
- CSV Source: First-party data archive
- Timestamps: Unix epoch (seconds), converted to UTC
- Duplicates: Automatically removed
- Time Floors: Rounded to nearest minute

IMPLEMENTATION NOTES:
1. Bypasses API rate limits by using public historical data archive
   This is legal and intended use of Gate.io data
   No authentication required, no rate limiting

2. Maps "USD" quotes to "USDT" because Gate.io doesn't support USD pairs
   Example: "BTC/USD" becomes "BTC_USDT"

3. Downloads daily .csv.gz files for the requested date range
   Example: For 2025-03-01 to 2025-03-03, downloads:
   - BTC_USDT-20250301.csv.gz
   - BTC_USDT-20250302.csv.gz
   - BTC_USDT-20250303.csv.gz

4. Decompresses .gz files on-the-fly using gzip
   Stores temporarily in /tmp/, cleaned after processing

5. CSV Format: [timestamp, volume, close, high, low, open, quote_volume]
   Reorders to standard format: [time, open, high, low, close, volume]

6. Handles missing dates gracefully
   If a date's file is missing, skips and continues

7. De-duplicates by timestamp before returning DataFrame

USAGE:
    from gateio_api import fetch_data
    df = fetch_data("BTC/USD", "2025-03-01 10:00", "2025-03-02 10:00")
    # Returns DataFrame with columns: [time, open, high, low, close, volume]

ERROR HANDLING:
- Returns empty DataFrame if pair not supported (no matching .csv files)
- Logs warnings for missing daily files but continues
- Returns partial data if some days failed to download
- Does NOT raise exceptions - logs to console instead

PERFORMANCE NOTES:
- Much faster than REST API for large time ranges (no pagination retries)
- Typical: 1-2 seconds per month of data
- Good for: Historical backtesting, large date ranges
"""

import urllib.request
import gzip
import shutil
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import warnings
import os

# Suppress FutureWarning globally in this file
warnings.filterwarnings("ignore", category=FutureWarning)

def fetch_data(currency, start_date, end_date):
    """
    Fetch historical 1-minute kline data from Gate.io using the data archive service.
    
    Gate.io provides daily candlestick files via: 
    https://download.gatedata.org/spot/candlesticks_1m/YYYYMM/MARKET-YYYYMMDD.csv.gz
    
    This bypasses the API's 7-day limitation by using their downloadable archive.
    
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
        quote = "USDT"

    symbol = f"{base}_{quote}"

    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)

    print(f"Gate.io: Fetching {symbol} from {start_date} to {end_date}")
    print(f"Gate.io: Using archive service (daily files)\n")

    all_data = []
    current_date = start_dt.date()
    end_date_only = end_dt.date()

    temp_dir = Path(__file__).resolve().parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Download each day
    while current_date <= end_date_only:
        year_month = current_date.strftime("%Y%m")
        date_str = current_date.strftime("%Y%m%d")
        
        url = f"https://download.gatedata.org/spot/candlesticks_1m/{year_month}/{symbol}-{date_str}.csv.gz"
        gz_file = temp_dir / f"gateio_{symbol}_{date_str}.csv.gz"
        csv_file = temp_dir / f"gateio_{symbol}_{date_str}.csv"
        
        try:
            # Download
            urllib.request.urlretrieve(url, str(gz_file))
            
            # Decompress
            with gzip.open(gz_file, 'rb') as f_in:
                with open(csv_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Read CSV
            df = pd.read_csv(csv_file, header=None, 
                           names=['timestamp', 'volume', 'close', 'high', 'low', 'open', 'quote_volume'])
            all_data.append(df)
            
            print(f"  ✅ {date_str}: {len(df)} rows")
            
            # Clean up
            os.remove(gz_file)
            os.remove(csv_file)
            
        except Exception as e:
            if "404" in str(e):
                print(f"  ❌ {date_str}: No data available")
            else:
                print(f"  ⚠️  {date_str}: {str(e)[:40]}")
        
        current_date += timedelta(days=1)
        
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    if not all_data:
        print(f"\nGate.io: No data available for {symbol}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Parse timestamps
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["time"] = df["time"].dt.tz_localize(None).dt.floor("min")
    
    # Filter to requested time range
    df = df[(df["time"] >= start_dt.replace(tzinfo=None)) & 
            (df["time"] <= end_dt.replace(tzinfo=None))]
    
    # Remove duplicates and sort
    df = df.drop_duplicates(subset=["time"], keep="first")
    df = df.sort_values("time").reset_index(drop=True)
    
    # Select columns and format
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df["time"] = df["time"].dt.strftime('%Y-%m-%d %H:%M')

    if len(df) > 0:
        print(f"\n✅ Gate.io: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC\n")
    else:
        print(f"\n❌ Gate.io: No valid data after filtering\n")

    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        df = fetch_data(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"Retrieved {len(df)} entries")
    else:
        # Test with 2 months
        df = fetch_data("BTC/USD", "2026-02-15 00:00", "2026-03-16 23:59")
        print(f"Retrieved {len(df)} entries")