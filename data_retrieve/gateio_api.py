import urllib.request
import gzip
import shutil
import pandas as pd
from datetime import datetime, timezone, timedelta
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

    # Download each day
    while current_date <= end_date_only:
        year_month = current_date.strftime("%Y%m")
        date_str = current_date.strftime("%Y%m%d")
        
        url = f"https://download.gatedata.org/spot/candlesticks_1m/{year_month}/{symbol}-{date_str}.csv.gz"
        gz_file = f"/tmp/gateio_{symbol}_{date_str}.csv.gz"
        csv_file = f"/tmp/gateio_{symbol}_{date_str}.csv"
        
        try:
            # Download
            urllib.request.urlretrieve(url, gz_file)
            
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