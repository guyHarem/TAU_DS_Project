import requests
import pandas as pd
import sys
from datetime import datetime

def fetch_data(currency, start_date, end_date):
    # Parse currency pair
    base, quote = currency.split('/')
    pair = f"{base}-{quote}"
    
    # Convert dates to ISO format (UTC)
    start_iso = datetime.strptime(start_date, "%Y-%m-%d %H:%M").strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = datetime.strptime(end_date, "%Y-%m-%d %H:%M").strftime("%Y-%m-%dT%H:%M:%SZ")
    
    url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
    params = {
        "granularity": 60,  # 1 minute
        "start": start_iso,
        "end": end_iso
    }
    
    print(f"Requesting from Coinbase API:")
    print(f"  URL: {url}")
    print(f"  Start: {start_iso}")
    print(f"  End: {end_iso}")
    print(f"  Granularity: 60 seconds (1 minute)")
    
    resp = requests.get(url, params=params)
    print(f"  Response Status: {resp.status_code}")
    
    resp.raise_for_status()
    data = resp.json()
    
    print(f"  Data received: {len(data)} candles")
    
    if not data or len(data) == 0:
        print(f"Coinbase returned no data")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    
    df = pd.DataFrame(
        data,
        columns=["time", "low", "high", "open", "close", "volume"]
    )
    
    # Convert to UTC datetime and remove timezone info
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    
    # Sort by time ascending (Coinbase returns newest first)
    df = df.sort_values("time")
    
    # Reorder columns
    df = df[["time", "open", "high", "low", "close", "volume"]]
    
    # Round timestamps to nearest minute
    df["time"] = df["time"].dt.floor("min")
    
    print(f"Coinbase: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    
    return df

if __name__ == "__main__":
    print("="*70)
    print("COINBASE API TEST - 1 HOUR OF MINUTE DATA")
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
        
        # Save to separate CSV
        filename = "coinbase_test_2022-03-15.csv"
        df.to_csv(filename, index=False)
        print(f"\n✓ Data saved to: {filename}")
        
        # Show some statistics
        print(f"\nData Statistics:")
        print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"  Open price range: ${df['open'].min():.2f} - ${df['open'].max():.2f}")
        print(f"  Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"  Total volume: {df['volume'].sum():.6f} BTC")
    else:
        print("\n✗ No data retrieved!")
    
    print("\n" + "="*70)