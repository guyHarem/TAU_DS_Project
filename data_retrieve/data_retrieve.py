"""
DATA RETRIEVAL ORCHESTRATOR
===========================

Master module that coordinates fetching cryptocurrency OHLCV data from 6 exchanges,
merges the data by timestamp, and saves to CSV.

ARCHITECTURE:
                            User Input (CLI)
                                  |
                    [Currencies + Time Range]
                                  |
                    +-------------+-----------+
                    |                         |
            Load Exchange Modules    Validate Inputs
                    |
        +---+---+---+---+---+---+
        |   |   |   |   |   |   |
      BIN BFX CBX GIO KRK MXC  (6 Exchange APIs)
        |   |   |   |   |   |   |
        +---+---+---+---+---+---+
                    |
            Fetch from each API
                    |
        +---+---+---+---+---+---+
        | DataFrame | DataFrame |...
        +---+---+---+---+---+---+
                    |
            Merge on 'time'
            (outer join)
                    |
            Save to CSV
                    |
    data/raw_data/combined_BTCUSD_data.csv

SUPPORTED EXCHANGES (Configurable):
- Coinbase (enabled by default)
- Binance (enabled by default)
- Bitfinex (enabled by default)
- Gate.io (enabled by default)
- Kraken (disabled by default - uncomment _apis dict to enable)
- MEXC (disabled by default - uncomment _apis dict to enable)

SUPPORTED CURRENCIES:
- Base Assets: BTC, ETH, DOGE, SOL, XRP, LINK
- Quote Asset: USD (automatically mapped to USDT for some exchanges)

DATA FLOW:
1. User selects currencies (e.g., "BTC,ETH,DOGE")
2. User specifies time range (start and end in UTC)
3. Script dynamically loads each exchange API module
4. For each currency pair, fetch data from all enabled exchanges:
   - Each API called with: currency="BTC/USD", start_date="2025-03-01 10:00", end_date="2025-03-02 10:00"
   - Each API returns: DataFrame with [time, open, high, low, close, volume]
5. All DataFrames merged via outer join on 'time' column
   - Columns prefixed with exchange name (e.g., BINANCE:open, COINBASE:close)
6. Combined DataFrame saved to: data/raw_data/combined_{BASE}{QUOTE}_data.csv

USAGE:

Interactive Mode (Recommended):
    python data_retrieve.py

This will prompt for:
    - Cryptocurrencies: "BTC,ETH,DOGE" (comma-separated)
    - Start date: "2025-03-01 10:00" (UTC)
    - End date: "2025-03-02 10:00" (UTC)
    - Confirmation: "y" to proceed

Output Example:
    === Cryptocurrency Data Retrieval ===
    Available currencies: BTC, ETH, DOGE, SOL, XRP, LINK
    Enter comma-separated list (e.g., BTC,ETH,DOGE):
    BTC,ETH

    --- Time Range (UTC) ---
    Format: YYYY-MM-DD HH:MM
    Enter start date (UTC): 2025-03-01 10:00
    Enter end date (UTC): 2025-03-02 10:00

    Proceed with data retrieval? (y/n): y

    === Fetching data for BTC/USD ===
    --- Fetching from COINBASE ---
    ✅ Coinbase: Retrieved 1440 entries
    --- Fetching from BINANCE ---
    ✅ Request 1: 1000 records
    ✅ Request 2: 440 records
    ✅ Binance: Retrieved 1440 entries
    ...
    --- Combining data from all exchanges ---
    ✓ Combined BTC/USD: 1440 rows from 4 exchanges
    ✓ Combined data saved to: ../data/raw_data/combined_BTCUSD_data.csv

OUTPUT FORMAT:

File: data/raw_data/combined_{BASE}{QUOTE}_data.csv
Example: combined_BTCUSD_data.csv

Columns:
    - time: UTC timestamp (YYYY-MM-DD HH:MM)
    - COINBASE:open, COINBASE:high, COINBASE:low, COINBASE:close, COINBASE:volume
    - BINANCE:open, BINANCE:high, BINANCE:low, BINANCE:close, BINANCE:volume
    - BITFINEX:open, BITFINEX:high, ... (repeated for each exchange)
    - GATEIO:open, GATEIO:high, ... (repeated for each exchange)

Sample Row:
    time,COINBASE:open,COINBASE:high,COINBASE:low,COINBASE:close,COINBASE:volume,...
    2025-03-01 10:00,54230.5,54245.3,54220.1,54235.8,12.5,...

CONFIGURATION:

Active Exchanges (edit _apis dict):
    _apis = {
        "coinbase": ...,      # Enabled
        "binance": ...,       # Enabled
        "bitfinex": ...,      # Enabled
        "gateio": ...,        # Enabled
        # "kraken": ...,       # Disabled (comment out to enable)
        # "mexc": ...,         # Disabled (comment out to enable)
    }

To enable Kraken or MEXC:
1. Uncomment lines in _apis dict
2. Re-run script

Supported Currencies (edit get_currencies() function):
    bases = ["BTC", "ETH", "DOGE", "SOL", "XRP", "LINK"]
    quote = "USD"

TROUBLESHOOTING:

"No data returned for BTC/USD"
    - Exchange may not support the pair
    - Time range may be outside available data
    - Check if USD vs USDT mapping is needed (see individual exchange docs)

"HTTP error 429" (Rate Limited)
    - One exchange hit rate limit
    - Increase sleep times in individual API files
    - Or: fetch smaller date ranges separately

Missing data for certain timestamps
    - Normal - some exchanges have data gaps
    - Check that at least 1 exchange has data for critical dates
    - Use Gate.io (archive) for more historical coverage

"Pair not supported or API unreachable"
    - Exchange API may be down
    - Currency pair truly not supported
    - Check internet connectivity

NEXT STEPS:

After collecting data:
1. Run data_analysis module to engineer 100+ features
2. Feed featured data to ML models (LSTM, GRU, Transformer, XGBoost, etc.)
3. Evaluate model predictions on arbitrage opportunities

See ../data_analysis/README.md for feature engineering details.
"""

import pandas as pd
import importlib.util
from datetime import datetime
from pathlib import Path

data_retrieve_dir = Path(__file__).resolve().parent
_apis = {
    "coinbase": data_retrieve_dir / "coinbase_api.py",
    "binance": data_retrieve_dir / "binance_api.py",
    "bitfinex": data_retrieve_dir / "bitfinex_api.py",
    "gateio": data_retrieve_dir / "gateio_api.py",
    # "kraken": data_retrieve_dir / "kraken_api.py",
    # "mexc": data_retrieve_dir / "mexc_api.py",
}

def get_currencies():
    print("Available currencies: BTC, ETH, DOGE, SOL, XRP, LINK")
    print("Enter comma-separated list (e.g., BTC,ETH,DOGE):")
    currency_input = input("Currencies: ").strip().upper()
    if currency_input == "":
        bases = ["BTC", "ETH", "DOGE", "SOL", "XRP", "LINK"]
    else:
        bases = [c.strip() for c in currency_input.split(",") if c.strip()]
    quote = "USD"
    return bases, quote

def get_timerange():
    print("\n--- Time Range (UTC) ---")
    print("Format: YYYY-MM-DD HH:MM")
    start_date = input("Enter start date (UTC): ").strip()
    if start_date == "":
        start_date = "2025-03-01 10:00"
    # else:
    #     start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M")

    end_date = input("Enter end date (UTC): ").strip()
    if end_date == "":
        end_date = "2025-05-01 10:00"
    # else:
    #     end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M")

    return start_date, end_date

def load_module(module_name, file_path):
    """Dynamically load a Python module"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def fetch_data_from_modules(base, quote, start_date, end_date):
    currency = f"{base}/{quote}"
    print(f"\n=== Fetching data for {currency} ===")
    all_exchange_data = {exchange: [] for exchange in _apis.keys()}

    for exchange_name, api_file in _apis.items():
        print(f"\n--- Fetching from {exchange_name.upper()} ---")
        try:
            module = load_module(exchange_name, api_file)
            print(f"  Calling {api_file} with {currency} from {start_date} to {end_date} (UTC)")
            df = module.fetch_data(currency, start_date, end_date)
            all_exchange_data[exchange_name].append(df)

            if all_exchange_data[exchange_name]:
                combined = pd.concat(all_exchange_data[exchange_name], ignore_index=True)
                combined = combined.drop_duplicates(subset=['time']).sort_values('time')
                all_exchange_data[exchange_name] = combined
                print(f"  Total {exchange_name}: {len(combined)} entries")
            else:
                all_exchange_data[exchange_name] = None
        except Exception as e:
            print(f"Error with {exchange_name}: {str(e)}")
            all_exchange_data[exchange_name] = None
    return all_exchange_data

def merge_dataframes(all_exchange_data):
    dataframes = {}
    for exchange_name, df in all_exchange_data.items():
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            df_renamed = df.copy()
            for col in df_renamed.columns:
                if col != 'time':
                    df_renamed.rename(columns={col: f"{exchange_name.upper()}:{col}"}, inplace=True)
            dataframes[exchange_name] = df_renamed
        else:
            # NEW: Print warning when exchange returns no data
            if df is not None and df.empty:
                print(f"  ⚠️  WARNING: {exchange_name.upper()} returned no data for this time range")

    if len(dataframes) > 0:
        print("\n--- Combining data from all exchanges ---")
        combined_df = list(dataframes.values())[0]
        for df in list(dataframes.values())[1:]:
            combined_df = pd.merge(combined_df, df, on='time', how='outer')
        combined_df = combined_df.sort_values('time')
        return combined_df
    else:
        return None

def save_to_csv(df, base, quote):
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    currency = f"{base}{quote}"
    filename = data_dir / f"combined_{currency}_data.csv"

    df.to_csv(filename, index=False)
    print(f"✓ Combined data saved to: {filename}")

def main():
    print("=== Cryptocurrency Data Retrieval ===")
    print("Note: All times should be in UTC\n")

    # Get currencies from user
    bases, quote = get_currencies()
    
    # Get time range from user
    start_date, end_date = get_timerange()
    
    # print(f"Currency: {base}/{quote} ; Start Date: {start_date} ; End Date: {end_date}")
    confirm = input("\nProceed with data retrieval? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    try:
        for base in bases:
            all_exchange_data = fetch_data_from_modules(base, quote, start_date, end_date)

            # Merge all exchanges for this currency
            combined_df = merge_dataframes(all_exchange_data)

            if combined_df is not None:
                # Save combined data to CSV
                save_to_csv(combined_df, base, quote)
                print(f"\n=== Data retrieval complete for {base}/{quote} ===")
                print(f"Total rows: {len(combined_df)}")
                print(f"Time range: {combined_df['time'].min()} to {combined_df['time'].max()} UTC")
                print(f"Columns: {len(combined_df.columns)}")
            else:
                print(f"\nNo data was retrieved from any exchange for {base}/{quote}.")

    except ValueError as e:
        print(f"[DEBUG] ValueError: {e}")
        print(f"[DEBUG] start_date: {start_date}, end_date: {end_date}")
        # If you want to see DataFrame columns:
        # print([df.columns for df in all_exchange_data.values() if df is not None])
        return

if __name__ == "__main__":
    main()