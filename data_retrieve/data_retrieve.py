import pandas as pd
import importlib.util
from datetime import datetime
from pathlib import Path


_apis = {
    "coinbase": "coinbase_api.py",
    "binance": "binance_api.py",
    "bitfinex": "bitfinex_api.py",
    "mexc": "mexc_api.py",
    "gateio": "gateio_api.py",
    "kraken": "kraken_api.py" 
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
    else:
        start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M")

    end_date = input("Enter end date (UTC): ").strip()
    if end_date == "":
        end_date = "2025-05-01 10:00"
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M")

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

    if dataframes:
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
    print("Available currencies: BTC, ETH, DOGE, SOL, XRP, LINK")
    print("Enter comma-separated list (e.g., BTC,ETH,DOGE):")
    bases, quote = get_currencies()
    
    # Get time range from user
    start_date, end_date = get_timerange()
    
    print(f"Currency: {base}/{quote} ; Start Date: {start_date} ; End Date: {end_date}")
    confirm = input("\nProceed with data retrieval? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    try:
        for base in bases:
            all_exchange_data = fetch_data_from_modules(base, quote, start_date, end_date)

            # Merge all exchanges for this currency
            combined_df = merge_dataframes(all_exchange_data)

            if combined_df:
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