import subprocess
import sys
import pandas as pd
from datetime import datetime, timedelta
import importlib.util

def load_module(module_name, file_path):
    """Dynamically load a Python module"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def split_time_range(start_date, end_date, chunk_minutes=300):
    """Split time range into chunks of specified minutes"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
    chunks = []
    current_start = start_dt
    while current_start < end_dt:
        current_end = min(current_start + timedelta(minutes=chunk_minutes), end_dt)
        chunks.append({
            'start': current_start.strftime("%Y-%m-%d %H:%M"),
            'end': current_end.strftime("%Y-%m-%d %H:%M")
        })
        current_start = current_end
    return chunks

def main():
    print("=== Cryptocurrency Data Retrieval ===")
    print("Note: All times should be in UTC\n")
    
    # Get currencies from user
    print("Available currencies: BTC, ETH, DOGE")
    print("Enter comma-separated list (e.g., BTC,ETH,DOGE):")
    currency_input = input("Currencies: ").strip().upper()
    bases = [c.strip() for c in currency_input.split(",") if c.strip()]
    quote = "USD"
    
    # Get time range from user
    print("\n--- Time Range (UTC) ---")
    print("Format: YYYY-MM-DD HH:MM")
    start_date = input("Enter start date (UTC): ").strip()
    end_date = input("Enter end date (UTC): ").strip()
    
    try:
        # Validate dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
        total_minutes = (end_dt - start_dt).total_seconds() / 60
        print(f"\nTotal time range: {total_minutes:.0f} minutes ({total_minutes/60:.1f} hours)")
        chunks = split_time_range(start_date, end_date, chunk_minutes=300)
        print(f"This will be split into {len(chunks)} request(s) of max 300 minutes each")
        confirm = input("\nProceed with data retrieval? (y/n): ").strip().lower()
        if confirm != 'y':
            return

        apis = {
            "coinbase": "coinbase_api.py",
            "binance": "binance_api.py",
            "bitfinex": "bitfinex_api.py",
            "mexc": "mexc_api.py",
            "gateio": "gateio_api.py",
            "kraken": "kraken_api.py" 
        }

        for base in bases:
            currency = f"{base}/{quote}"
            print(f"\n=== Fetching data for {currency} ===")
            all_exchange_data = {exchange: [] for exchange in apis.keys()}

            for exchange_name, api_file in apis.items():
                print(f"\n--- Fetching from {exchange_name.upper()} ---")
                try:
                    module = load_module(exchange_name, api_file)
                    for i, chunk in enumerate(chunks, 1):
                        print(f"  Chunk {i}/{len(chunks)}: {chunk['start']} to {chunk['end']} UTC")
                        try:
                            df = module.fetch_data(currency, chunk['start'], chunk['end'])
                            all_exchange_data[exchange_name].append(df)
                        except Exception as e:
                            print(f"  Error in chunk {i}: {str(e)}")
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

            # Merge all exchanges for this currency
            dataframes = {}
            for exchange_name, df in all_exchange_data.items():
                if df is not None and not df.empty:
                    df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                    df_renamed = df.copy()
                    for col in df_renamed.columns:
                        if col != 'time':
                            df_renamed.rename(columns={col: f"{exchange_name.upper()}:{col}"}, inplace=True)
                    dataframes[exchange_name] = df_renamed

            if dataframes:
                print("\n--- Combining data from all exchanges ---")
                combined_df = list(dataframes.values())[0]
                for df in list(dataframes.values())[1:]:
                    combined_df = pd.merge(combined_df, df, on='time', how='outer')
                combined_df = combined_df.sort_values('time')
                filename = f"combined_{base}{quote}_data.csv"
                combined_df.to_csv(filename, index=False)
                print(f"\n=== Data retrieval complete for {currency} ===")
                print(f"Combined data saved to {filename}")
                print(f"Total rows: {len(combined_df)}")
                print(f"Time range: {combined_df['time'].min()} to {combined_df['time'].max()} UTC")
                print(f"Columns: {len(combined_df.columns)}")
            else:
                print(f"\nNo data was retrieved from any exchange for {currency}.")

    except ValueError as e:
        print(f"[DEBUG] ValueError: {e}")
        print(f"[DEBUG] start_date: {start_date}, end_date: {end_date}")
        # If you want to see DataFrame columns:
        # print([df.columns for df in all_exchange_data.values() if df is not None])
        return

if __name__ == "__main__":
    main()