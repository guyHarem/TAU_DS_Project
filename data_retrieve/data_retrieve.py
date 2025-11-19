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
    
    # Get currency pair from user
    print("Available currency pairs:")
    print("1. BTC/USD")
    print("2. ETH/USD")
    print("3. Custom pair")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        currency = "BTC/USD"
    elif choice == "2":
        currency = "ETH/USD"
    elif choice == "3":
        currency = input("Enter currency pair (e.g., BTC/USD): ").strip()
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Get time range from user
    print("\n--- Time Range (UTC) ---")
    print("Format: YYYY-MM-DD HH:MM")
    start_date = input("Enter start date (UTC): ").strip()
    end_date = input("Enter end date (UTC): ").strip()
    
    try:
        # Validate dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
        
        # Calculate total duration
        total_minutes = (end_dt - start_dt).total_seconds() / 60
        
        print(f"\nTotal time range: {total_minutes:.0f} minutes ({total_minutes/60:.1f} hours)")
        
        # Split into 300-minute chunks
        chunks = split_time_range(start_date, end_date, chunk_minutes=300)
        print(f"This will be split into {len(chunks)} request(s) of max 300 minutes each")
        
        confirm = input("\nProceed with data retrieval? (y/n): ").strip().lower()
        if confirm != 'y':
            return
        
        # Load and run each API module
        apis = {
            "coinbase": "coinbase_api.py",
            "binance": "binance_api.py",
            "bitfinex": "bitfinex_api.py"
        }
        
        all_exchange_data = {exchange: [] for exchange in apis.keys()}
        
        for exchange_name, api_file in apis.items():
            print(f"\n--- Fetching from {exchange_name.upper()} ---")
            
            try:
                # Load the module
                module = load_module(exchange_name, api_file)
                
                # Fetch data for each chunk
                for i, chunk in enumerate(chunks, 1):
                    print(f"  Chunk {i}/{len(chunks)}: {chunk['start']} to {chunk['end']} UTC")
                    
                    try:
                        df = module.fetch_data(currency, chunk['start'], chunk['end'])
                        all_exchange_data[exchange_name].append(df)
                    except Exception as e:
                        print(f"  Error in chunk {i}: {str(e)}")
                
                # Combine all chunks for this exchange
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
        
        # Merge all exchanges
        dataframes = {}
        for exchange_name, df in all_exchange_data.items():
            if df is not None and not df.empty:
                # Rename columns with exchange prefix
                df_renamed = df.copy()
                for col in df_renamed.columns:
                    if col != 'time':
                        df_renamed.rename(columns={col: f"{exchange_name.upper()}:{col}"}, inplace=True)
                dataframes[exchange_name] = df_renamed
        
        if dataframes:
            print("\n--- Combining data from all exchanges ---")
            
            # Start with the first dataframe
            combined_df = list(dataframes.values())[0]
            
            # Merge with remaining dataframes on 'time' column
            for df in list(dataframes.values())[1:]:
                combined_df = pd.merge(combined_df, df, on='time', how='outer')
            
            # Sort by time
            combined_df = combined_df.sort_values('time')
            
            # Generate filename
            base, quote = currency.split('/')
            filename = f"combined_{base}{quote}_data.csv"
            
            # Save to CSV
            combined_df.to_csv(filename, index=False)
            
            print(f"\n=== Data retrieval complete ===")
            print(f"Combined data saved to {filename}")
            print(f"Total rows: {len(combined_df)}")
            print(f"Time range: {combined_df['time'].min()} to {combined_df['time'].max()} UTC")
            print(f"Columns: {len(combined_df.columns)}")
        else:
            print("\nNo data was retrieved from any exchange.")
        
    except ValueError as e:
        print(f"Invalid date format. Please use YYYY-MM-DD HH:MM")
        return

if __name__ == "__main__":
    main()