import requests
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

_supported_pairs_cache = None

def is_supported_pair(base, quote):
    global _supported_pairs_cache
    if _supported_pairs_cache is None:
        url = "https://api-pub.bitfinex.com/v2/conf/pub:list:pair:exchange"
        resp = requests.get(url)
        if resp.status_code != 200:
            print("Bitfinex: Could not fetch supported pairs")
            return False
        _supported_pairs_cache = set(p.upper() for p in resp.json()[0])

    pair1 = f"{base.upper()}{quote.upper()}"
    pair2 = f"{base.upper()}:{quote.upper()}"
    pair1_reversed = f"{quote.upper()}{base.upper()}"
    pair2_reversed = f"{quote.upper()}:{base.upper()}"
    is_reversed = False

    if pair1 in _supported_pairs_cache:
        pair, is_reversed = pair1, False
    elif pair2 in _supported_pairs_cache:
        pair, is_reversed = pair2, False
    elif pair1_reversed in _supported_pairs_cache:
        pair, is_reversed = pair1_reversed, True
    elif pair2_reversed in _supported_pairs_cache:
        pair, is_reversed = pair2_reversed, True
    else:
        return None, False
    
    return pair, is_reversed

def split_time_range(start_date, end_date, chunk_minutes=10000):
    """Split time range into chunks of specified minutes"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)

    chunks = []
    current_start = start_dt
    while current_start < end_dt:
        current_end = min(current_start + timedelta(minutes=chunk_minutes), end_dt)
        chunks.append({
            'start': int(current_start.timestamp() * 1000),
            'end': int(current_end.timestamp() * 1000)
        })
        current_start = current_end
    return chunks

def fetch_data(currency, start_date, end_date):
    base, quote = currency.split('/')
    pair, is_reversed = is_supported_pair(base, quote)
    if not pair:
        print(f"Bitfinex: {currency} is not supported in either direction")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    symbol = f"t{pair}"
    chunks = split_time_range(start_date, end_date)
    url = f"https://api-pub.bitfinex.com/v2/candles/trade:1m:{symbol}/hist"

    chunks_dfs = []
    for i, chunk in enumerate(chunks, 1):
        params = {
        'start': chunk['start'],
        'end': chunk['end'],
        'limit': 10000,
        'sort': 1,
        }

        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"Bitfinex: API error (status {resp.status_code}) for {currency} in chunk {i}")
            continue

        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            continue

        df_chunk = pd.DataFrame(
            data,
            columns=['time', 'open', 'close', 'high', 'low', 'volume']
        )
        chunks_dfs.append(df_chunk)

    if not chunks_dfs:
        print(f"Bitfinex: No data for {currency}")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.concat(chunks_dfs, ignore_index=True)
    
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
    df['time'] = df['time'].dt.floor('min')
    df = df.drop_duplicates(subset=["time"]).sort_values("time")
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    if is_reversed:
        df[["open", "high", "low", "close"]] = 1 / df[["open", "high", "low", "close"]].astype(float)
        df["volume"] = df["volume"].astype(float) * df["close"].astype(float)

    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

    print(f"Bitfinex: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    return df

def save_to_csv(df, args):
    project_root = Path(__file__).resolve().parent.parent
    archive_dir = project_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    safe_start = args.start.replace(" ", "_").replace(":", "-")
    currency = args.currency.replace('/', '')
    filename = archive_dir / f"bitfinex_{currency}_{safe_start}.csv"

    df.to_csv(filename, index=False)
    print(f"✓ Data saved to: {filename}")

def arguments_parser():
    parser = argparse.ArgumentParser(description="Fetch historical Bitfinex data")
    parser.add_argument("currency", type=str, help="Currency pair, e.g., BTC/USD")
    parser.add_argument("start", type=str, help="Start datetime (YYYY-MM-DD HH:MM)")
    parser.add_argument("end", type=str, help="End datetime (YYYY-MM-DD HH:MM)")
    parser.add_argument("--save_to_csv", action="store_true", help="Save output to CSV file")
    return parser

if __name__ == "__main__":
    args = arguments_parser().parse_args()
    df = fetch_data(args.currency, args.start, args.end)

    if args.save_to_csv and not df.empty:
        save_to_csv(df, args)