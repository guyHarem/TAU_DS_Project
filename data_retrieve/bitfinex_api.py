import requests
import pandas as pd
import sys
from datetime import datetime, timezone

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
    if pair1 in _supported_pairs_cache:
        return pair1
    elif pair2 in _supported_pairs_cache:
        return pair2
    else:
        return None

def fetch_data(currency, start_date, end_date):
    base, quote = currency.split('/')
    supported_pair = is_supported_pair(base, quote)
    if not supported_pair:
        print(f"Bitfinex: {currency} is not a supported pair")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    symbol = f"t{supported_pair}"
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    url = f"https://api-pub.bitfinex.com/v2/candles/trade:1m:{symbol}/hist"
    params = {
        'start': start_ms,
        'end': end_ms,
        'limit': 10000
    }

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"Bitfinex: API error (status {resp.status_code}) for {currency}")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        print(f"Bitfinex: No data for {currency}")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(
        data,
        columns=['time', 'open', 'close', 'high', 'low', 'volume']
    )
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
    df['time'] = df['time'].dt.floor('min')
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.sort_values('time')
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

    print(f"Bitfinex: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    return df

if __name__ == "__main__":
    # Allow running with arguments: currency, start_date, end_date
    if len(sys.argv) == 4:
        df = fetch_data(sys.argv[1], sys.argv[2], sys.argv[3])
        print(df.head())
    else:
        # Default: DOGE/USD for a recent interval
        df = fetch_data("DOGE/USD", "2025-11-28 10:00", "2025-11-28 16:00")
        print(df.head())