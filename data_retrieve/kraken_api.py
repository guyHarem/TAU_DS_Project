"""
KRAKEN API Module
=================

Fetches 1-minute OHLCV candlestick data from the Kraken cryptocurrency exchange.

EXCHANGE DETAILS:
- REST Endpoint: https://api.kraken.com/0/public/OHLC
- Pair Format: Internal altname (e.g., "XBTUSDT" for BTC/USDT)
- Asset Mapping: BTC→XBT, DOGE→XDG, USD→USD (custom naming)
- Rate Limiting: Custom counter-based system (budget: 15.0, decay: 0.33/sec)
- Pair Lookup: Fetches altname mapping from /AssetPairs endpoint
- Retry Logic: Up to 6 retries with exponential backoff
- Pair Reversal: Supported (via altname mapping)

DATA QUALITY:
- Timestamps: Unix epoch (seconds), converted to UTC
- Duplicates: Automatically removed
- Time Floors: Rounded to nearest minute
- Asset Mapping: Transparent conversion (BTC -> XBT internally)

IMPLEMENTATION NOTES:
1. Asset name mapping (Kraken uses non-standard names):
   - BTC → XBT (Kraken's internal code for Bitcoin)
   - DOGE → XDG (DOGE's internal code)
   - USD → USD (unchanged)
   Example: "BTC/USD" internally becomes "XBTUSDT"

2. Fetches supported pair altnames from /AssetPairs endpoint on first call
   - Caches result globally to avoid repeated lookups
   - Altname maps standard names to API parameter values
   - Example: "XBTUSDT" is the altname for BTC/USDT pair

3. Custom rate limiting with counter-based budget system:
   - Counter Budget: 15.0 (maximum calls before throttle)
   - Decay Rate: 0.33 per second (budget increases over time)
   - Throttle: If counter > 0, sleep until budget available
   - Prevents hitting Kraken's 15-call-per-second limit

4. Automatic retry on failures:
   - Max 6 retries with exponential backoff
   - Handles temporary API errors gracefully

5. If original pair fails, automatically tries reversed pair
   Example: If XBTUSDT not found, tries USDTXBT

6. De-duplicates by timestamp before returning DataFrame

USAGE:
    from kraken_api import fetch_data
    df = fetch_data("BTC/USD", "2025-03-01 10:00", "2025-03-02 10:00")
    # Returns DataFrame with columns: [time, open, high, low, close, volume]

ERROR HANDLING:
- Returns empty DataFrame if pair altname not found
- Returns empty DataFrame on network/API errors after all retries
- Logs warnings for retry attempts but continues
- Does NOT raise exceptions - logs to console instead

PERFORMANCE NOTES:
- Rate limiting: Can be slower due to conservative throttling
- Typical: 0.5-2 seconds per request (depends on budget availability)
- Good for: Small time ranges, when using alongside other exchanges
"""

import requests
import pandas as pd
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
# region Variables
_altname_map_cache = None

KRAKEN_RATE_LIMIT_MAX_COUNTER = 15.0
KRAKEN_RATE_LIMIT_DECAY_PER_SECOND = 0.33
KRAKEN_RATE_LIMIT_BUFFER = 1.0
KRAKEN_MAX_RETRIES = 6

_asset_map = {
    # Kraken uses different symbols for some assets
        "BTC": "XBT",
        "ETH": "ETH",
        "DOGE": "XDG",
        "USD": "USD",
        "USDT": "USD",
        "EUR": "EUR",
        "SOL": "SOL",
        "XRP": "XRP",
        "LINK": "LINK",
    }
# endregion Variables

def get_supported_pair(base, quote):
    global _altname_map_cache
    base = _asset_map.get(base.upper(), base.upper())
    quote = _asset_map.get(quote.upper(), quote.upper())

    if _altname_map_cache is None:
        url = "https://api.kraken.com/0/public/AssetPairs"
        resp = requests.get(url)
        if resp.status_code != 200:
            print("Kraken: Could not fetch supported pairs")
            return None, False
        data = resp.json()
        if 'result' not in data:
            print("Kraken: Could not fetch supported pairs")
            return None, False
        _altname_map_cache = {details['altname'].upper(): pair for pair, details in data['result'].items()}

    pair_name = f"{base}{quote}".upper()
    reversed_pair_name = f"{quote}{base}".upper()

    if pair_name in _altname_map_cache:
        return _altname_map_cache[pair_name], False
    if reversed_pair_name in _altname_map_cache:
        return _altname_map_cache[reversed_pair_name], True
    return None, False

def parse_utc_datetime(value):
    """Parse YYYY-MM-DD HH:MM into timezone-aware UTC datetime."""
    return datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)

def throttle_request(counter, last_counter_update):
    """Apply Kraken-style counter decay and sleep if the next call would exceed the budget."""
    now = time.monotonic()
    elapsed = now - last_counter_update
    counter = max(0.0, counter - elapsed * KRAKEN_RATE_LIMIT_DECAY_PER_SECOND)

    projected_counter = counter + 1.0
    if projected_counter > KRAKEN_RATE_LIMIT_MAX_COUNTER:
        excess = projected_counter - KRAKEN_RATE_LIMIT_MAX_COUNTER
        sleep_seconds = excess / KRAKEN_RATE_LIMIT_DECAY_PER_SECOND + KRAKEN_RATE_LIMIT_BUFFER
        time.sleep(sleep_seconds)
        now = time.monotonic()
        elapsed = now - last_counter_update
        counter = max(0.0, counter - elapsed * KRAKEN_RATE_LIMIT_DECAY_PER_SECOND)

    counter += 1.0
    return counter, now

def parse_throttle_retry_after(errors):
    """Parse Kraken throttled timestamp and convert it to seconds to wait."""
    for error in errors:
        prefix = 'EService: Throttled:'
        if error.startswith(prefix):
            raw_timestamp = error.split(':', maxsplit=2)[-1].strip()
            try:
                retry_at = float(raw_timestamp)
            except ValueError:
                continue
            return max(1.0, retry_at - time.time())
    return None

def extract_trades_payload(result, pair):
    """Extract trade rows and pagination cursor for the exact requested pair only."""
    return result.get(pair), result.get('last')

def aggregate_trades_to_ohlcv(trades, start_dt, end_dt):
    """Aggregate trade rows into 1-minute OHLCV candles."""
    if not trades:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    records = []
    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()

    for trade in trades:
        if len(trade) < 3:
            continue
        try:
            price = float(trade[0])
            volume = float(trade[1])
            trade_time = float(trade[2])
        except (TypeError, ValueError):
            continue

        if start_ts <= trade_time < end_ts:
            records.append((trade_time, price, volume))

    if not records:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    trades_df = pd.DataFrame(records, columns=['trade_time', 'price', 'volume'])
    trades_df = trades_df.sort_values('trade_time')
    trades_df['minute'] = pd.to_datetime(trades_df['trade_time'], unit='s', utc=True).dt.floor('min')

    candles = (
        trades_df.groupby('minute', as_index=False)
        .agg(
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('volume', 'sum')
        )
    )

    candles = candles.rename(columns={'minute': 'time'})
    candles['time'] = candles['time'].dt.tz_localize(None)
    candles = candles.sort_values('time').drop_duplicates(subset=['time'])
    candles['time'] = candles['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return candles[['time', 'open', 'high', 'low', 'close', 'volume']]

def fetch_data(currency, start_date, end_date):
    """
    Fetch historical 1-minute kline data from Kraken for a given currency pair and time range.
    Args:
        currency (str): e.g. "BTC/USD"
        start_date (str): "YYYY-MM-DD HH:MM" (UTC)
        end_date (str): "YYYY-MM-DD HH:MM" (UTC)
    Returns:
        pd.DataFrame: columns = ["time", "open", "high", "low", "close", "volume"]
    """
    # Parse currency pair and resolve Kraken pair once.
    base, quote = currency.split('/')
    pair, is_reversed = get_supported_pair(base, quote)
    if not pair:
        print(f"Kraken: Currency pair {currency} not found.")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    start_dt = parse_utc_datetime(start_date)
    end_dt = parse_utc_datetime(end_date)
    if start_dt >= end_dt:
        print(f"Kraken: Invalid range for {currency}. start must be before end.")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    url = "https://api.kraken.com/0/public/Trades"
    since_cursor = int(start_dt.timestamp())
    all_trades = []
    page = 0
    counter = 0.0
    last_counter_update = time.monotonic()
    stop_fetching = False
    while True:
        page += 1
        params = {
            'pair': pair,
            'since': since_cursor,
            'count': 1000,
        }

        retry_count = 0
        while True:
            counter, last_counter_update = throttle_request(counter, last_counter_update)
            try:
                resp = requests.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                retry_count += 1
                if retry_count > KRAKEN_MAX_RETRIES:
                    print(f"Kraken: Request error for {currency} on page {page}: {exc}")
                    stop_fetching = True
                    break
                sleep_seconds = min(2 ** retry_count, 30)
                print(f"Kraken: Request error for {currency} on page {page}; retrying in {sleep_seconds:.1f}s")
                time.sleep(sleep_seconds)
                continue

            if resp.status_code == 429:
                retry_count += 1
                if retry_count > KRAKEN_MAX_RETRIES:
                    print(f"Kraken: API error (status 429) for {currency} on page {page}")
                    stop_fetching = True
                    break
                sleep_seconds = min(2 ** retry_count + KRAKEN_RATE_LIMIT_BUFFER, 60)
                print(f"Kraken: HTTP 429 for {currency} on page {page}; retrying in {sleep_seconds:.1f}s")
                time.sleep(sleep_seconds)
                continue

            if resp.status_code != 200:
                print(f"Kraken: API error (status {resp.status_code}) for {currency} on page {page}")
                break

            data = resp.json()
            if not isinstance(data, dict):
                print(f"Kraken: Unexpected response format for {currency} on page {page}")
                break

            errors = data.get('error', [])
            if errors:
                retry_after = parse_throttle_retry_after(errors)
                is_rate_limited = any(error == 'EAPI:Rate limit exceeded' for error in errors)
                if retry_after is not None or is_rate_limited:
                    retry_count += 1
                    if retry_count > KRAKEN_MAX_RETRIES:
                        print(f"Kraken: API returned errors for {currency} on page {page}: {errors}")
                        stop_fetching = True
                        break
                    if retry_after is None:
                        retry_after = min(2 ** retry_count + KRAKEN_RATE_LIMIT_BUFFER, 60)
                    else:
                        retry_after = min(max(retry_after, 1.0), 120.0)
                    print(f"Kraken: Rate limited for {currency} on page {page}; retrying in {retry_after:.1f}s")
                    time.sleep(retry_after)
                    continue

                print(f"Kraken: API returned errors for {currency} on page {page}: {errors}")
                break

            break

        if stop_fetching:
            break

        if resp.status_code != 200:
            break

        if not isinstance(data, dict):
            break

        if data.get('error', []):
            break

        result = data.get('result', {})
        if not isinstance(result, dict):
            break

        trades, next_cursor = extract_trades_payload(result, pair)
        if trades is None:
            print(f"Kraken: Response missing expected pair key '{pair}' for {currency} on page {page}")
            break

        if not trades:
            break

        all_trades.extend(trades)

        trade_times = []
        for trade in trades:
            if len(trade) < 3:
                continue
            try:
                trade_times.append(float(trade[2]))
            except (TypeError, ValueError):
                continue

        if not trade_times:
            break

        latest_trade_time = max(trade_times)
        if latest_trade_time >= end_dt.timestamp():
            break

        try:
            next_cursor = int(next_cursor)
        except (TypeError, ValueError):
            break

        if next_cursor <= since_cursor:
            break

        since_cursor = next_cursor

    if not all_trades:
        print(f"Kraken: No data for {currency}")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    df = aggregate_trades_to_ohlcv(all_trades, start_dt, end_dt)
    if df.empty:
        print(f"Kraken: No trades in requested time window for {currency}")
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

    if is_reversed:
        open_vals = df['open'].astype(float)
        high_vals = df['high'].astype(float)
        low_vals = df['low'].astype(float)
        close_vals = df['close'].astype(float)

        df['open'] = 1 / open_vals
        df['high'] = 1 / low_vals
        df['low'] = 1 / high_vals
        df['close'] = 1 / close_vals
        df["volume"] = df["volume"].astype(float) * df["close"].astype(float)
    else:
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

    print(f"Kraken: Retrieved {len(df)} entries from {df['time'].min()} to {df['time'].max()} UTC")
    return df
    
def save_to_csv(df, args):
    project_root = Path(__file__).resolve().parent.parent
    archive_dir = project_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    safe_start = args.start.replace(" ", "_").replace(":", "-")
    currency = args.currency.replace('/', '')
    filename = archive_dir / f"kraken_{currency}_{safe_start}.csv"

    df.to_csv(filename, index=False)
    print(f"✓ Data saved to: {filename}")

def arguments_parser():
    parser = argparse.ArgumentParser(description="Fetch historical Kraken data")
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