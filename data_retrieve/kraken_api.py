import requests
import pandas as pd
from datetime import datetime, timezone
import argparse

def fetch_data(currency, start_date, end_date, interval=1):
    """
    Fetch historical 1-minute kline data from Kraken for a given currency pair and time range.
    Args:
        currency (str): e.g. "BTC/USD"
        start_date (str): "YYYY-MM-DD HH:MM" (UTC)
        end_date (str): "YYYY-MM-DD HH:MM" (UTC)
    Returns:
        pd.DataFrame: columns = ["time", "open", "high", "low", "close", "volume"]
    """
    # Parse currency pair and convert to Kraken format
    base, quote = currency.split('/')

    # Kraken uses different symbols for some assets
    asset_map = {
        "BTC": "XXBT",
        "ETH": "XETH",
        "DOGE": "XDG",
        "USD": "ZUSD",
        "EUR": "ZEUR",
        "USDT": "ZUSDT"
    }
    base_k = asset_map.get(base.upper(), base.upper())
    quote_k = asset_map.get(quote.upper(), quote.upper())
    kraken_pair = f"{base_k}{quote_k}"

    # Convert dates to epoch seconds (Kraken uses UTC)
    t_start = int(datetime.strptime(start_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp())
    t_end   = int(datetime.strptime(end_date, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp())

    print(f"Requesting from Kraken API:")
    print(f"  Pair: {kraken_pair}")
    print(f"  Start: {start_date} UTC")
    print(f"  End: {end_date} UTC")
    print(f"  Granularity: {interval} minute(s) (trade aggregation)")

    url = "https://api.kraken.com/0/public/Trades"
    since = t_start * 1_000_000_000  # nanoseconds

    all_trades = []

    # Fetch trades in a loop
    while True:
        resp = requests.get(url, params={"pair": kraken_pair, "since": since})
        print(f"  Response Status: {resp.status_code}")
        resp.raise_for_status()

        data = resp.json()["result"]
        trades = data.get(kraken_pair, [])

        if not trades:
            break

        for tr in trades:
            price, volume, t, *_ = tr
            ts = float(t)

            if ts < t_start:
                continue
            if ts > t_end:
                break

            all_trades.append((ts, float(price), float(volume)))

        # Update pagination cursor
        since = int(data["last"])

        # Stop if overshoot
        if trades and float(trades[-1][2]) > t_end:
            break

    print(f"  Trades collected inside time range: {len(all_trades)}")

    # If no trades → return an empty candle DataFrame
    if not all_trades:
        print("  Kraken returned no trades for this interval.")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    # Build DataFrame
    df = pd.DataFrame(all_trades, columns=["time", "price", "volume"])
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df = df.set_index("datetime")

    # Resample to 1-minute candles
    ohlc = df["price"].resample("1min").ohlc()
    ohlc["volume"] = df["volume"].resample("1min").sum()

    ohlc.reset_index(inplace=True)
    ohlc.rename(columns={"datetime": "time"}, inplace=True)

    print(f"Kraken: Generated {len(ohlc)} candles from {ohlc['time'].min()} to {ohlc['time'].max()} UTC")

    return ohlc[["time", "open", "high", "low", "close", "volume"]]

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"✓ Data saved to: {filename}")

def arguments_parser():
    parser = argparse.ArgumentParser(description="Fetch historical Kraken data")
    parser.add_argument("currency", type=str, help="Currency pair, e.g., BTC/USD")
    parser.add_argument("start", type=str, help="Start datetime (YYYY-MM-DD HH:MM)")
    parser.add_argument("end", type=str, help="End datetime (YYYY-MM-DD HH:MM)")
    parser.add_argument("--interval", type=int, default=1, help="Candle interval in minutes (default: 1)")
    parser.add_argument("--save_to_csv", action="store_true", help="Save output to CSV file")
    return parser

if __name__ == "__main__":
    args = arguments_parser().parse_args()
    df = fetch_data(args.currency, args.start, args.end, args.interval)

    print(f"Retrieved {len(df)} entries")

    if args.save_to_csv and not df.empty:
        filename = f"kraken_{args.currency.replace('/', '')}_{args.start.replace(' ', '_')}.csv"
        save_to_csv(df, filename)