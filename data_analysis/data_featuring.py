import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

#region Hyper-parameters
TRADING_COST_PCT = 0.2
SAFETY_MARGIN_PCT = 0.1
REAL_OPPORTUNITY_THRESHOLD = TRADING_COST_PCT + SAFETY_MARGIN_PCT

ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw_data"
FEATURED_DATA_PATH = DATA_PATH / "featured_data"
#endregion

#region Prepare data
btcusd_data = pd.read_csv(f'{RAW_DATA_PATH}/combined_BTCUSD_data.csv')
ethusd_data = pd.read_csv(f'{RAW_DATA_PATH}/combined_ETHUSD_data.csv')
dogeusd_data = pd.read_csv(f'{RAW_DATA_PATH}/combined_DOGEUSD_data.csv')
linkusd_data = pd.read_csv(f'{RAW_DATA_PATH}/combined_LINKUSD_data.csv')
solusd_data = pd.read_csv(f'{RAW_DATA_PATH}/combined_SOLUSD_data.csv')
xrpusd_data = pd.read_csv(f'{RAW_DATA_PATH}/combined_XRPUSD_data.csv')

data_frames = [btcusd_data, ethusd_data, dogeusd_data, linkusd_data, solusd_data, xrpusd_data]
exchanges = ["BINANCE","BITFINEX","COINBASE","GATEIO","KRAKEN"]
original_features = ["high","low","open","close","volume"] # <- unused, delete
windows = [5, 15, 30]


# Change time string from CSV to time type
for df in data_frames:
    df['time'] = pd.to_datetime(df['time'])

#endregion

#region LAYER 2 FEATURES
def add_L2_minmax_features(df):
    # Close
    close_cols = [f"{ex}:close" for ex in exchanges if f"{ex}:close" in df.columns]
    df['min_close'] = df[close_cols].min(axis=1, skipna=True)
    df['max_close'] = df[close_cols].max(axis=1, skipna=True)
    df['buy_exchange'] = df[close_cols].idxmin(axis=1, skipna=True).str.split(':').str[0]
    df['sell_exchange'] = df[close_cols].idxmax(axis=1, skipna=True).str.split(':').str[0]

    # High/Low
    high_cols = [f"{ex}:high" for ex in exchanges if f"{ex}:high" in df.columns]
    low_cols = [f"{ex}:low" for ex in exchanges if f"{ex}:low" in df.columns]
    df['max_high'] = df[high_cols].max(axis=1, skipna=True)
    df['min_low'] = df[low_cols].min(axis=1,skipna= True)
    df['high_exchange'] = df[high_cols].idxmax(axis=1, skipna=True).str.split(':').str[0]
    df['low_exchange'] = df[low_cols].idxmin(axis=1, skipna=True).str.split(':').str[0]

def add_L2_exchange_features(df):
    close_cols = [f"{ex}:close" for ex in exchanges if f"{ex}:close" in df.columns]
    df['num_exchanges_available'] = df[close_cols].notna().sum(axis=1)

    for exchange in exchanges:
        if f'{exchange}:high' in df.columns and f'{exchange}:low' in df.columns and f'{exchange}:close' in df.columns:
            df[f'{exchange}_volatility'] = (df[f'{exchange}:high'] - df[f'{exchange}:low']) / df[f'{exchange}:close'] * 100
    
        if f'{exchange}:close' in df.columns and f'{exchange}:open' in df.columns:
            df[f'{exchange}_price_change'] = (df[f'{exchange}:close'] - df[f'{exchange}:open']) / df[f'{exchange}:open'] * 100
    
def add_L2_time_features(df):
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df[f'overlap_hours'] = df[f'hour'].apply(lambda x: 1 if 19 <= x <= 21 else 0)  # Example: active trading hours

def add_L2_cross_exchange_price_ratio(df):
    # For each pair of exchanges, calculate price ratios
    for i, ex1 in enumerate(exchanges):
        for ex2 in exchanges[i+1:]:  # Avoid duplicate pairs
            close1 = f"{ex1}:close"
            close2 = f"{ex2}:close"
            
            if close1 in df.columns and close2 in df.columns:
                df[f'price_ratio_{ex1}_{ex2}'] = df[close2] / df[close1]
    
    # Average price ratio across all exchange pairs
    ratio_cols = [col for col in df.columns if col.startswith('price_ratio_') and col != 'price_ratio_buy_sell']
    if ratio_cols:
        df['avg_price_ratio'] = df[ratio_cols].mean(axis=1)
        df['max_price_ratio'] = df[ratio_cols].max(axis=1)
        df['min_price_ratio'] = df[ratio_cols].min(axis=1)
        df['price_ratio_std'] = df[ratio_cols].std(axis=1)
        df.drop(columns=[col for col in ratio_cols], inplace=True)

#endregion

#region LAYER 3 FEATURES
def add_L3_spreads(df):
    df['spread_close_absolute'] = df['max_close'] - df['min_close']
    df['spread_close_pct'] = (df['spread_close_absolute'] / df['min_close']) * 100

    df['spread_highlow_absolute'] = df['max_high'] - df['min_low']
    df['spread_highlow_pct'] = (df['spread_highlow_absolute'] / df['min_low']) * 100

def add_L3_buy_sell_exchange_features(df):
    # Need .apply() because each row uses a DIFFERENT column
    df['volume_buy_exchange'] = df.apply(
        lambda row: row[f"{row['buy_exchange']}:volume"], 
        axis=1
    )
    df['volume_sell_exchange'] = df.apply(
        lambda row: row[f"{row['sell_exchange']}:volume"], 
        axis=1
    )

def add_L3_price_change_features(df): 
    df['price_change_buy_exchange'] = df.apply(
        lambda row: row[f"{row['buy_exchange']}_price_change"] if f"{row['buy_exchange']}_price_change" in df.columns else np.nan,
        axis=1
    )
    df['price_change_sell_exchange'] = df.apply(
        lambda row: row[f"{row['sell_exchange']}_price_change"] if f"{row['sell_exchange']}_price_change" in df.columns else np.nan,
        axis=1
    )

def add_L3_buy_sell_exchange_price_ratio(df): 
    df['price_ratio_buy_sell'] = df.apply(
        lambda row: row[f"{row['sell_exchange']}:close"] / row[f"{row['buy_exchange']}:close"],
        axis=1
    )

def add_L3_rolling_stats(df): 
    for window in windows:
        # Volume rolling statistics - HARD ROLLING on time axis
        for idx in range(len(df)):
            buy_ex = df.iloc[idx]['buy_exchange']
            sell_ex = df.iloc[idx]['sell_exchange']
            
            # Hard rolling window: look back exactly 'window' rows in time on the SPECIFIC exchange's volume
            buy_vol_col = f'{buy_ex}:volume'
            if buy_vol_col in df.columns:
                # Get exactly last 'window' rows of current buy exchange's volume
                window_data = df.iloc[max(0, idx-window+1):idx+1][buy_vol_col]
                df.loc[df.index[idx], f'volume_buy_rolling_std_{window}'] = window_data.std(ddof=0)
            
            sell_vol_col = f'{sell_ex}:volume'
            if sell_vol_col in df.columns:
                # Get exactly last 'window' rows of current sell exchange's volume
                window_data = df.iloc[max(0, idx-window+1):idx+1][sell_vol_col]
                df.loc[df.index[idx], f'volume_sell_rolling_std_{window}'] = window_data.std(ddof=0)

def add_L3_volatility_features(df): 
    volatility_cols = [f'{exchange}_volatility' for exchange in exchanges if f'{exchange}_volatility' in df.columns]
    
    if volatility_cols:
        df[f'volatility_avg'] = df[volatility_cols].mean(axis=1)
        df[f'volatility_max'] = df[volatility_cols].max(axis=1)
        df[f'volatility_min'] = df[volatility_cols].min(axis=1)
    
    df['price_position_buy_exchange'] = df.apply(
        lambda row: (
            (row[f"{row['buy_exchange']}:close"] - row[f"{row['buy_exchange']}:low"]) /
            (row[f"{row['buy_exchange']}:high"] - row[f"{row['buy_exchange']}:low"])
        ) if (f"{row['buy_exchange']}:close" in df.columns and 
              f"{row['buy_exchange']}:low" in df.columns and 
              f"{row['buy_exchange']}:high" in df.columns and
              not np.isclose(row[f"{row['buy_exchange']}:high"], row[f"{row['buy_exchange']}:low"])) else np.nan,
        axis=1
    )
    df['price_position_buy_exchange'] = df['price_position_buy_exchange'].replace([np.inf, -np.inf], np.nan)

    df['price_position_sell_exchange'] = df.apply(
        lambda row: (
            (row[f"{row['sell_exchange']}:close"] - row[f"{row['sell_exchange']}:low"]) /
            (row[f"{row['sell_exchange']}:high"] - row[f"{row['sell_exchange']}:low"])
        ) if (f"{row['sell_exchange']}:close" in df.columns and 
              f"{row['sell_exchange']}:low" in df.columns and 
              f"{row['sell_exchange']}:high" in df.columns and
              not np.isclose(row[f"{row['sell_exchange']}:high"], row[f"{row['sell_exchange']}:low"])) else np.nan,
        axis=1
    )
    df['price_position_sell_exchange'] = df['price_position_sell_exchange'].replace([np.inf, -np.inf], np.nan)
    
#endregion

#region LAYER 4 FEATURES
def add_L4_rolling_stats(df): 
    for window in windows:
        # Spread rolling statistics
        df[f'spread_rolling_std_{window}'] = df[f'spread_close_pct'].rolling(window=window).std()
        df[f'spread_rolling_max_{window}'] = df[f'spread_close_pct'].rolling(window=window).max()
        df[f'spread_rolling_min_{window}'] = df[f'spread_close_pct'].rolling(window=window).min()

def add_L4_spreads(df):
    df['opportunity_gap'] = df['spread_highlow_pct'] - df['spread_close_pct']
    df['min_volume'] = df[['volume_buy_exchange', 'volume_sell_exchange']].min(axis=1, skipna=True)
    df['volume_ratio'] = np.where(
        df['volume_buy_exchange'] != 0,
        df['volume_sell_exchange'] / df['volume_buy_exchange'],
        np.nan
    )
    for window in windows:
        # Spread range
        df[f'spread_range_{window}'] = (df[f'spread_close_pct'].rolling(window=window).max() - 
                                        df[f'spread_close_pct'].rolling(window=window).min())
        
def add_L4_zscore(df):
    for window in windows:
        rolling_mean = df[f'spread_close_pct'].rolling(window=window).mean()
        rolling_std = df[f'spread_close_pct'].rolling(window=window).std()
        df[f'spread_zscore_{window}'] = np.where(
            np.isclose(rolling_std, 0, 1e-9),
            np.nan,
            (df['spread_close_pct'] - rolling_mean) / rolling_std
        )
        df[f'spread_zscore_{window}'] = df[f'spread_zscore_{window}'].replace([np.inf, -np.inf], np.nan)

def add_L4_moving_averages(df):
    volume_cols = [f"{ex}:volume" for ex in exchanges if f"{ex}:volume" in df.columns]
    for window in windows:
        df[f'spread_ma_{window}'] = df[f'spread_close_pct'].rolling(window=window).mean()
        df[f'spread_ema_{window}'] = df[f'spread_close_pct'].ewm(span=window, adjust=False).mean()

        # Pre-compute rolling means for all exchange volume columns that exist
        for col in volume_cols:
            df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        
        # Per-row selection based on buy/sell exchange
        df[f'volume_ma_buy_{window}'] = df.apply(
            lambda row: row[f"{row['buy_exchange']}:volume_ma_{window}"],
            axis=1
        )
        df[f'volume_ma_sell_{window}'] = df.apply(
            lambda row: row[f"{row['sell_exchange']}:volume_ma_{window}"],
            axis=1
        )
        
        # Drop intermediate columns
        df.drop(columns=[f'{col}_ma_{window}' for col in volume_cols], inplace=True)

def add_L4_rate_change_features(df): 
    df[f'spread_rate_change'] = df[f'spread_close_pct'] - df[f'spread_close_pct'].shift(1)
    
    df[f'spread_rate_change_pct'] = np.where(
        np.isclose(df[f'spread_close_pct'].shift(1), 0, 1e-9),
        np.nan,
        df['spread_rate_change'] / df[f'spread_close_pct'].shift(1) * 100
    )
    df[f'spread_rate_change_pct'] = df[f'spread_rate_change_pct'].replace([np.inf, -np.inf], np.nan)

    df[f'spread_rate_acceleration'] = df[f'spread_rate_change'] - df[f'spread_rate_change'].shift(1)

def add_L4_flags(df): 
    df['is_opportunity'] = (df['spread_close_pct'] >= TRADING_COST_PCT).astype(int)
    df['is_real_opportunity'] = (df['spread_close_pct'] >= REAL_OPPORTUNITY_THRESHOLD).astype(int)

#endregion

#region LAYER 5 FEATURES
def add_L5_bollinger_bands(df, num_std=2): 
    for window in windows:
        df[f'spread_bb_upper_{window}'] = df[f'spread_ma_{window}'] + (df[f'spread_rolling_std_{window}'] * num_std)
        df[f'spread_bb_lower_{window}'] = df[f'spread_ma_{window}'] - (df[f'spread_rolling_std_{window}'] * num_std)
        
        upper = df[f'spread_bb_upper_{window}']
        lower = df[f'spread_bb_lower_{window}']
        denominator = upper - lower
        df[f'spread_bb_position_{window}'] = np.where(
            np.isclose(denominator, 0, 1e-9),
            np.nan,
            (df['spread_close_pct'] - lower) / denominator
        )
        df[f'spread_bb_position_{window}'] = df[f'spread_bb_position_{window}'].replace([np.inf, -np.inf], np.nan)

def add_L5_lag_features(df, lags=[1, 5, 10, 30]): 
    for lag in lags:
        # Spread lags
        df[f'spread_lag_{lag}'] = df[f'spread_close_pct'].shift(lag)
        
        # Volume lags
        df[f'volume_buy_lag_{lag}'] = df[f'volume_buy_exchange'].shift(lag)
        df[f'volume_sell_lag_{lag}'] = df[f'volume_sell_exchange'].shift(lag)
        df[f'min_volume_lag_{lag}'] = df[f'min_volume'].shift(lag)
        
        # Opportunity flag lags - BOTH versions
        df[f'is_opportunity_lag_{lag}'] = df[f'is_opportunity'].shift(lag)
        df[f'is_real_opportunity_lag_{lag}'] = df[f'is_real_opportunity'].shift(lag)
        
        # Price change lags
        df[f'price_change_buy_lag_{lag}'] = df[f'price_change_buy_exchange'].shift(lag)
        df[f'price_change_sell_lag_{lag}'] = df[f'price_change_sell_exchange'].shift(lag)
        
        # Volatility lags
        df[f'volatility_avg_lag_{lag}'] = df[f'volatility_avg'].shift(lag)
    
    # Categorical lags (exchange names)
    df[f'buy_exchange_lag_1'] = df[f'buy_exchange'].shift(1)
    df[f'sell_exchange_lag_1'] = df[f'sell_exchange'].shift(1)

    diff_lags = [1, 5]
    for lag in diff_lags:
        df[f'spread_diff_from_lag_{lag}'] = df[f'spread_close_pct'] - df[f'spread_lag_{lag}']
        df[f'volume_diff_from_lag_{lag}'] = df[f'min_volume'] - df[f'min_volume_lag_{lag}']

#endregion

#region ADDITIONAL METHODS
def save_featured_data():
    
    print("\n=== SAVING FEATURED DATA ===\n")
       
    datasets = {
        'BTCUSD': btcusd_data,
        'ETHUSD': ethusd_data,
        'DOGEUSD': dogeusd_data,
        'LINKUSD': linkusd_data,
        'SOLUSD': solusd_data,
        'XRPUSD': xrpusd_data
    }
    for name, df in datasets.items():
        output_file = f'{FEATURED_DATA_PATH}/featured_{name}_data.csv'
        df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file} ({len(df)} rows, {len(df.columns)} columns)")
    
    print("\n🎉 All featured data saved successfully!")

def layer2(df):
    print("\n  ✓ === LAYER 2 FEATURES ===")
    print("    • Adding minmax features...", end="", flush=True)
    add_L2_minmax_features(df)
    print(" ✓")
    print("    • Adding exchange features...", end="", flush=True)
    add_L2_exchange_features(df)
    print(" ✓")
    print("    • Adding time features...", end="", flush=True)
    add_L2_time_features(df)
    print(" ✓")
    print("    • Adding cross exchange price ratio features...", end="", flush=True)
    add_L2_cross_exchange_price_ratio(df)
    print(" ✓")
    
def layer3(df):
    print("\n  ✓ === LAYER 3 FEATURES ===")
    print("    • Adding spreads features...", end="", flush=True)
    add_L3_spreads(df)
    print(" ✓")
    print("    • Adding buy/sell exchange features...", end="", flush=True)
    add_L3_buy_sell_exchange_features(df)
    print(" ✓")
    print("    • Adding price change features...", end="", flush=True)
    add_L3_price_change_features(df)
    print(" ✓")
    print("    • Adding buy/sell exchange price ratio features...", end="", flush=True)
    add_L3_buy_sell_exchange_price_ratio(df)
    print(" ✓")
    print("    • Adding rolling stats features...", end="", flush=True)
    add_L3_rolling_stats(df)
    print(" ✓")
    print("    • Adding volatility features...", end="", flush=True)
    add_L3_volatility_features(df)
    print(" ✓")

def layer4(df):
    print("\n  ✓ === LAYER 4 FEATURES ===")
    print("    • Adding rolling stats features...", end="", flush=True)
    add_L4_rolling_stats(df)
    print(" ✓")
    print("    • Adding spreads features...", end="", flush=True)
    add_L4_spreads(df)
    print(" ✓")
    print("    • Adding zscore features...", end="", flush=True)
    add_L4_zscore(df)
    print(" ✓")
    print("    • Adding moving averages features...", end="", flush=True)
    add_L4_moving_averages(df)
    print(" ✓")
    print("    • Adding rate change features...", end="", flush=True)
    add_L4_rate_change_features(df)
    print(" ✓")
    print("    • Adding flags features...", end="", flush=True)
    add_L4_flags(df)
    print(" ✓")

def layer5(df):
    print("\n  ✓ === LAYER 5 FEATURES ===")
    print("    • Adding bollinger bands features...", end="", flush=True)
    add_L5_bollinger_bands(df)
    print(" ✓")
    print("    • Adding lag features...", end="", flush=True)
    add_L5_lag_features(df)
    print(" ✓")

#endregion

def main():
    print("\n" + "="*60)
    print("=== FEATURE ENGINEERING PIPELINE ===")
    print("="*60)
    print(f"Processing {len(data_frames)} cryptocurrencies...\n")
    
    symbol_names = ['BTCUSD', 'ETHUSD', 'DOGEUSD', 'LINKUSD', 'SOLUSD', 'XRPUSD']
    for idx, df in enumerate(data_frames):
        symbol = symbol_names[idx] if idx < len(symbol_names) else f"Coin{idx}"
        print(f"\nProcessing: {symbol} ({idx+1}/{len(data_frames)})")
        layer2(df)
        layer3(df)
        layer4(df)
        layer5(df)
    
    print("\n" + "="*60)
    print("✅ All features added successfully!")
    print("="*60 + "\n")
    save_featured_data()
    return


if __name__ == "__main__":
    main()