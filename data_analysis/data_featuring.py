import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

#region Hyper-parameters
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

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
original_features = ["high","low","open","close","volume"]

# Change time string from CSV to time type
for df in data_frames:
    df['time'] = pd.to_datetime(df['time'])

    ## FEATURE ENGINEERING ##
#endregion

#region LAYER 2 FEATURES
def add_close_spread(df):  # L2, from L1 close prices
    """
    LAYER 2: Core Spread Features
    Uses: L1 close prices from all exchanges
    Creates: min_close, max_close, buy_exchange, sell_exchange, spread_close_pct, 
             is_opportunity, is_real_opportunity, num_exchanges_available
    
    Identifies arbitrage opportunities by finding min/max close prices across all exchanges.
    This is the foundation - all other arbitrage features depend on spread_close_pct.
    Uses ALL exchanges (skipna=True) to handle partial data.
    """
    close_cols = [f"{ex}:close" for ex in exchanges if f"{ex}:close" in df.columns]
    
    # min max close identification
    df['min_close'] = df[close_cols].min(axis=1, skipna=True)
    df['max_close'] = df[close_cols].max(axis=1, skipna=True)
    df['buy_exchange'] = df[close_cols].idxmin(axis=1, skipna=True).str.split(':').str[0]
    df['sell_exchange'] = df[close_cols].idxmax(axis=1, skipna=True).str.split(':').str[0]
    
    # spread close calculation
    df['spread_close_absolute'] = df['max_close'] - df['min_close']
    df['spread_close_pct'] = (df['spread_close_absolute'] / df['min_close']) * 100
    
    # Opportunity flags (from section 11)
    df['is_opportunity'] = (df['spread_close_pct'] >= TRADING_COST_PCT).astype(int)
    df['is_real_opportunity'] = (df['spread_close_pct'] >= REAL_OPPORTUNITY_THRESHOLD).astype(int)
    
    # data quality check
    df['num_exchanges_available'] = df[close_cols].notna().sum(axis=1)

def add_high_low_spread(df):  # L2, from L1 high/low prices
    """
    LAYER 2: High-Low Spread Features
    Uses: L1 high and low prices from all exchanges
    Creates: max_high, min_low, spread_highlow_pct, opportunity_gap
    
    Calculates theoretical maximum arbitrage if perfect timing was achieved.
    Uses ALL exchanges (skipna=True). Depends on spread_close_pct from add_close_spread.
    """
    high_cols = [f"{ex}:high" for ex in exchanges if f"{ex}:high" in df.columns]
    low_cols = [f"{ex}:low" for ex in exchanges if f"{ex}:low" in df.columns]
    # high low identification
    df['max_high'] = df[high_cols].max(axis=1, skipna=True)
    df['min_low'] = df[low_cols].min(axis=1,skipna= True)
    df['high_exchange'] = df[high_cols].idxmax(axis=1, skipna=True).str.split(':').str[0]
    df['low_exchange'] = df[low_cols].idxmin(axis=1, skipna=True).str.split(':').str[0]
    
    # high low spread calculation
    df['spread_highlow_absolute'] = df['max_high'] - df['min_low']
    df['spread_highlow_pct'] = (df['spread_highlow_absolute'] / df['min_low']) * 100
    df['opportunity_gap'] = df['spread_highlow_pct'] - df['spread_close_pct']

def add_time_features(df):  # L2, from L1 time
    """
    LAYER 2: Time Features
    Uses: L1 time column only
    Creates: hour, minute, day_of_week, is_weekend, overlap_hours
    
    Extracts time-based features to capture cyclical patterns.
    No exchange data needed - time always available.
    """
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df[f'overlap_hours'] = df[f'hour'].apply(lambda x: 1 if 19 <= x <= 21 else 0)  # Example: active trading hours

#endregion

#region LAYER 3 FEATURES
def add_volume_features(df):  # L3, from L2 buy/sell_exchange, from L1 buy/sell volume
    """
    LAYER 3: Volume Features  
    Uses: L2 buy_exchange + sell_exchange, L1 volume from those exchanges
    Creates: volume_buy_exchange, volume_sell_exchange, min_volume, volume_ratio
    
    Extracts volume from the identified buy/sell exchanges to assess executability.
    Uses ONLY BUY and SELL exchanges (2 exchanges).
    If buy or sell exchange volume is NaN, all features become NaN.
    """
    # Need .apply() because each row uses a DIFFERENT column
    df['volume_buy_exchange'] = df.apply(
        lambda row: row[f"{row['buy_exchange']}:volume"], 
        axis=1
    )
    df['volume_sell_exchange'] = df.apply(
        lambda row: row[f"{row['sell_exchange']}:volume"], 
        axis=1
    )
    
    # Now we can use vectorized operations
    df['min_volume'] = df[['volume_buy_exchange', 'volume_sell_exchange']].min(axis=1, skipna=True)
    df['volume_ratio'] = df['volume_sell_exchange'] / df['volume_buy_exchange']

def add_volatility_features(df):  # L3, from L2 buy/sell_exchange, from L1 high/low/close prices
    """
    LAYER 3: Volatility Features
    Uses: L2 buy_exchange + sell_exchange, L1 high/low/close from all exchanges
    Creates: {EXCHANGE}_volatility, volatility_avg/max/min, price_position_buy/sell_exchange
    
    Measures intra-minute volatility and where close sits in high-low range.
    Uses ALL exchanges for volatility aggregates, ONLY BUY/SELL for price positions.
    """
    for exchange in exchanges:
        if f'{exchange}:high' in df.columns and f'{exchange}:low' in df.columns and f'{exchange}:close' in df.columns:
            df[f'{exchange}_volatility'] = (df[f'{exchange}:high'] - df[f'{exchange}:low']) / df[f'{exchange}:close'] * 100
    
    # Get available volatility columns
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

def add_price_change_features(df):  # L3, from L2 buy/sell_exchange, from L1 close/open price
    """
    LAYER 3: Price Change Features
    Uses: L2 buy_exchange + sell_exchange, L1 close and open from all exchanges
    Creates: {EXCHANGE}_price_change, price_change_buy/sell_exchange
    
    Calculates intra-minute price change (close-open) to detect momentum.
    Uses ALL exchanges for per-exchange changes, ONLY BUY/SELL for specific features.
    """
    for exchange in exchanges:
        if f'{exchange}:close' in df.columns and f'{exchange}:open' in df.columns:
            df[f'{exchange}_price_change'] = (df[f'{exchange}:close'] - df[f'{exchange}:open']) / df[f'{exchange}:open'] * 100
    
    df['price_change_buy_exchange'] = df.apply(
        lambda row: row[f"{row['buy_exchange']}_price_change"] if f"{row['buy_exchange']}_price_change" in df.columns else np.nan,
        axis=1
    )
    df['price_change_sell_exchange'] = df.apply(
        lambda row: row[f"{row['sell_exchange']}_price_change"] if f"{row['sell_exchange']}_price_change" in df.columns else np.nan,
        axis=1
    )
  
def add_bollinger_bands(df, windows=[5, 15, 30], num_std=2):  # L3, from L2 spread_close_pct
    """
    LAYER 3: Bollinger Bands
    Uses: L2 spread_close_pct only
    Creates: spread_bb_ma/std/upper/lower/position_{window} for each window
    
    Detects statistical extremes in spread using Bollinger Bands.
    Note: spread_bb_ma_{window} duplicates spread_ma_{window} from add_moving_averages
    """
    for window in windows:
        df[f'spread_bb_upper_{window}'] = df[f'spread_ma_{window}'] + (df[f'spread_rolling_std_{window}'] * num_std)
        df[f'spread_bb_lower_{window}'] = df[f'spread_ma_{window}'] - (df[f'spread_rolling_std_{window}'] * num_std)
        
        lower = df[f'spread_bb_lower_{window}']
        upper = df[f'spread_bb_upper_{window}']
        denominator = upper - lower
        df[f'spread_bb_position_{window}'] = np.where(
            np.isclose(denominator, 0, 1e-9),
            np.nan,
            (df['spread_close_pct'] - lower) / denominator
        )
        df[f'spread_bb_position_{window}'] = df[f'spread_bb_position_{window}'].replace([np.inf, -np.inf], np.nan, inplace=True)

def add_rate_change_features(df):  # L3, from L2 spread_close_pct
    """
    LAYER 3: Rate Change Features
    Uses: L2 spread_close_pct only
    Creates: spread_rate_change, spread_rate_change_pct, spread_rate_acceleration
    
    Calculates first and second derivatives of spread (momentum and acceleration).
    """
    df[f'spread_rate_change'] = df[f'spread_close_pct'] - df[f'spread_close_pct'].shift(1)
    
    df[f'spread_rate_change_pct'] = np.where(
        np.isclose(df[f'spread_close_pct'].shift(1), 0, 1e-9),
        np.nan,
        df['spread_rate_change'] / df[f'spread_close_pct'].shift(1) * 100
    )
    df[f'spread_rate_change_pct'] = df[f'spread_rate_change_pct'].replace([np.inf, -np.inf], np.nan, inplace=True)

    df[f'spread_rate_acceleration'] = df[f'spread_rate_change'] - df[f'spread_rate_change'].shift(1)

def add_cross_ex_price_ratio(df):  # L3, from L2 buy/sell_exchange, from L1 close prices
    """
    LAYER 3: Cross-Exchange Price Ratios
    Uses: L2 buy_exchange + sell_exchange, L1 close prices from all exchanges
    Creates: price_ratio_buy_sell, price_ratio_{EX1}_{EX2} for all pairs, avg/max/min/std aggregates
    
    Calculates price ratios between all exchange pairs to measure market fragmentation.
    Ratios complement spreads by being price-normalized.
    """
    # Price ratio between buy and sell exchanges
    df['price_ratio_buy_sell'] = df.apply(
        lambda row: row[f"{row['sell_exchange']}:close"] / row[f"{row['buy_exchange']}:close"],
        axis=1
    )
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

#endregion

#region LAYER 4 FEATURES

def add_moving_averages(df, windows=[5, 15, 30]):  # L4, from L3 volume_buy/sell_exchange, from L2 spread_close_pct
    """
    LAYER 4: Moving Averages
    Uses: L3 volume_buy_exchange + volume_sell_exchange, L2 spread_close_pct
    Creates: spread_ma/ema_{window}, volume_ma_buy/sell_{window} for each window
    
    Calculates SMA and EMA for spread and volume to detect trends.
    ⚠️  WARNING: Volume MAs mix exchanges over time - see ROLLING_STATS_ISSUE.md
    """
    # get name of buy exchange
    for window in windows:
        df[f'spread_ma_{window}'] = df[f'spread_close_pct'].rolling(window=window).mean()
        df[f'spread_ema_{window}'] = df[f'spread_close_pct'].ewm(span=window, adjust=False).mean()

        # Pre-compute rolling means for all exchange volume columns that exist
        vol_cols = [f'{ex}:volume' for ex in exchanges if f'{ex}:volume' in df.columns]
        for col in vol_cols:
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
        df.drop(columns=[f'{col}_ma_{window}' for col in vol_cols], inplace=True)
    
def add_rolling_stats(df, windows=[5, 15, 30]):  # L4, from L3 volume_buy/sell_exchange, from L2 spread_close_pct + is_opportunity + is_real_opportunity
    """
    LAYER 4: Rolling Statistics
    Uses: L3 volume_buy/sell_exchange, L2 spread_close_pct + is_opportunity + is_real_opportunity
    Creates: spread_rolling_std/max/min/range/zscore, volume_rolling_std, opportunities_in_last_{window}
    
    Calculates rolling statistics for spread stability and opportunity clustering.
    ⚠️  WARNING: Volume stats mix exchanges over time - see ROLLING_STATS_ISSUE.md
    """
    
    for window in windows:
        # Spread rolling statistics
        df[f'spread_rolling_std_{window}'] = df[f'spread_close_pct'].rolling(window=window).std()
        df[f'spread_rolling_max_{window}'] = df[f'spread_close_pct'].rolling(window=window).max()
        df[f'spread_rolling_min_{window}'] = df[f'spread_close_pct'].rolling(window=window).min()
        
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
        
        # Spread range
        df[f'spread_range_{window}'] = (df[f'spread_close_pct'].rolling(window=window).max() - 
                                        df[f'spread_close_pct'].rolling(window=window).min())
        
        # Z-score
        rolling_mean = df[f'spread_close_pct'].rolling(window=window).mean()
        rolling_std = df[f'spread_close_pct'].rolling(window=window).std()
        df[f'spread_zscore_{window}'] = np.where(
            np.isclose(rolling_std, 0, 1e-9),
            np.nan,
            (df['spread_close_pct'] - rolling_mean) / rolling_std
        )
        df[f'spread_zscore_{window}'] = df[f'spread_zscore_{window}'].replace([np.inf, -np.inf], np.nan, inplace=True)

def add_lag_features(df, lags=[1, 5, 10, 30]):  # L4, from L3 volume/price_change/volatility, from L2 spread/buy_sell_exchange/opportunities
    """
    LAYER 4: Lag Features
    Uses: L3 volume_buy/sell_exchange + price_change_buy/sell_exchange + volatility_avg,
          L2 spread_close_pct + buy/sell_exchange + is_opportunity + is_real_opportunity
    Creates: Lagged versions of all above features (spread_lag, volume_lag, opportunity_lag, etc.)
    
    Creates time-lagged versions of key features for time-series prediction.
    Essential for ML models to learn temporal dependencies.
    """
    
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
    
    # Diff features (change from lag)
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
    add_close_spread(df)
    add_high_low_spread(df)
    add_time_features(df)

def layer3(df):
    add_volume_features(df)
    add_volatility_features(df)
    add_price_change_features(df)
    add_bollinger_bands(df)
    add_rate_change_features(df)
    add_cross_ex_price_ratio(df)

def layer4(df):
    add_moving_averages(df)
    add_rolling_stats(df)
    add_lag_features(df)

#endregion

def main():
    print("\n=== ADDING FEATURES ===\n")
        
    for df in data_frames:
        layer2(df)
        layer3(df)
        layer4(df) 
    
    print("✅ Features added!\n")
    save_featured_data()
    return


if __name__ == "__main__":
    main()