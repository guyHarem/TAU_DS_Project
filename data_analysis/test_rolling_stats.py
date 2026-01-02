import pandas as pd
import numpy as np

# Create a simple test dataframe
df = pd.DataFrame({
    'time': pd.date_range('2025-01-01', periods=15, freq='1min'),
    'BINANCE_volume': [100, 150, np.nan, 200, 180, 160, np.nan, 220, 190, 210, 
                       np.nan, 240, 250, np.nan, 300],
    'COINBASE_volume': [120, np.nan, 180, 190, 170, np.nan, 210, 200, 220, 230, 
                        240, np.nan, 260, 270, 280],
    'spread': [0.5, 0.6, 0.7, 0.55, 0.65, 0.75, 0.8, 0.6, 0.7, 0.65, 
               0.55, 0.6, 0.7, 0.8, 0.5]
})

print("=" * 80)
print("ORIGINAL DATAFRAME")
print("=" * 80)
print(df.to_string())
print("\n")

# ========== TEST 1: Pandas native rolling (default min_periods=window) ==========
print("=" * 80)
print("TEST 1: Pandas Native .rolling() - Default min_periods=window")
print("This waits for full window before calculating")
print("=" * 80)

df['BINANCE_rolling_mean_5_default'] = df['BINANCE_volume'].rolling(window=5).mean()

print(df[['BINANCE_volume', 'BINANCE_rolling_mean_5_default']].to_string())
print("\nNotice: Rows 0-3 are NaN (waiting for 5 values)")
print("Row 4 first gets a value (rows 0-4 = 5 values, but 1 is NaN, so mean of 4 values)")
print("\n")

# ========== TEST 2: Pandas rolling with min_periods=1 ==========
print("=" * 80)
print("TEST 2: Pandas .rolling() - min_periods=1")
print("Calculates with whatever data exists (ignores nulls)")
print("=" * 80)

df['BINANCE_rolling_mean_5_minp1'] = df['BINANCE_volume'].rolling(window=5, min_periods=1).mean()

print(df[['BINANCE_volume', 'BINANCE_rolling_mean_5_minp1']].to_string())
print("\nNotice:")
print("Row 0: mean of [100] (1 value)")
print("Row 1: mean of [100, 150] (2 values, ignores NaN at row 2)")
print("Row 2: mean of [100, 150, NaN] → only [100, 150] (2 values)")
print("Row 3: mean of last 5 rows [150, NaN, 200] → only [150, 200] (2 values)")
print("Row 4: mean of last 5 rows [NaN, 200, 180] → only [200, 180] (2 values)")
print("\n")

# ========== TEST 3: Compare spread (no nulls) ==========
print("=" * 80)
print("TEST 3: Spread rolling (no nulls) - see the difference")
print("=" * 80)

df['spread_rolling_mean_5_default'] = df['spread'].rolling(window=5).mean()
df['spread_rolling_mean_5_minp1'] = df['spread'].rolling(window=5, min_periods=1).mean()

print(df[['spread', 'spread_rolling_mean_5_default', 'spread_rolling_mean_5_minp1']].to_string())
print("\nSpread has no nulls, so both behave the same starting at row 4")
print("\n")

# ========== TEST 4: What you want - Exchange-specific per row ==========
print("=" * 80)
print("TEST 4: Exchange-Specific Rolling (What You Want)")
print("=" * 80)

# Create a dataframe where each row specifies which exchange to use
df['use_exchange'] = ['BINANCE', 'COINBASE', 'BINANCE', 'COINBASE', 'BINANCE', 
                      'COINBASE', 'BINANCE', 'COINBASE', 'BINANCE', 'COINBASE',
                      'BINANCE', 'COINBASE', 'BINANCE', 'COINBASE', 'BINANCE']

print("\nDataframe with exchange selection per row:")
print(df[['time', 'use_exchange', 'BINANCE_volume', 'COINBASE_volume']].to_string())

# Method: Pre-calculate rolling for each exchange, then extract
df['BINANCE_rolling_mean_5'] = df['BINANCE_volume'].rolling(window=5, min_periods=1).mean()
df['COINBASE_rolling_mean_5'] = df['COINBASE_volume'].rolling(window=5, min_periods=1).mean()

# Now extract based on which exchange is selected for that row
df['rolling_mean_for_my_exchange'] = df.apply(
    lambda row: row[f"{row['use_exchange']}_rolling_mean_5"],
    axis=1
)

print("\n" + "=" * 80)
print("RESULT: Each row uses rolling mean from its specified exchange")
print("=" * 80)
print(df[['use_exchange', 'BINANCE_volume', 'COINBASE_volume', 
          'BINANCE_rolling_mean_5', 'COINBASE_rolling_mean_5', 
          'rolling_mean_for_my_exchange']].to_string())

print("\n\nExplanation:")
print("Row 0: use_exchange=BINANCE → extract BINANCE_rolling_mean_5 = 100.0 (mean of [100])")
print("Row 1: use_exchange=COINBASE → extract COINBASE_rolling_mean_5 = 120.0 (mean of [120])")
print("Row 2: use_exchange=BINANCE → extract BINANCE_rolling_mean_5 = mean([100, 150]) = 125.0")
print("Row 3: use_exchange=COINBASE → extract COINBASE_rolling_mean_5 = mean([120, 180, 190]) = 163.33")
print("\n✅ Each row only looks at its specific exchange's hard window of 5 rows!")
print("✅ Nulls are ignored within that window, but window doesn't extend past them")