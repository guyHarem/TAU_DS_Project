# Rolling Statistics Issue: Exchange Mixing Problem

## The Problem

When calculating rolling statistics (moving averages, Bollinger Bands, rolling stats) on exchange-specific features (like `volume_buy_exchange` or `volume_sell_exchange`), the functions mix data from **different exchanges across time** because the `buy_exchange` and `sell_exchange` can change minute-to-minute.

### Example Scenario

Your data structure:
```
Row  Time   COINBASE  BINANCE  KRAKEN  buy_exchange  volume_buy_exchange
107  10:00  100.5     100.2    100.3   COINBASE      1500
106  09:59  NULL      100.1    100.2   BINANCE       2000
105  09:58  NULL      100.0    100.1   BINANCE       1800
104  09:57  NULL      99.9     99.8    KRAKEN        1200
103  09:56  99.8      99.7     99.6    BINANCE       1700
...
93   09:46  NULL      99.5     99.6    KRAKEN        1200
92   09:45  98.0      98.2     98.1    COINBASE      1600
```

**When calculating `volume_ma_buy_15` at row 107:**
```python
df['volume_ma_buy_15'] = df['volume_buy_exchange'].rolling(window=15).mean()
```

The rolling window looks at rows 93-107 (15 rows = 15 minutes) and calculates the mean of `volume_buy_exchange`:
- Uses [1500 (COINBASE), 2000 (BINANCE), 1800 (BINANCE), 1200 (KRAKEN), ..., 1600 (COINBASE)]
- **Problem**: These volumes are from DIFFERENT exchanges mixed together!

## Two Separate Issues

### Issue 1: Exchange Mixing
The moving average mixes volumes from different exchanges (COINBASE, BINANCE, KRAKEN, etc.) into a single metric. 

**Is this a problem?**
- **For general market trends**: NO - it's acceptable if you want to know "what's the recent volume environment?"
- **For exchange-specific patterns**: YES - if you want COINBASE-specific behavior when buying from COINBASE

### Issue 2: Sparse Data
Some exchanges only have data sporadically. For example:
- COINBASE data at rows 107 and 92 only
- But we're calculating a 15-minute MA with only 2 COINBASE data points scattered across 15 minutes
- The MA includes many minutes where COINBASE had no data

**Result**: The "rolling 15" might only have 2-5 actual data points instead of 15.

## Impact on Different Layers

### Affected Functions:
1. **`add_moving_averages` (Layer 7)**:
   - `volume_ma_buy_{window}` - mixes different exchange volumes
   - `volume_ma_sell_{window}` - mixes different exchange volumes
   - `spread_ma_{window}` - OK (spread is already aggregated across exchanges)
   - `spread_ema_{window}` - OK

2. **`add_bollinger_bands` (Layer 8)**:
   - All BB features use `spread_close_pct` - OK (no exchange mixing)

3. **`add_rolling_stats` (Layer 9)**:
   - `volume_buy_rolling_std_{window}` - mixes different exchange volumes
   - `volume_sell_rolling_std_{window}` - mixes different exchange volumes
   - `spread_*` features - OK

4. **`add_lag_features` (Layer 12)**:
   - Lag features are snapshots, not rolling calculations - mostly OK
   - But lagged volumes/price changes might reference different exchanges than current ones

## Solutions

### Solution 1: Accept Exchange Mixing (Simplest - Current Approach)

**Keep current implementation** - treat volume features as "general market volume trend" rather than exchange-specific.

**Pros:**
- Simple, fast
- Works with sparse data
- Tells you "recent volume activity in the market"
- Acceptable for many ML use cases

**Cons:**
- Doesn't capture exchange-specific behavior
- Mixes different liquidity sources

**Use when:** You care about general market patterns, not exchange-specific nuances.

---

### Solution 2: Add Minimum Data Quality Requirements

Require a minimum number of valid data points to prevent sparse calculations.

```python
def add_moving_averages(df, windows=[5, 15, 30]):
    """
    Add moving averages with data quality requirements
    """
    for window in windows:
        # Require at least 60% of data points to be valid
        min_periods = max(1, int(window * 0.6))
        
        df[f'spread_ma_{window}'] = df['spread_close_pct'].rolling(
            window=window, 
            min_periods=min_periods
        ).mean()
        
        df[f'volume_ma_buy_{window}'] = df['volume_buy_exchange'].rolling(
            window=window, 
            min_periods=min_periods
        ).mean()
        
        df[f'volume_ma_sell_{window}'] = df['volume_sell_exchange'].rolling(
            window=window, 
            min_periods=min_periods
        ).mean()
        
        df[f'spread_ema_{window}'] = df['spread_close_pct'].ewm(
            span=window, 
            adjust=False, 
            min_periods=min_periods
        ).mean()
```

**Pros:**
- Prevents MAs based on too little data
- More reliable calculations
- Easy to implement

**Cons:**
- Creates more NaN rows in your dataset
- May lose valid data points

**Use when:** Data quality is more important than data quantity.

---

### Solution 3: Add Data Quality Tracking (Recommended)

Keep all calculations but track how much valid data went into each one.

```python
def add_moving_averages(df, windows=[5, 15, 30]):
    """
    Add moving averages with quality tracking
    """
    for window in windows:
        # Standard rolling calculations
        df[f'spread_ma_{window}'] = df['spread_close_pct'].rolling(
            window=window, 
            min_periods=1
        ).mean()
        
        df[f'volume_ma_buy_{window}'] = df['volume_buy_exchange'].rolling(
            window=window, 
            min_periods=1
        ).mean()
        
        df[f'volume_ma_sell_{window}'] = df['volume_sell_exchange'].rolling(
            window=window, 
            min_periods=1
        ).mean()
        
        df[f'spread_ema_{window}'] = df['spread_close_pct'].ewm(
            span=window, 
            adjust=False
        ).mean()
        
        # Track data quality - how many valid points were used
        df[f'spread_ma_{window}_count'] = df['spread_close_pct'].rolling(
            window=window
        ).count()
        
        df[f'volume_ma_buy_{window}_count'] = df['volume_buy_exchange'].rolling(
            window=window
        ).count()
        
        df[f'volume_ma_sell_{window}_count'] = df['volume_sell_exchange'].rolling(
            window=window
        ).count()
        
        # Quality ratio (0-1, where 1 = all data points present)
        df[f'spread_ma_{window}_quality'] = df[f'spread_ma_{window}_count'] / window
        df[f'volume_ma_buy_{window}_quality'] = df[f'volume_ma_buy_{window}_count'] / window
        df[f'volume_ma_sell_{window}_quality'] = df[f'volume_ma_sell_{window}_count'] / window
```

**Pros:**
- Keeps all data
- Allows filtering/weighting by quality later
- Model can learn to use/ignore low-quality MAs
- Provides transparency

**Cons:**
- Adds extra columns
- Slightly more complex

**Use when:** You want maximum flexibility and transparency.

**Usage Example:**
```python
# Filter out low-quality MAs
df_filtered = df[df['volume_ma_buy_15_quality'] >= 0.6]  # At least 60% valid data

# Or use quality as a model feature
# Model learns which quality levels are reliable
```

---

### Solution 4: Time-Based Rolling (Temporal Correctness)

Use pandas' time-based rolling to ensure you only look at recent CALENDAR time, not just row count.

```python
def add_moving_averages(df, windows=[5, 15, 30]):
    """
    Add time-based moving averages
    Ensures calculations only use data from the last N minutes of actual time
    """
    # Ensure time is the index
    df_indexed = df.set_index('time')
    
    for window in windows:
        # Time-based rolling: '15T' = 15 minutes of time
        df[f'spread_ma_{window}'] = df_indexed['spread_close_pct'].rolling(
            f'{window}T', 
            min_periods=1
        ).mean().values
        
        df[f'volume_ma_buy_{window}'] = df_indexed['volume_buy_exchange'].rolling(
            f'{window}T', 
            min_periods=1
        ).mean().values
        
        df[f'volume_ma_sell_{window}'] = df_indexed['volume_sell_exchange'].rolling(
            f'{window}T', 
            min_periods=1
        ).mean().values
        
        df[f'spread_ema_{window}'] = df_indexed['spread_close_pct'].ewm(
            span=window, 
            adjust=False
        ).mean().values
```

**Pros:**
- Guaranteed to only use data from the last N **minutes** of time
- Handles irregular time gaps correctly
- No future data leakage

**Cons:**
- Still mixes exchanges, just ensures temporal correctness
- Slightly slower
- Requires time to be sortable/indexable

**Use when:** Your data has irregular time gaps and temporal accuracy is critical.

---

### Solution 5: Exchange-Specific Rolling (Most Accurate, Most Complex)

Calculate rolling features separately for each exchange, then select the relevant one.

```python
def add_exchange_specific_rolling(df, windows=[5, 15, 30]):
    """
    Calculate rolling features specific to each exchange
    Most accurate but slowest approach
    """
    for exchange in exchanges:
        # Create exchange-specific volume column
        # Only has values when this exchange is the buy exchange, NaN otherwise
        df[f'{exchange}_volume_when_buy'] = df.apply(
            lambda row: row[f'{exchange}:volume'] 
                if row['buy_exchange'] == exchange 
                else np.nan,
            axis=1
        )
        
        # Rolling calculations only on that exchange's data
        for window in windows:
            df[f'{exchange}_volume_ma_{window}'] = df[
                f'{exchange}_volume_when_buy'
            ].rolling(window=window, min_periods=1).mean()
    
    # Then select the relevant exchange's MA based on current buy_exchange
    for window in windows:
        df[f'volume_ma_buy_exchange_specific_{window}'] = df.apply(
            lambda row: row[f"{row['buy_exchange']}_volume_ma_{window}"],
            axis=1
        )
```

**Pros:**
- Pure exchange-specific patterns
- No cross-exchange contamination
- Most accurate representation

**Cons:**
- Much slower (uses `.apply()`)
- Creates many intermediate columns (6 exchanges × 3 windows = 18+ columns)
- Still has sparse data issues for each individual exchange

**Use when:** Exchange-specific behavior is critical and you have computational resources.

---

### Solution 6: Hybrid Approach (Recommended for Production)

Combine Solution 3 (quality tracking) with minimal data requirements.

```python
def add_moving_averages_hybrid(df, windows=[5, 15, 30]):
    """
    Hybrid: Quality tracking + minimum data requirements
    Best balance of reliability and data retention
    """
    for window in windows:
        # Require at least 40% valid data (less strict than Solution 2)
        min_periods = max(1, int(window * 0.4))
        
        # Calculate MAs
        df[f'spread_ma_{window}'] = df['spread_close_pct'].rolling(
            window=window, 
            min_periods=min_periods
        ).mean()
        
        df[f'volume_ma_buy_{window}'] = df['volume_buy_exchange'].rolling(
            window=window, 
            min_periods=min_periods
        ).mean()
        
        df[f'volume_ma_sell_{window}'] = df['volume_sell_exchange'].rolling(
            window=window, 
            min_periods=min_periods
        ).mean()
        
        # Track quality
        df[f'volume_ma_buy_{window}_count'] = df['volume_buy_exchange'].rolling(
            window=window
        ).count()
        
        df[f'volume_ma_buy_{window}_quality'] = (
            df[f'volume_ma_buy_{window}_count'] / window
        )
        
        # Flag low-quality MAs
        df[f'volume_ma_buy_{window}_reliable'] = (
            df[f'volume_ma_buy_{window}_quality'] >= 0.5
        ).astype(int)
```

---

## Recommendation Summary

For your cryptocurrency arbitrage project:

| Solution | Best For | Difficulty | Data Loss | Accuracy |
|----------|----------|------------|-----------|----------|
| 1. Accept Mixing | Quick analysis, general trends | Easy | None | Low |
| 2. Min Periods | Higher quality data | Easy | Some | Medium |
| 3. Quality Tracking ⭐ | Transparency + flexibility | Medium | None | Medium-High |
| 4. Time-Based | Irregular timestamps | Medium | Some | Medium |
| 5. Exchange-Specific | Exchange behavior analysis | Hard | None | Highest |
| 6. Hybrid ⭐⭐ | Production ML models | Medium | Minimal | High |

**Recommended:** Start with **Solution 3 (Quality Tracking)** for analysis, then move to **Solution 6 (Hybrid)** for final model training.

---

## Implementation Example

Here's a complete implementation of Solution 3 for `add_moving_averages`:

```python
def add_moving_averages(df, windows=[5, 15, 30]):
    """
    LAYER 7: Moving Averages with Quality Tracking
    
    Calculates moving averages and tracks how much valid data went into each calculation.
    This allows filtering/weighting by data quality later.
    """
    for window in windows:
        # Spread MA (no quality issues - spread always calculated)
        df[f'spread_ma_{window}'] = df['spread_close_pct'].rolling(
            window=window, min_periods=1
        ).mean()
        
        # Volume MAs (may have sparse data)
        df[f'volume_ma_buy_{window}'] = df['volume_buy_exchange'].rolling(
            window=window, min_periods=1
        ).mean()
        
        df[f'volume_ma_sell_{window}'] = df['volume_sell_exchange'].rolling(
            window=window, min_periods=1
        ).mean()
        
        df[f'spread_ema_{window}'] = df['spread_close_pct'].ewm(
            span=window, adjust=False
        ).mean()
        
        # Quality tracking - count of valid points used
        df[f'volume_ma_buy_{window}_valid_count'] = df['volume_buy_exchange'].rolling(
            window=window
        ).count()
        
        df[f'volume_ma_sell_{window}_valid_count'] = df['volume_sell_exchange'].rolling(
            window=window
        ).count()
        
        # Quality ratio (0.0 to 1.0)
        df[f'volume_ma_buy_{window}_quality'] = (
            df[f'volume_ma_buy_{window}_valid_count'] / window
        )
        
        df[f'volume_ma_sell_{window}_quality'] = (
            df[f'volume_ma_sell_{window}_valid_count'] / window
        )
        
        # Binary quality flag (1 = reliable, 0 = unreliable)
        quality_threshold = 0.6  # Require 60% valid data
        df[f'volume_ma_buy_{window}_reliable'] = (
            df[f'volume_ma_buy_{window}_quality'] >= quality_threshold
        ).astype(int)
        
        df[f'volume_ma_sell_{window}_reliable'] = (
            df[f'volume_ma_sell_{window}_quality'] >= quality_threshold
        ).astype(int)

# Same pattern applies to add_rolling_stats
def add_rolling_stats(df, windows=[5, 15, 30]):
    """
    LAYER 9: Rolling Statistics with Quality Tracking
    """
    for window in windows:
        # Spread stats (always reliable)
        df[f'spread_rolling_std_{window}'] = df['spread_close_pct'].rolling(
            window=window, min_periods=1
        ).std()
        
        df[f'spread_rolling_max_{window}'] = df['spread_close_pct'].rolling(
            window=window, min_periods=1
        ).max()
        
        df[f'spread_rolling_min_{window}'] = df['spread_close_pct'].rolling(
            window=window, min_periods=1
        ).min()
        
        # Volume stats (may be sparse)
        df[f'volume_buy_rolling_std_{window}'] = df['volume_buy_exchange'].rolling(
            window=window, min_periods=1
        ).std()
        
        df[f'volume_sell_rolling_std_{window}'] = df['volume_sell_exchange'].rolling(
            window=window, min_periods=1
        ).std()
        
        # Quality tracking for volume stats
        df[f'volume_buy_std_{window}_valid_count'] = df['volume_buy_exchange'].rolling(
            window=window
        ).count()
        
        df[f'volume_buy_std_{window}_quality'] = (
            df[f'volume_buy_std_{window}_valid_count'] / window
        )
        
        # Opportunity counts (usually reliable if spread exists)
        df[f'opportunities_in_last_{window}'] = df['is_opportunity'].rolling(
            window=window, min_periods=1
        ).sum()
        
        df[f'real_opportunities_in_last_{window}'] = df['is_real_opportunity'].rolling(
            window=window, min_periods=1
        ).sum()
        
        # ... rest of rolling stats ...
```

---

## Usage in Analysis/Modeling

### Filtering Low-Quality Data:
```python
# Keep only rows with high-quality volume MAs
df_reliable = df[
    (df['volume_ma_buy_15_quality'] >= 0.6) &
    (df['volume_ma_sell_15_quality'] >= 0.6)
]
```

### Using Quality as a Feature:
```python
# Let the model learn which quality levels are reliable
features = [
    'spread_ma_15',
    'volume_ma_buy_15',
    'volume_ma_buy_15_quality',  # Quality as feature
    'volume_ma_sell_15_quality',
    ...
]
```

### Weighted Averaging:
```python
# Weight MAs by their quality
df['weighted_volume_ma'] = (
    df['volume_ma_buy_15'] * df['volume_ma_buy_15_quality']
)
```

---

## Conclusion

The exchange mixing problem is real but manageable. For your project:

1. **Start simple**: Accept current behavior for initial analysis
2. **Add quality tracking**: Implement Solution 3 to understand data quality
3. **Filter if needed**: Remove low-quality rows or use quality as a feature
4. **Advanced (optional)**: If exchange-specific behavior matters, implement Solution 5

The most important thing is **transparency** - know what your rolling calculations are doing and track their quality.
