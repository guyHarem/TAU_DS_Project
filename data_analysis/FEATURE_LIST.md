# Feature Engineering Documentation

## Overview
This document provides comprehensive documentation for all features engineered for cryptocurrency arbitrage analysis. Features are organized by dependency layers where each layer builds upon the previous ones.

## Layer System
- **Layer 1 (L1)**: Raw data from exchanges (BINANCE:close, COINBASE:high, etc.)
- **Layer 2 (L2)**: Features created directly from L1 raw data
- **Layer 3 (L3)**: Features created from L2 and/or L1  
- **Layer 4 (L4)**: Features created from L3 and/or L2 and/or L1

---

## **LAYER 2: CORE SPREAD FEATURES** (`add_close_spread`)

### Features Created:
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `min_close` | Lowest close price across all exchanges | `min(BINANCE:close, BITFINEX:close, ...)` | ALL (skipna=True) |
| `max_close` | Highest close price across all exchanges | `max(BINANCE:close, BITFINEX:close, ...)` | ALL (skipna=True) |
| `spread_close_absolute` | Absolute price difference | `max_close - min_close` | Derived |
| `spread_close_pct` | Spread as percentage | `(max_close - min_close) / min_close * 100` | Derived |
| `buy_exchange` | Exchange with lowest close price (where to buy) | Exchange name of `min_close` | ALL |
| `sell_exchange` | Exchange with highest close price (where to sell) | Exchange name of `max_close` | ALL |
| `is_opportunity` | Basic opportunity flag | `spread_close_pct >= 0.5%` | Derived |
| `is_real_opportunity` | Conservative opportunity flag | `spread_close_pct >= 0.6%` | Derived |
| `num_exchanges_available` | Data quality indicator | `count(non-null close prices)` | ALL |

**Input Features:** All exchange close prices (`BINANCE:close`, `BITFINEX:close`, etc.)

**Why Important for Analysis:**
- Core arbitrage opportunity identification - shows where to buy/sell and profit margin
- Data quality tracking through `num_exchanges_available`

**Why Important for Model:**
- `spread_close_pct` is the primary target for prediction
- `is_real_opportunity` is the classification target
- `buy_exchange` and `sell_exchange` identify the trading pair

**Missing Data Behavior:** Uses `skipna=True` - if only 3 of 6 exchanges have data, finds min/max among those 3


---

## **LAYER 3: VOLUME FEATURES** (`add_volume_features`)

### Features Created:
| Feature Name | Description | Formula | Layer Dependencies |
|--------------|-------------|---------|-------------------|
| `volume_buy_exchange` | Trading volume on the buy exchange | Volume from `buy_exchange` | L2: buy_exchange, L1: volume |
| `volume_sell_exchange` | Trading volume on the sell exchange | Volume from `sell_exchange` | L2: sell_exchange, L1: volume |
| `min_volume` | Minimum volume between both exchanges | `min(volume_buy, volume_sell)` | L3 (derived) |
| `volume_ratio` | Liquidity comparison | `volume_sell / volume_buy` | L3 (derived) |

**Input Features:** L2: buy_exchange, sell_exchange | L1: all exchange volume columns

**Why Important for Analysis:**
- Verifies if arbitrage opportunity is executable (high spread with low volume is useless)
- `min_volume` represents maximum tradeable size for the opportunity

**Why Important for Model:**
- Predicts opportunity quality (spread + volume = actionable signal)
- Helps filter false positives (spreads that look good but can't be traded)

**Missing Data Behavior:** If buy or sell exchange volume is NaN, all features become NaN. If `buy_exchange == sell_exchange`, `volume_ratio = 1.0` (invalid for arbitrage).

---

## **LAYER 2: HIGH-LOW SPREAD FEATURES** (`add_high_low_spread`)

### Features Created:
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `min_low` | Lowest LOW across all exchanges | `min(BINANCE:low, BITFINEX:low, ...)` | ALL (skipna=True) |
| `max_high` | Highest HIGH across all exchanges | `max(BINANCE:high, BITFINEX:high, ...)` | ALL (skipna=True) |
| `high_exchange` | Exchange with highest high | Exchange name of `max_high` | ALL |
| `low_exchange` | Exchange with lowest low | Exchange name of `min_low` | ALL |
| `spread_highlow_absolute` | Theoretical max absolute spread | `max_high - min_low` | Derived |
| `spread_highlow_pct` | Theoretical max spread percentage | `(max_high - min_low) / min_low * 100` | Derived |
| `opportunity_gap` | Unrealized potential | `spread_highlow_pct - spread_close_pct` | Derived |

**Input Features:** All exchange high and low prices

**Why Important for Analysis:**
- Shows theoretical maximum profit if timing was perfect (comparison baseline)
- `opportunity_gap` indicates how much potential was missed due to timing

**Why Important for Model:**
- Indicates intra-minute price volatility
- Large gap suggests unstable spreads (higher risk)
- Can help model learn when spreads are sustainable vs. fleeting

**Missing Data Behavior:** Uses `skipna=True` - works with partial exchange data

---

## **LAYER 4: TIME FEATURES** (`add_time_features`)

### Features Created:
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `hour` | Hour of day (0-23) | `time.dt.hour` | None (time-based) |
| `minute` | Minute within hour (0-59) | `time.dt.minute` | None (time-based) |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) | `time.dt.dayofweek` | None (time-based) |
| `is_weekend` | Weekend flag | `1 if day_of_week >= 5 else 0` | None (time-based) |
| `overlap_hours` | Peak trading hours flag | `1 if 19 <= hour <= 21 else 0` | None (time-based) |

**Input Features:** `time` column only

**Why Important for Analysis:**
- Identifies temporal patterns in arbitrage opportunities
- Discovers if certain hours/days have more opportunities
- Detects market microstructure effects (e.g., US-Asian market overlap)

**Why Important for Model:**
- Cyclical features help model learn time-based patterns
- Different market hours may have different liquidity/volatility profiles
- Enables time-stratified predictions

**Missing Data Behavior:** No impact from missing exchange data - time is always available

---

## **LAYER 5: VOLATILITY FEATURES** (`add_volatility_features`)

### Features Created:
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `{EXCHANGE}_volatility` | Intra-minute volatility per exchange | `(high - low) / close * 100` | Each exchange individually |
| `volatility_avg` | Average volatility across all exchanges | `mean(all exchange volatilities)` | ALL (mean of available) |
| `volatility_max` | Maximum volatility | `max(all exchange volatilities)` | ALL |
| `volatility_min` | Minimum volatility | `min(all exchange volatilities)` | ALL |
| `price_position_buy_exchange` | Buy price position in its range | `(close - low) / (high - low)` on buy exchange | BUY only |
| `price_position_sell_exchange` | Sell price position in its range | `(close - low) / (high - low)` on sell exchange | SELL only |

**Input Features:** All exchange high, low, close prices; `buy_exchange`, `sell_exchange`

**Why Important for Analysis:**
- Measures market stability and risk
- High volatility = spread may disappear quickly
- Price position shows intra-minute momentum direction

**Why Important for Model:**
- **Volatility**: Indicates opportunity risk - high volatility opportunities may be unreliable
- **Price Position**: 
  - Buy exchange near high (→1.0) = price rebounding, bad buy signal
  - Buy exchange near low (→0.0) = price at bottom, good buy signal
  - Sell exchange near high (→1.0) = good sell signal
  - Helps model predict spread persistence

**Missing Data Behavior:** 
- Individual volatility only calculated for exchanges with all OHLC data
- Aggregates use available exchanges
- Price positions are NaN if buy/sell exchange data incomplete or high==low (division by zero)

---

## **LAYER 6: PRICE CHANGE FEATURES** (`add_price_change_features`)

### Features Created:
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `{EXCHANGE}_price_change` | Intra-minute price change per exchange | `(close - open) / open * 100` | Each exchange individually |
| `price_change_buy_exchange` | Momentum on buy exchange | Price change at `buy_exchange` | BUY only |
| `price_change_sell_exchange` | Momentum on sell exchange | Price change at `sell_exchange` | SELL only |

**Input Features:** All exchange close and open prices; `buy_exchange`, `sell_exchange`

**Why Important for Analysis:**
- Shows directional momentum within the minute
- Helps identify if opportunity is expanding or contracting
- Detects divergence between exchanges (one rising, one falling)

**Why Important for Model:**
- **Per-Exchange Changes**: Captures market-wide trends
- **Buy/Sell Specific**: 
  - Positive buy exchange change = price rising, might reduce opportunity
  - Negative sell exchange change = price falling, might reduce opportunity
  - Divergent changes create/destroy spreads
- Model learns which price change patterns predict opportunities

**Missing Data Behavior:** Only calculated for exchanges with both open and close data. Some duplication (e.g., if COINBASE is buy exchange, both `COINBASE_price_change` and `price_change_buy_exchange` have same value).

---

## **LAYER 7: MOVING AVERAGES** (`add_moving_averages`)

### Features Created (windows=[5,15,30]):
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `spread_ma_{window}` | Simple moving average of spread | `spread_close_pct.rolling(window).mean()` | None (uses spread) |
| `volume_ma_buy_{window}` | MA of buy exchange volume | `volume_buy_exchange.rolling(window).mean()` | BUY only |
| `volume_ma_sell_{window}` | MA of sell exchange volume | `volume_sell_exchange.rolling(window).mean()` | SELL only |
| `spread_ema_{window}` | Exponential MA of spread | `spread_close_pct.ewm(span=window).mean()` | None (uses spread) |

**Input Features:** `spread_close_pct`, `volume_buy_exchange`, `volume_sell_exchange`

**Why Important for Analysis:**
- Identifies trending vs. mean-reverting spreads
- `spread > spread_ma` = widening spread (entry signal)
- `spread < spread_ma` = narrowing spread (exit signal)
- Volume MAs detect unusual volume spikes

**Why Important for Model:**
- Trend-following features: help model detect momentum
- EMA reacts faster than SMA - captures rapid changes
- Deviation from MA can trigger predictions
- Volume MAs help assess opportunity reliability

**Missing Data Behavior:** 
- First `window` rows are NaN
- **CRITICAL ISSUE**: Mixes different exchanges if buy/sell exchange changes between minutes (see ROLLING_STATS_ISSUE.md)
- Volume MAs may combine volumes from different exchanges over time

---

## **LAYER 8: BOLLINGER BANDS** (`add_bollinger_bands`)

### Features Created (windows=[5,15,30], num_std=2):
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `spread_bb_ma_{window}` | BB middle line (MA) | `spread_close_pct.rolling(window).mean()` | None (uses spread) |
| `spread_bb_std_{window}` | Standard deviation | `spread_close_pct.rolling(window).std()` | None (uses spread) |
| `spread_bb_upper_{window}` | Upper band | `spread_bb_ma + (2 * spread_bb_std)` | Derived |
| `spread_bb_lower_{window}` | Lower band | `spread_bb_ma - (2 * spread_bb_std)` | Derived |
| `spread_bb_position_{window}` | Normalized position in bands | `(spread - lower) / (upper - lower)` | Derived |

**Input Features:** `spread_close_pct`

**Why Important for Analysis:**
- Detects statistical extremes (spread at/beyond bands)
- Spread at upper band = unusually high (mean reversion expected)
- Spread at lower band = unusually low (no opportunity expected)
- Band width shows volatility (wide bands = volatile market)

**Why Important for Model:**
- **BB Position** normalizes spread relative to recent history (0-1 scale):
  - 0.0 = at lower band (low spread)
  - 0.5 = at middle/average
  - 1.0 = at upper band (high spread)
  - >1.0 = broke above (extreme opportunity)
- Helps model detect when spreads are abnormal vs. normal
- Mean reversion signals for opportunity timing

**Missing Data Behavior:** 
- First `window` rows are NaN
- Position is NaN if bands are identical (zero std dev)
- **NOTE**: `spread_bb_ma` duplicates `spread_ma` from Layer 7 (redundant calculation)

---

## **LAYER 9: ROLLING STATISTICS** (`add_rolling_stats`)

### Features Created (windows=[5,15,30]):
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `spread_rolling_std_{window}` | Rolling standard deviation | `spread_close_pct.rolling(window).std()` | None (uses spread) |
| `spread_rolling_max_{window}` | Maximum spread in window | `spread_close_pct.rolling(window).max()` | None (uses spread) |
| `spread_rolling_min_{window}` | Minimum spread in window | `spread_close_pct.rolling(window).min()` | None (uses spread) |
| `volume_buy_rolling_std_{window}` | Volume volatility (buy) | `volume_buy_exchange.rolling(window).std()` | BUY only |
| `volume_sell_rolling_std_{window}` | Volume volatility (sell) | `volume_sell_exchange.rolling(window).std()` | SELL only |
| `opportunities_in_last_{window}` | Count of basic opportunities | `is_opportunity.rolling(window).sum()` | None (uses flag) |
| `real_opportunities_in_last_{window}` | Count of real opportunities | `is_real_opportunity.rolling(window).sum()` | None (uses flag) |
| `spread_range_{window}` | Spread range in window | `rolling_max - rolling_min` | Derived |
| `spread_zscore_{window}` | Statistical z-score | `(spread - rolling_mean) / rolling_std` | Derived |

**Input Features:** `spread_close_pct`, `volume_buy_exchange`, `volume_sell_exchange`, `is_opportunity`, `is_real_opportunity`

**Why Important for Analysis:**
- **Spread Stats**: Measure spread stability/variability
- **Opportunity Counts**: Detect opportunity clustering (many opportunities recently = hot market)
- **Spread Range**: Shows if market is tight or wide
- **Z-Score**: Identifies statistical anomalies

**Why Important for Model:**
- **Spread Std/Range**: Indicates predictability (low std = stable, high std = chaotic)
- **Z-Score**: Highlights unusual spreads:
  - Z-score > 2 = spread is 2 std devs above average (very unusual, might revert)
  - Z-score < -2 = spread is very low
  - Model learns which z-score levels are sustainable vs. transient
- **Opportunity Counts**: Auto-correlation signal (if many recent opportunities, next minute may also have one)
- **Volume Std**: Measures volume consistency (stable volume = more reliable opportunity)

**Missing Data Behavior:**
- First `window` rows are NaN
- Z-score is NaN if std = 0 (division by zero protection)
- **CRITICAL ISSUE**: Same as Layer 7 - mixes data from different exchanges (see ROLLING_STATS_ISSUE.md)

---

## **LAYER 10: RATE CHANGE FEATURES** (`add_rate_change_features`)

### Features Created:
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `spread_rate_change` | 1st derivative of spread | `spread_close_pct - spread_close_pct.shift(1)` | None (uses spread) |
| `spread_rate_change_pct` | Percentage rate change | `spread_rate_change / spread_close_pct.shift(1) * 100` | None (uses spread) |
| `spread_rate_acceleration` | 2nd derivative of spread | `spread_rate_change - spread_rate_change.shift(1)` | None (uses spread) |

**Input Features:** `spread_close_pct`

**Why Important for Analysis:**
- **Rate Change**: Shows if spread is widening or narrowing
  - Positive = spread growing (opportunity emerging/expanding)
  - Negative = spread shrinking (opportunity closing)
- **Acceleration**: Shows if growth is accelerating or decelerating

**Why Important for Model:**
- **Momentum features**: Helps model predict near-term spread direction
- Positive acceleration = spread expanding rapidly (good entry signal)
- Negative acceleration = spread contracting (exit signal)
- Model learns velocity patterns that precede opportunities

**Missing Data Behavior:** 
- First row is NaN (no previous data)
- Rate change pct is NaN if previous spread = 0 (division by zero protection)
- Acceleration is NaN for first 2 rows

---

## **LAYER 11: CROSS-EXCHANGE PRICE RATIOS** (`add_cross_ex_price_ratio`)

### Features Created:
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `price_ratio_buy_sell` | Price ratio between trading pair | `sell_exchange:close / buy_exchange:close` | BUY + SELL |
| `price_ratio_{EX1}_{EX2}` | Ratio for each exchange pair | `EX2:close / EX1:close` | ALL pairs (15 combinations) |
| `avg_price_ratio` | Average ratio across all pairs | `mean(all price ratios)` | ALL |
| `max_price_ratio` | Maximum ratio (most extreme) | `max(all price ratios)` | ALL |
| `min_price_ratio` | Minimum ratio (tightest pair) | `min(all price ratios)` | ALL |
| `price_ratio_std` | Price dispersion | `std(all price ratios)` | ALL |

**Input Features:** All exchange close prices, `buy_exchange`, `sell_exchange`

**Why Important for Analysis:**
- **Ratios vs. Spreads**: Ratios are price-normalized (0.5% on BTC or ETH = same ratio, different $ spread)
- **Price Dispersion** (`price_ratio_std`): 
  - High std = fragmented market, exchanges pricing differently
  - Low std = efficient market, tight pricing

**Why Important for Model:**
- **`price_ratio_buy_sell`**: Alternative representation of spread (multiplicative vs. additive)
- **Per-Pair Ratios**: Captures all exchange relationships, not just buy/sell
- **Aggregates**:
  - `price_ratio_std` high → predicts more opportunities (market inefficiency)
  - `price_ratio_std` low → predicts fewer opportunities (tight market)
  - Model learns market structure signals
- Ratios can reveal patterns that absolute spreads miss

**Missing Data Behavior:** 
- Only creates ratios for exchange pairs where both have close data
- Aggregates use available ratios only
- More exchange coverage = more ratio features

---

## **LAYER 12: LAG FEATURES** (`add_lag_features`)

### Features Created (lags=[1,5,10,30]):
| Feature Name | Description | Formula | Uses Exchanges |
|--------------|-------------|---------|----------------|
| `spread_lag_{lag}` | Historical spread | `spread_close_pct.shift(lag)` | None (uses spread) |
| `volume_buy_lag_{lag}` | Historical buy volume | `volume_buy_exchange.shift(lag)` | BUY only (lagged) |
| `volume_sell_lag_{lag}` | Historical sell volume | `volume_sell_exchange.shift(lag)` | SELL only (lagged) |
| `min_volume_lag_{lag}` | Historical min volume | `min_volume.shift(lag)` | Derived (lagged) |
| `is_opportunity_lag_{lag}` | Historical opportunity flag | `is_opportunity.shift(lag)` | None (lagged flag) |
| `is_real_opportunity_lag_{lag}` | Historical real opportunity flag | `is_real_opportunity.shift(lag)` | None (lagged flag) |
| `price_change_buy_lag_{lag}` | Historical buy price change | `price_change_buy_exchange.shift(lag)` | BUY only (lagged) |
| `price_change_sell_lag_{lag}` | Historical sell price change | `price_change_sell_exchange.shift(lag)` | SELL only (lagged) |
| `volatility_avg_lag_{lag}` | Historical volatility | `volatility_avg.shift(lag)` | None (lagged) |
| `buy_exchange_lag_1` | Previous buy exchange | `buy_exchange.shift(1)` | Categorical |
| `sell_exchange_lag_1` | Previous sell exchange | `sell_exchange.shift(1)` | Categorical |
| `spread_diff_from_lag_1` | Spread change from 1 min ago | `spread_close_pct - spread_lag_1` | Derived |
| `spread_diff_from_lag_5` | Spread change from 5 min ago | `spread_close_pct - spread_lag_5` | Derived |
| `volume_diff_from_lag_1` | Volume change from 1 min ago | `min_volume - min_volume_lag_1` | Derived |

**Input Features:** Most features from previous layers

**Why Important for Analysis:**
- Auto-correlation analysis (do opportunities cluster?)
- Persistence patterns (if opportunity 5 min ago, likely now?)
- Exchange stickiness (same exchange remains cheapest?)

**Why Important for Model:**
- **Time-series prediction**: Past values predict future values
- **Lag features** are critical for ML models to learn temporal patterns:
  - `spread_lag_1` = immediate history
  - `spread_lag_30` = longer-term trend
- **Opportunity lags**: 
  - If `is_real_opportunity_lag_1 = 1`, high probability current row also = 1 (persistence)
  - Model learns opportunity duration patterns
- **Diff features**: Momentum/change signals
- **Exchange lags**: Model learns if certain exchanges stay cheap/expensive

**Missing Data Behavior:**
- First `lag` rows are NaN (no historical data)
- If source data was NaN, lag will also be NaN
- **CRITICAL**: Lagged volume/price_change features reference specific exchanges that may differ from current buy/sell exchange

---

## **SUMMARY TABLE: Feature Layer Overview**

| Actual Layer | Function | Dependencies | # Features | Uses All Exchanges | Uses Buy/Sell Only |
|--------------|----------|--------------|------------|-------------------|-------------------|
| **L2** | `add_close_spread` | L1: close prices | 9 | ✓ (min/max) | ✓ (identify) |
| **L2** | `add_high_low_spread` | L1: high/low prices | 7 | ✓ | ✗ |
| **L2** | `add_time_features` | L1: time | 5 | ✗ | ✗ |
| **L3** | `add_volume_features` | L2: buy/sell_exchange, L1: volume | 4 | ✗ | ✓ |
| **L3** | `add_volatility_features` | L2: buy/sell_exchange, L1: OHLC | 10+ | ✓ (aggregates) | ✓ (position) |
| **L3** | `add_price_change_features` | L2: buy/sell_exchange, L1: open/close | 8+ | ✓ (per-exchange) | ✓ (specific) |
| **L3** | `add_bollinger_bands` | L2: spread_close_pct | 15 (5x3 windows) | ✗ | ✗ |
| **L3** | `add_rate_change_features` | L2: spread_close_pct | 3 | ✗ | ✗ |
| **L3** | `add_cross_ex_price_ratio` | L2: buy/sell_exchange, L1: close | 20+ (15 pairs + aggs) | ✓ | ✓ (main ratio) |
| **L4** | `add_moving_averages` | L3: volume_buy/sell, L2: spread_close_pct | 12 (4x3 windows) | ✗ | ✓ (volume) |
| **L4** | `add_rolling_stats` | L3: volume, L2: spread/oppty | 27+ (varies) | ✗ | ✓ (volume std) |
| **L4** | `add_lag_features` | L3: volume/price_change/volatility, L2: spread/buy_sell/oppty | 35+ (varies by lags) | ✗ | ✓ (volume/price) |

**Total Features Created: ~160+** (depending on window/lag parameters)

### **Layer Dependencies Summary:**
- **L1 (Raw Data)**: BINANCE:close, COINBASE:high, etc. - the original exchange OHLCV data
- **L2 (First Derived)**: Features created directly from L1 (spread, time features)
- **L3 (Second Derived)**: Features using L2 features (like buy_exchange) plus L1 data
- **L4 (Third Derived)**: Features using L3 features (like volume_buy_exchange) plus L2/L1

---

## **CRITICAL ISSUES & CONSIDERATIONS**

### **Missing Data Impact:**
1. **Functions using ALL exchanges** (add_close_spread, add_high_low_spread, volatility aggregates, price ratios): Work with partial data via `skipna=True`
2. **Functions using BUY/SELL only** (volume, price_change_specific, moving averages, rolling stats, lags): Become NaN if specific exchange data missing
3. **Data quality**: Check `num_exchanges_available` - fewer exchanges = less reliable spread identification

### **Rolling Features Problem:**
- L4 functions (add_moving_averages, add_rolling_stats): **Mix data from different exchanges** when buy/sell exchange changes between minutes
- See `ROLLING_STATS_ISSUE.md` for detailed explanation and solutions

### **Redundant Features:**
- `spread_bb_ma_{window}` (L3) duplicates `spread_ma_{window}` (L4) - consider removing from one
- Per-exchange price changes duplicate buy/sell specific ones (intentional for different model uses)

### **Model Feature Importance Hierarchy:**
1. **Critical**: L2 spread and time features (core arbitrage metrics)
2. **High Value**: L3 volume, volatility, price change (opportunity quality)
3. **Medium Value**: Layers 7, 9, 12 (moving averages, rolling stats, lags)
4. **Advanced**: Layers 8, 11 (Bollinger bands, price ratios)

---

## **Implementation Notes**

- ✅ **All features applied** to all cryptocurrency DataFrames (BTC, ETH, DOGE, LINK, SOL, XRP)
- ✅ **Original OHLCV data preserved** - no overwriting
- ✅ **NaN handling**: Rolling/lag features have NaN for first N rows (expected behavior)
- ✅ **Feature selection**: Not all features may be used in final model (correlation analysis needed)
- ⚠️ **Data quality**: Consider filtering rows where `num_exchanges_available < 3`
- ⚠️ **Rolling features**: Be aware of exchange-mixing issue (see ROLLING_STATS_ISSUE.md)

