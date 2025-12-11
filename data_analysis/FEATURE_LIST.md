# Feature Engineering Roadmap

## Overview
This document lists all features to engineer for arbitrage analysis. Implement them one section at a time.

---

## **SECTION 1: Basic Spread Features (PRIORITY 1)**

### 1.1 Close Price Spreads
| Feature Name | Description | Formula |
|--------------|-------------|---------|
| `min_close` | Lowest close price across all exchanges | `min(BINANCE:close, BITFINEX:close, ...)` |
| `max_close` | Highest close price across all exchanges | `max(BINANCE:close, BITFINEX:close, ...)` |
| `spread_close_absolute` | Absolute price difference | `max_close - min_close` |
| `spread_close_pct` | Spread as percentage | `(max_close - min_close) / min_close * 100` |
| `buy_exchange` | Exchange with lowest close price (where to buy) | Exchange name of `min_close` |
| `sell_exchange` | Exchange with highest close price (where to sell) | Exchange name of `max_close` |

**Why:** Core arbitrage opportunity calculation - shows where to buy/sell and profit margin.

---

### 1.2 Volume Features
| Feature Name | Description | Formula |
|--------------|-------------|---------|
| `volume_buy_exchange` | Trading volume on the buy exchange | Volume from `buy_exchange` |
| `volume_sell_exchange` | Trading volume on the sell exchange | Volume from `sell_exchange` |
| `min_volume` | Minimum volume between both exchanges | `min(volume_buy, volume_sell)` |
| `volume_ratio` | Liquidity comparison | `volume_sell / volume_buy` |

**Why:** Check if you can actually execute the trade - high spread with low volume is useless.

---

### 1.3 High-Low Theoretical Maximum (Optional Context)
| Feature Name | Description | Formula |
|--------------|-------------|---------|
| `min_low` | Lowest LOW across all exchanges | `min(BINANCE:low, BITFINEX:low, ...)` |
| `max_high` | Highest HIGH across all exchanges | `max(BINANCE:high, BITFINEX:high, ...)` |
| `spread_highlow_pct` | Theoretical max spread | `(max_high - min_low) / min_low * 100` |
| `opportunity_gap` | What you're missing | `spread_highlow_pct - spread_close_pct` |

**Why:** Shows theoretical maximum profit if timing was perfect (for comparison only).

---

## **SECTION 2: Time-Based Features (PRIORITY 1)**

| Feature Name | Description | Formula |
|--------------|-------------|---------|
| `hour` | Hour of day (0-23) | `time.dt.hour` |
| `minute` | Minute within hour (0-59) | `time.dt.minute` |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) | `time.dt.dayofweek` |
| `is_weekend` | Weekend flag | `day_of_week >= 5` |
| `is_trading_hours` | Peak trading hours (8am-5pm UTC) | `8 <= hour <= 17` |

**Why:** Arbitrage opportunities may be more common at certain times (e.g., overlap of Asian/US markets).

---

## **SECTION 3: Volatility Features (PRIORITY 2)**

### 3.1 Intra-Minute Volatility
| Feature Name | Description | Formula |
|--------------|-------------|---------|
| `volatility_BINANCE` | Volatility on Binance | `(BINANCE:high - BINANCE:low) / BINANCE:close * 100` |
| `volatility_BITFINEX` | Volatility on Bitfinex | `(BITFINEX:high - BITFINEX:low) / BITFINEX:close * 100` |
| `volatility_COINBASE` | Volatility on Coinbase | `(COINBASE:high - COINBASE:low) / COINBASE:close * 100` |
| `volatility_avg` | Average volatility across exchanges | `mean(volatility_BINANCE, volatility_BITFINEX, ...)` |
| `volatility_max` | Maximum volatility | `max(volatility_BINANCE, volatility_BITFINEX, ...)` |

**Why:** High volatility = more risk but also more opportunity. Shows market stability.

---

### 3.2 Price Position in Range
| Feature Name | Description | Formula |
|--------------|-------------|---------|
| `price_position_buy_exchange` | Where close is in the minute's range (buy side) | `(close - low) / (high - low)` on buy exchange |
| `price_position_sell_exchange` | Where close is in the minute's range (sell side) | `(close - low) / (high - low)` on sell exchange |

**Why:** 
- If close is near HIGH (value ~1.0) = price trending up, bad time to buy
- If close is near LOW (value ~0.0) = price trending down, bad time to sell

---

## **SECTION 4: Price Change Features (PRIORITY 2)**

| Feature Name | Description | Formula |
|--------------|-------------|---------|
| `price_change_BINANCE` | Price change within minute (Binance) | `(BINANCE:close - BINANCE:open) / BINANCE:open * 100` |
| `price_change_BITFINEX` | Price change within minute (Bitfinex) | `(BITFINEX:close - BITFINEX:open) / BITFINEX:open * 100` |
| `price_change_COINBASE` | Price change within minute (Coinbase) | `(COINBASE:close - COINBASE:open) / COINBASE:open * 100` |
| `price_change_buy_exchange` | Momentum on buy exchange | Price change on `buy_exchange` |
| `price_change_sell_exchange` | Momentum on sell exchange | Price change on `sell_exchange` |

**Why:** Shows momentum - if buy exchange is trending up, might want to wait.

---

## **SECTION 5: Moving Averages (PRIORITY 2)**

### 5.1 Spread Moving Averages
| Feature Name | Description | Formula | Window |
|--------------|-------------|---------|--------|
| `spread_ma_5` | 5-minute moving average of spread | `spread_close_pct.rolling(5).mean()` | 5 min |
| `spread_ma_15` | 15-minute moving average | `spread_close_pct.rolling(15).mean()` | 15 min |
| `spread_ma_30` | 30-minute moving average | `spread_close_pct.rolling(30).mean()` | 30 min |

**Why:** 
- If `spread > spread_ma_5` → spread is widening (good entry signal)
- If `spread < spread_ma_5` → spread is closing (exit signal)

---

### 5.2 Volume Moving Averages
| Feature Name | Description | Formula | Window |
|--------------|-------------|---------|--------|
| `volume_ma_10_buy` | 10-minute avg volume on buy exchange | `volume_buy_exchange.rolling(10).mean()` | 10 min |
| `volume_ma_10_sell` | 10-minute avg volume on sell exchange | `volume_sell_exchange.rolling(10).mean()` | 10 min |

**Why:** Check if current volume is normal - low volume spike might be unreliable.

---

### 5.3 Exponential Moving Average (EMA)
| Feature Name | Description | Formula | Span |
|--------------|-------------|---------|------|
| `spread_ema_5` | 5-minute EMA of spread (more reactive) | `spread_close_pct.ewm(span=5).mean()` | 5 min |
| `spread_ema_15` | 15-minute EMA | `spread_close_pct.ewm(span=15).mean()` | 15 min |

**Why:** EMA reacts faster to recent changes than SMA - better for fast-moving markets.

---

## **SECTION 6: Bollinger Bands (PRIORITY 3)**

| Feature Name | Description | Formula | Window |
|--------------|-------------|---------|--------|
| `spread_bb_ma` | Bollinger Band middle line | `spread_close_pct.rolling(20).mean()` | 20 min |
| `spread_bb_std` | Standard deviation | `spread_close_pct.rolling(20).std()` | 20 min |
| `spread_bb_upper` | Upper band (2 std devs above) | `spread_bb_ma + (2 * spread_bb_std)` | 20 min |
| `spread_bb_lower` | Lower band (2 std devs below) | `spread_bb_ma - (2 * spread_bb_std)` | 20 min |
| `spread_bb_position` | Where spread is in the bands | `(spread - spread_bb_lower) / (spread_bb_upper - spread_bb_lower)` | 20 min |

**Why:** 
- If spread hits upper band → extreme opportunity (mean reversion expected)
- If spread is near middle → typical market condition

---

## **SECTION 7: Rolling Statistics (PRIORITY 3)**

| Feature Name | Description | Formula | Window |
|--------------|-------------|---------|--------|
| `spread_rolling_std_10` | Rolling standard deviation of spread | `spread_close_pct.rolling(10).std()` | 10 min |
| `spread_rolling_min_10` | Minimum spread in last 10 minutes | `spread_close_pct.rolling(10).min()` | 10 min |
| `spread_rolling_max_10` | Maximum spread in last 10 minutes | `spread_close_pct.rolling(10).max()` | 10 min |

**Why:** Shows if spread is stable or jumping around - high std = unpredictable.

---

## **SECTION 8: Change & Rate Features (PRIORITY 3)**

| Feature Name | Description | Formula |
|--------------|-------------|---------|
| `spread_change` | Change in spread from previous minute | `spread_close_pct - spread_close_pct.shift(1)` |
| `spread_change_pct` | Percentage change in spread | `(spread_close_pct - spread_close_pct.shift(1)) / spread_close_pct.shift(1) * 100` |
| `spread_acceleration` | Rate of change of spread | `spread_change - spread_change.shift(1)` |

**Why:** 
- Positive change → spread widening (opportunity growing)
- Negative change → spread closing (opportunity disappearing)

---

## **SECTION 9: Cross-Exchange Price Ratios (PRIORITY 3)**

| Feature Name | Description | Formula |
|--------------|-------------|---------|
| `price_ratio_buy_sell` | Price ratio between buy and sell exchanges | `buy_exchange:close / sell_exchange:close` |
| `price_diff_buy_sell` | Absolute price difference | `sell_exchange:close - buy_exchange:close` |

**Why:** Alternative way to measure spread - sometimes ratios reveal patterns better than absolute differences.

---

## **SECTION 10: Lag Features (FOR ML - PRIORITY 4)**

| Feature Name | Description | Formula | Lag |
|--------------|-------------|---------|-----|
| `spread_lag_1` | Spread 1 minute ago | `spread_close_pct.shift(1)` | 1 min |
| `spread_lag_5` | Spread 5 minutes ago | `spread_close_pct.shift(5)` | 5 min |
| `spread_lag_10` | Spread 10 minutes ago | `spread_close_pct.shift(10)` | 10 min |
| `volume_lag_1` | Volume 1 minute ago | `min_volume.shift(1)` | 1 min |
| `buy_exchange_lag_1` | Previous buy exchange (categorical) | `buy_exchange.shift(1)` | 1 min |

**Why:** For ML models to predict future spreads based on historical patterns.

---

## **SECTION 11: Opportunity Flags (PRIORITY 1)**

| Feature Name | Description | Formula | Threshold |
|--------------|-------------|---------|-----------|
| `is_opportunity` | Spread exceeds trading costs | `spread_close_pct >= 0.50` | 0.50% |
| `is_real_opportunity` | Spread exceeds costs + safety margin | `spread_close_pct >= 0.60` | 0.60% |
| `opportunity_quality` | Categorical quality rating | Based on spread + volume + volatility | Low/Medium/High |

**Why:** Quick filters for identifying tradeable opportunities.

---

## **Implementation Order:**

### **Phase 1 (Do First):**
1. Section 1: Basic Spread Features
2. Section 2: Time Features
3. Section 11: Opportunity Flags

### **Phase 2 (Do Second):**
4. Section 3: Volatility Features
5. Section 4: Price Change Features
6. Section 5: Moving Averages (5.1 only)

### **Phase 3 (Do Later):**
7. Section 5: Moving Averages (5.2, 5.3)
8. Section 6: Bollinger Bands
9. Section 7: Rolling Statistics

### **Phase 4 (For ML):**
10. Section 8: Change Features
11. Section 9: Price Ratios
12. Section 10: Lag Features

---

## **Notes:**

- ✅ **Start with Phase 1** - gives you the core analysis
- ✅ **Test each section** before moving to next
- ✅ **NaN values** will appear in rolling/lag features for first N rows - this is normal
- ✅ **Apply same features** to BTC, ETH, and DOGE DataFrames
- ✅ **Keep original columns** - don't overwrite OHLCV data