# Feature Engineering Documentation

## Overview

This document provides comprehensive documentation for all features engineered by `data_featuring.py`. Features are organized in 5 dependency layers, each building on previous layers.

## Layer System

- **Layer 1 (L1)**: Raw exchange data (BINANCE:close, COINBASE:high, etc.)
- **Layer 2**: Core features created directly from L1 (spreads, time, exchanges)
- **Layer 3**: Volume, volatility, and price metrics from L2 + L1
- **Layer 4**: Advanced metrics using L3 + L2 (moving averages, rolling stats, z-scores)
- **Layer 5**: Temporal features using previous layers (Bollinger bands, lags)

---

## LAYER 2: CORE FEATURES

Function: `layer2(df)` in data_featuring.py

### Minmax Features (`add_L2_minmax_features`)

| Feature | Type | Formula | Uses |
|---------|------|---------|------|
| `min_close` | numeric | `min(all exchange closes)` | ALL exchanges |
| `max_close` | numeric | `max(all exchange closes)` | ALL exchanges |
| `buy_exchange` | categorical | Exchange with `min_close` | Identifies |
| `sell_exchange` | categorical | Exchange with `max_close` | Identifies |

**Purpose**: Identifies the basic arbitrage opportunity (lowest buy price, highest sell price)

**Example**: If BTC close is $50,000 on Binance and $50,100 on Coinbase, then:
- `min_close = 50000, max_close = 50100, buy_exchange = BINANCE, sell_exchange = COINBASE`

### Exchange-Specific Features (`add_L2_exchange_features`)

Per-exchange features (created for each exchange):

| Feature | Type | Formula |
|---------|------|---------|
| `{EXCHANGE}_volatility` | numeric | `(high - low) / close * 100` |
| `{EXCHANGE}_price_change` | numeric | `(close - open) / open * 100` |
| `num_exchanges_available` | numeric | Count of exchanges with close price |

**Purpose**: Volatility shows risk, price change shows momentum, data availability shows quality

### Time Features (`add_L2_time_features`)

| Feature | Type | Values |
|---------|------|--------|
| `hour` | numeric | 0-23 |
| `minute` | numeric | 0-59 |
| `day_of_week` | numeric | 0-6 (Mon-Sun) |
| `is_weekend` | binary | {0, 1} |
| `overlap_hours` | binary | {0, 1} for 19:00-21:00 UTC |

**Purpose**: Enable temporal pattern analysis

### Cross-Exchange Price Ratios (`add_L2_cross_exchange_price_ratio`)

Pairwise ratios: `price_ratio_{EX1}_{EX2} = close_EX2 / close_EX1`

Aggregates: `avg_price_ratio, max_price_ratio, min_price_ratio, price_ratio_std`

**Purpose**: Price-normalized market dispersion metrics

---

## LAYER 3: VOLUME AND VOLATILITY FEATURES

Function: `layer3(df)` in data_featuring.py

### Spread Extensions (`add_L3_spreads`)

| Feature | Formula |
|---------|---------|
| `spread_close_absolute` | `max_close - min_close` |
| `spread_close_pct` | `(spread / min_close) * 100` |
| `spread_highlow_absolute` | `max_high - min_low` |
| `spread_highlow_pct` | `(spread / min_low) * 100` |

### Buy/Sell Exchange Features (`add_L3_buy_sell_exchange_features`)

| Feature | Formula |
|---------|---------|
| `volume_buy_exchange` | Volume at the buy exchange |
| `volume_sell_exchange` | Volume at the sell exchange |

**Purpose**: Determines maximum tradeable size

### Price Changes & Ratios (`add_L3_price_change_features`, `add_L3_buy_sell_exchange_price_ratio`)

| Feature | Formula |
|---------|---------|
| `price_change_buy_exchange` | Price change on buy exchange |
| `price_change_sell_exchange` | Price change on sell exchange |
| `price_ratio_buy_sell` | `close_sell / close_buy` |

### Rolling Volume Statistics (`add_L3_rolling_stats`)

For windows [5, 15, 30]:

| Feature | Formula |
|---------|---------|
| `volume_buy_rolling_std_{window}` | Std dev of buy exchange volume |
| `volume_sell_rolling_std_{window}` | Std dev of sell exchange volume |

### Volatility Aggregates (`add_L3_volatility_features`)

| Feature | Formula |
|---------|---------|
| `volatility_avg` | Mean of all exchange volatilities |
| `volatility_max` | Max volatility |
| `volatility_min` | Min volatility |
| `price_position_buy_exchange` | `(close - low) / (high - low)` [0-1] |
| `price_position_sell_exchange` | `(close - low) / (high - low)` [0-1] |

**Price Position**: 0=low, 0.5=mid, 1=high of the minute

---

## LAYER 4: ADVANCED TECHNICAL FEATURES

Function: `layer4(df)` in data_featuring.py

### Rolling Spread Statistics (`add_L4_rolling_stats`)

For windows [5, 15, 30]:

| Feature | Formula |
|---------|---------|
| `spread_rolling_std_{window}` | Std dev of spread |
| `spread_rolling_max_{window}` | Max spread |
| `spread_rolling_min_{window}` | Min spread |
| `spread_range_{window}` | Max - Min |

### Z-Score Features (`add_L4_zscore`)

For windows [5, 15, 30]:

| Feature | Formula |
|---------|---------|
| `spread_zscore_{window}` | `(spread - rolling_mean) / rolling_std` |

**Interpretation**: Z > 2 = unusually high, Z < -2 = unusually low

### Moving Averages (`add_L4_moving_averages`)

For windows [5, 15, 30]:

| Feature | Formula |
|---------|---------|
| `spread_ma_{window}` | Simple moving average |
| `spread_ema_{window}` | Exponential moving average |
| `volume_ma_buy_{window}` | MA of buy volume |
| `volume_ma_sell_{window}` | MA of sell volume |

### Rate of Change Features (`add_L4_rate_change_features`)

| Feature | Formula |
|---------|---------|
| `spread_rate_change` | `spread[t] - spread[t-1]` |
| `spread_rate_change_pct` | Percentage change |
| `spread_rate_acceleration` | Change in rate |

**Purpose**: Momentum signals (widening vs narrowing spreads)

### Opportunity Flags (`add_L4_flags`)

| Feature | Threshold |
|---------|-----------|
| `is_opportunity` | `spread ≥ 0.2%` |
| `is_real_opportunity` | `spread ≥ 0.3%` (after trading costs) |

---

## LAYER 5: TEMPORAL FEATURES

Function: `layer5(df)` in data_featuring.py

### Bollinger Bands (`add_L5_bollinger_bands`)

For windows [5, 15, 30]:

| Feature | Formula |
|---------|---------|
| `spread_bb_upper_{window}` | MA + (2 × std) |
| `spread_bb_lower_{window}` | MA - (2 × std) |
| `spread_bb_position_{window}` | `(spread - lower) / (upper - lower)` |

**BB Position**: 0=lower band, 0.5=middle, 1=upper band, >1 = breakout

### Lag Features (`add_L5_lag_features`)

For lags [1, 5, 10, 30]:

| Feature Type | Examples |
|---|---|
| Spread lags | `spread_lag_1, spread_lag_5, spread_lag_30` |
| Volume lags | `volume_buy_lag_1, volume_sell_lag_1, min_volume_lag_1` |
| Opportunity lags | `is_opportunity_lag_1, is_real_opportunity_lag_5` |
| Price change lags | `price_change_buy_lag_1, price_change_sell_lag_1` |
| Other lags | `volatility_avg_lag_1, buy_exchange_lag_1, sell_exchange_lag_1` |
| Diff features | `spread_diff_from_lag_1, spread_diff_from_lag_5` |

**Purpose**: Time-series prediction and autocorrelation

---

## FEATURE COUNT SUMMARY

| Layer | Functions | Total Features |
|-------|-----------|---|
| **2** | 4 functions (minmax, exchange, time, ratios) | ~30 |
| **3** | 6 functions (spreads, volume, price change, volatility) | ~50 |
| **4** | 6 functions (rolling stats, zscore, MA, rate change, flags) | ~50 |
| **5** | 2 functions (bollinger bands, lags) | ~40-50 |
| **TOTAL** | 18 feature engineering functions | **~160+** |

---

## Critical Notes

### Missing Data Behavior

- Aggregates across exchanges use `skipna=True` (require ≥2 exchanges)
- Rolling features create NaN for first N rows (where N = window size)
- Lag features create NaN for first N rows (where N = lag size)
- Exchange-specific features are NaN if that exchange has no data

### Data Quality Metric

`num_exchanges_available` tells you confidence:
- ≥ 5: Excellent
- 3-4: Good
- 2: Minimal
- 1: Very limited (essentially no arbitrage)

### Redundant Features (Intentional)

- `spread_bb_ma_{window}` duplicates `spread_ma_{window}` for feature selection flexibility

---

## Model Feature Selection Guide

### Critical features (baseline model):
- Spreads: `spread_close_pct`
- Exchanges: `buy_exchange`, `sell_exchange`
- Volume: `volume_buy_exchange`, `volume_sell_exchange`
- Opportunity: `is_opportunity`, `is_real_opportunity`

### High value (improved model):
-Time: `hour`, `day_of_week`
- Volatility: `volatility_avg`
- MAs: `spread_ma_5`, `spread_ema_5`
- Lags: `spread_lag_1`, `is_real_opportunity_lag_1`

### Medium value (if computation allows):
- Rolling stats: `spread_rolling_std_5`
- Rate change: `spread_rate_change`
- Price position: `price_position_buy_exchange`
- Price ratios: `price_ratio_std`

### Advanced (experimental):
- Bollinger Bands: `spread_bb_position_5`
- Z-scores: `spread_zscore_5`
- Longer lags: `spread_lag_30`

---

## Output

Six featured data CSV files created, one per cryptocurrency:
- `featured_BTCUSD_data.csv`
- `featured_ETHUSD_data.csv`
- `featured_DOGEUSD_data.csv`
- `featured_LINKUSD_data.csv`
- `featured_SOLUSD_data.csv`
- `featured_XRPUSD_data.csv`

Each: ~1440 rows × 168 columns (all features + source data)

---

## See Also

- [README.md](README.md) - Module overview
- [trading_costs.md](trading_costs.md) - Economic constraints
- [data_analysis_results.txt](data_analysis_results.txt) - Example output

