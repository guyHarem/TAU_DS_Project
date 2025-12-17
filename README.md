# Cryptocurrency Cross-Exchange Arbitrage Analysis

A comprehensive data science project for identifying and analyzing cryptocurrency arbitrage opportunities across multiple exchanges in real-time.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Collection](#data-collection)
- [Feature Engineering](#feature-engineering)
- [Data Analysis](#data-analysis)
- [Machine Learning Models](#machine-learning-models-work-in-progress)
- [Trading Costs & Profitability](#trading-costs--profitability)
- [Results & Insights](#results--insights)
- [Future Work](#future-work)

---

## 🎯 Overview

This project analyzes cross-exchange arbitrage opportunities in the cryptocurrency market by:

1. **Collecting** minute-by-minute OHLCV (Open, High, Low, Close, Volume) data from 6 major exchanges
2. **Engineering** 100+ features to identify arbitrage patterns
3. **Analyzing** spread dynamics, temporal patterns, and exchange behaviors
4. **Building** predictive models to forecast profitable opportunities (in progress)

### Supported Cryptocurrencies

- Bitcoin (BTC)
- Ethereum (ETH)
- Dogecoin (DOGE)
- Chainlink (LINK)
- Solana (SOL)
- Ripple (XRP)

### Supported Exchanges

- **Binance** (0.10% fees)
- **Bitfinex** (0.10-0.20% fees)
- **Coinbase** (0.40-0.60% fees)
- **Gate.io** (0.15% fees)
- **MEXC** (0.00-0.20% fees)
- **Kraken** (0.16-0.26% fees)

---

## ✨ Features

### Core Capabilities

- **Real-time Data Collection**: Fetch historical and live data from 6 exchanges via REST APIs
- **Automated Feature Engineering**: Generate 100+ features including spreads, volatility, momentum, and temporal patterns
- **Comprehensive Analysis**: Analyze opportunity frequency, temporal patterns, exchange behaviors, and risk factors
- **Cost Modeling**: Realistic trading cost calculations including fees, slippage, and transfer times
- **Profitability Estimation**: Calculate potential profits accounting for all trading costs

### Advanced Features

- Moving averages (SMA, EMA)
- Bollinger Bands
- Z-scores and rolling statistics
- Cross-exchange price ratios
- Lag features for time-series prediction
- Opportunity classification (basic vs. real opportunities)

---

## 📁 Project Structure

```
TAU_DS_Project/
├── data/
│   ├── raw_data/                      # Combined data from all exchanges
│   │   ├── combined_BTCUSD_data.csv
│   │   ├── combined_ETHUSD_data.csv
│   │   └── ...
│   └── featured_data/                 # Engineered features for ML
│       ├── featured_BTCUSD_data.csv
│       └── ...
├── data_retrieve/                     # API clients for exchanges
│   ├── data_retrieve.py              # Main data collection script
│   ├── binance_api.py
│   ├── bitfinex_api.py
│   ├── coinbase_api.py
│   ├── gateio_api.py
│   ├── kraken_api.py
│   └── mexc_api.py
├── data_analysis/                     # Feature engineering & analysis
│   ├── data_analysis.py              # Main analysis pipeline
│   ├── quick_arbitrage_check.py      # Quick opportunity scanner
│   ├── diagnose_spreads.py           # Spread diagnostics
│   ├── FEATURE_LIST.md               # Complete feature documentation
│   └── trading_costs.md              # Trading cost breakdown
├── models/                            # ML models (work in progress)
│   ├── ds_model/                     # Model artifacts
│   └── plotter.py                    # Visualization utilities
├── archive/                           # Legacy scripts
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn requests
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/guyHarem/TAU_DS_Project.git
cd TAU_DS_Project
```

2. Verify the directory structure:
```bash
ls -la data/ data_retrieve/ data_analysis/
```

---

## 📊 Data Collection

### Quick Start

Run the data retrieval script:

```bash
cd data_retrieve
python data_retrieve.py
```

### Usage Options

The script supports two modes:

#### 1. Standard Mode (Cross-Exchange Arbitrage)
Fetches crypto/USD pairs from all 6 exchanges:

```
Choose mode: 1
Currencies: BTC,ETH,DOGE
Enter start date (UTC): 2025-12-01 10:00
Enter end date (UTC): 2025-12-01 18:00
```

#### 2. Triangular Arbitrage Mode (Binance Only)
For analyzing BTC/ETH/USDT triangular arbitrage:

```
Choose mode: 2
Enter start date (UTC): 2025-12-01 10:00
Enter end date (UTC): 2025-12-01 18:00
```

### Output Format

Data is saved as CSV files with the following structure:

```csv
time,BINANCE:open,BINANCE:high,BINANCE:low,BINANCE:close,BINANCE:volume,BITFINEX:open,...
2025-12-01 10:00,54230.5,54245.3,54220.1,54235.8,12.5,54240.2,...
```

### API Notes

- All timestamps are in **UTC**
- Data is requested in **1-minute intervals**
- Requests are automatically chunked into 300-minute batches
- Failed requests are logged but don't stop the entire process
- Some exchanges may have gaps in historical data

---

## 🔧 Feature Engineering

### Running Feature Engineering

```bash
cd data_analysis
python data_analysis.py
```

Select option: `ADD`

### Feature Categories

The system generates **100+ features** across 11 categories:

#### 1. Basic Spread Features (Priority 1)
- `min_close`, `max_close`: Lowest and highest prices across exchanges
- `spread_close_absolute`, `spread_close_pct`: Price spread calculations
- `buy_exchange`, `sell_exchange`: Optimal exchanges for arbitrage
- `num_exchanges_available`: Data quality check

#### 2. Volume Features
- `volume_buy_exchange`, `volume_sell_exchange`: Liquidity on each side
- `min_volume`: Limiting factor for trade size
- `volume_ratio`: Liquidity comparison

#### 3. High-Low Theoretical Maximum
- `max_high`, `min_low`: Theoretical best-case scenario
- `spread_highlow_pct`: Maximum possible spread
- `opportunity_gap`: Difference between theoretical and actual

#### 4. Time-Based Features (Priority 1)
- `hour`, `minute`, `day_of_week`: Temporal identifiers
- `is_weekend`: Weekend flag (lower volatility expected)
- `overlap_hours`: Market overlap periods (19:00-21:00 UTC)

#### 5. Volatility Features (Priority 2)
- `{exchange}_volatility`: Per-exchange volatility
- `volatility_avg`, `volatility_max`: Aggregate volatility metrics
- `price_position_buy_exchange`, `price_position_sell_exchange`: Price momentum indicators

#### 6. Price Change Features (Priority 2)
- `{exchange}_price_change`: Momentum on each exchange
- `price_change_buy_exchange`, `price_change_sell_exchange`: Momentum for arbitrage pair

#### 7. Moving Averages (Priority 2)
Windows: 5, 15, 30 minutes
- `spread_ma_{window}`: Simple moving averages
- `spread_ema_{window}`: Exponential moving averages
- `volume_ma_buy_{window}`, `volume_ma_sell_{window}`: Volume trends

#### 8. Bollinger Bands (Priority 3)
Windows: 5, 15, 30 minutes
- `spread_bb_ma_{window}`: Middle band
- `spread_bb_upper_{window}`, `spread_bb_lower_{window}`: Upper/lower bands (±2 std)
- `spread_bb_position_{window}`: Relative position in bands

#### 9. Rolling Statistics (Priority 3)
Windows: 5, 15, 30 minutes
- `spread_rolling_std_{window}`: Spread volatility
- `spread_rolling_max_{window}`, `spread_rolling_min_{window}`: Range
- `spread_range_{window}`: Total spread range
- `spread_zscore_{window}`: Statistical outlier detection
- `opportunities_in_last_{window}`: Recent opportunity frequency

#### 10. Rate of Change Features (Priority 3)
- `spread_rate_change`: First derivative of spread
- `spread_rate_change_pct`: Percentage change
- `spread_rate_acceleration`: Second derivative

#### 11. Cross-Exchange Price Ratios (Priority 3)
- `price_ratio_buy_sell`: Price ratio between exchanges
- `price_ratio_{ex1}_{ex2}`: All pairwise ratios
- `avg_price_ratio`, `max_price_ratio`, `min_price_ratio`: Aggregate ratios

#### 12. Lag Features (For ML - Priority 4)
Lags: 1, 5, 10, 30 minutes
- `spread_lag_{lag}`: Historical spreads
- `volume_buy_lag_{lag}`, `volume_sell_lag_{lag}`: Historical volumes
- `is_opportunity_lag_{lag}`: Historical opportunity flags
- `price_change_buy_lag_{lag}`: Historical momentum

#### 13. Opportunity Flags (Priority 1)
- `is_opportunity`: Spread ≥ 0.50% (basic trading cost threshold)
- `is_real_opportunity`: Spread ≥ 0.60% (cost + safety margin)
- `num_exchanges_available`: Data completeness indicator

### Feature Engineering Output

Featured data is saved to `data/featured_data/` with all engineered features:

```
✅ Saved: ../data/featured_data/featured_BTCUSD_data.csv (15234 rows, 156 columns)
```

---

## 📈 Data Analysis

### Running Analysis

```bash
cd data_analysis
python data_analysis.py
```

Select option: `ANALYZE`

### Analysis Phases

#### Phase 1: Opportunity Frequency Analysis
- Total opportunities vs. real opportunities
- Percentage of time with profitable spreads
- Average opportunity duration
- Average spreads during opportunities

**Example Output:**
```
--- BTCUSD Analysis ---
Total minutes analyzed: 15234
Opportunities (≥0.5%): 892 (5.85%)
Real opportunities (≥0.6%): 456 (2.99%)

Opportunity Duration Statistics:
  Average duration: 3.2 minutes
  Median duration: 2.0 minutes
  Max duration: 45 minutes
  Total opportunity events: 142
```

#### Phase 2: Temporal Pattern Analysis
- Hourly opportunity rates (which hours are best?)
- Day-of-week patterns
- Weekend vs. weekday comparison
- Market overlap analysis (Asian/US/European markets)

**Key Insights:**
- Opportunities tend to peak during market overlaps (19:00-21:00 UTC)
- Weekend rates may differ from weekday rates
- Certain hours consistently show higher spreads

#### Phase 3: Exchange Pattern Analysis
- Most common buy/sell exchange pairs
- Exchange-specific arbitrage patterns
- Top 10 most profitable pairs

**Example Output:**
```
Top 10 Most Profitable Exchange Pairs:
  MEXC → COINBASE: 89 times (19.5%) - Avg spread: 0.78%
  GATEIO → BITFINEX: 67 times (14.7%) - Avg spread: 0.71%
  BINANCE → KRAKEN: 54 times (11.8%) - Avg spread: 0.65%
```

#### Phase 4: Volume & Liquidity Analysis
- Average volume during opportunities
- Volume distribution across percentiles
- Volume sufficiency for different trade sizes
- Volume ratio analysis (buy vs. sell side)

#### Phase 5: Risk Factor Analysis
- Volatility during opportunities
- Opportunity gap (theoretical vs. actual)
- High-gap opportunities (harder to execute)

#### Phase 6: Profitability Estimation
Assumes $1000 per trade:
- Total potential profit
- Average profit per trade
- Conservative estimates (high volume only)
- Profit per hour if all opportunities are traded

**Example Output:**
```
Assuming $1000 per trade:
  Total real opportunities: 456
  Total potential profit: $684.50
  Average profit per trade: $1.50
  Profit per hour (if traded all): $42.18

Conservative Estimate (volume ≥ 50):
  Opportunities: 234
  Total profit: $389.23
  Average profit per trade: $1.66
```

### Quick Arbitrage Check

For rapid analysis of a single cryptocurrency:

```bash
cd data_analysis
python quick_arbitrage_check.py
```

This script:
- Loads BTC data only
- Calculates spreads instantly
- Shows opportunity statistics
- Runs in <5 seconds

---

## 🤖 Machine Learning Models (Work in Progress)

The `models/` directory contains ongoing work to build predictive models for arbitrage opportunities.

### Planned Model Architecture

#### 1. **Opportunity Classification Model**
**Goal:** Predict if the next minute will have an arbitrage opportunity

- **Model Type:** Binary classification (Random Forest, XGBoost, Neural Network)
- **Features:** Lag features, moving averages, volatility, temporal features
- **Target:** `is_real_opportunity` (next minute)
- **Evaluation Metrics:**
  - Precision (avoid false positives = wasted execution)
  - Recall (catch as many opportunities as possible)
  - F1-score
  - ROC-AUC

#### 2. **Spread Prediction Model**
**Goal:** Forecast the spread percentage for the next 1-5 minutes

- **Model Type:** Regression (LSTM, GRU, Temporal Convolutional Network)
- **Features:** Time-series features, lag features, rolling statistics
- **Target:** `spread_close_pct` (future values)
- **Evaluation Metrics:**
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² score

#### 3. **Optimal Entry/Exit Predictor**
**Goal:** Determine the best time to enter and exit an arbitrage trade

- **Model Type:** Reinforcement Learning (Q-Learning, DQN) or Time-series classification
- **State Space:** Current spread, volume, volatility, recent history
- **Action Space:** Enter trade, hold position, exit trade
- **Reward Function:** Actual profit minus trading costs

#### 4. **Multi-Crypto Portfolio Optimizer**
**Goal:** Allocate capital across multiple crypto pairs to maximize profit

- **Model Type:** Portfolio optimization (Mean-Variance, Kelly Criterion)
- **Features:** Expected returns per crypto, correlation matrix, volatility
- **Target:** Optimal capital allocation

### Model Investigation Ideas

- **Feature Importance Analysis:** Which features are most predictive of profitable opportunities?
- **Temporal Cross-Validation:** Ensure no look-ahead bias in time-series models
- **Ensemble Methods:** Combine multiple models for better predictions
- **Real-Time Inference:** Optimize models for sub-second predictions
- **Backtesting Framework:** Simulate trading with slippage, fees, and latency
- **Transfer Learning:** Can a model trained on BTC work for ETH?

### Model Challenges to Address

1. **Data Imbalance:** Opportunities are rare (~3-6% of time)
   - Solution: SMOTE, class weighting, or anomaly detection approaches

2. **Non-Stationarity:** Market conditions change over time
   - Solution: Online learning, rolling window training, or regime detection

3. **Execution Latency:** 10-60 minute transfer times between exchanges
   - Solution: Predict spread persistence, not just current spread

4. **Slippage & Market Impact:** Large orders affect prices
   - Solution: Volume-aware predictions

5. **Overfitting:** Too many features on limited data
   - Solution: Regularization, feature selection, cross-validation

---

## 💰 Trading Costs & Profitability

### Trading Cost Breakdown

| Cost Component | Best Case | Realistic | Conservative |
|----------------|-----------|-----------|--------------|
| Trading fees (buy + sell) | 0.20% | 0.30% | 0.80% |
| Transfer fees (network) | 0.04% | 0.05% | 0.10% |
| Slippage | 0.05% | 0.10% | 0.15% |
| Time risk | 0.05% | 0.05% | 0.10% |
| **TOTAL** | **0.34%** | **0.50%** | **1.15%** |

### Opportunity Thresholds

- **Basic Opportunity:** Spread ≥ 0.50% (covers realistic costs)
- **Real Opportunity:** Spread ≥ 0.60% (costs + 0.10% safety margin)

### Profitability Considerations

✅ **Advantages:**
- Multiple opportunities per day
- Low correlation to overall market direction
- Automated execution possible

⚠️ **Risks:**
- Transfer time (10-60 minutes): price may move
- Exchange downtime or withdrawal limits
- Regulatory changes
- Network congestion (higher fees)

📊 **Realistic Expectations:**
- 2-6% of time has profitable opportunities
- Average profit: $1-3 per $1000 trade
- Requires high frequency execution to be worthwhile

For detailed cost analysis, see [trading_costs.md](data_analysis/trading_costs.md)

---

## 📊 Results & Insights

### Key Findings (Example Results)

#### Opportunity Frequency
- **BTC:** 2.5-3.5% of minutes have real opportunities
- **ETH:** 3.0-4.5% (slightly more volatile = more opportunities)
- **DOGE:** 4.0-6.0% (high volatility, high opportunity rate)

#### Most Profitable Exchange Pairs
1. **MEXC → COINBASE:** MEXC often has lower prices, Coinbase higher (but high fees)
2. **Gate.io → Kraken:** Consistent spreads with reasonable fees
3. **Binance → Bitfinex:** High liquidity, reliable execution

#### Temporal Patterns
- **Best Hours:** 19:00-21:00 UTC (market overlaps)
- **Best Days:** Weekdays slightly better than weekends
- **Worst Hours:** 4:00-6:00 UTC (low liquidity)

#### Risk Factors
- High volatility periods → more opportunities but higher risk
- Low volume opportunities → may not be executable
- Opportunity gap >0.2% → execution at close price unlikely

---

## 🔮 Future Work

### Short-Term Goals
- [ ] Complete ML model training and evaluation
- [ ] Implement backtesting framework with realistic execution simulation
- [ ] Add real-time data streaming capability
- [ ] Build alerting system for high-probability opportunities

### Medium-Term Goals
- [ ] Integrate WebSocket APIs for sub-second data
- [ ] Develop automated trading bot (paper trading first)
- [ ] Add more exchanges (Bybit, OKX, Huobi)
- [ ] Implement triangular arbitrage analysis (BTC/ETH/USDT)

### Long-Term Goals
- [ ] Deploy cloud-based real-time arbitrage detection system
- [ ] Create web dashboard for monitoring opportunities
- [ ] Integrate with exchange APIs for automated execution
- [ ] Research cross-chain arbitrage (e.g., Ethereum L2s)

---

## 📚 Documentation

- **[FEATURE_LIST.md](data_analysis/FEATURE_LIST.md):** Complete documentation of all 100+ features
- **[trading_costs.md](data_analysis/trading_costs.md):** Detailed breakdown of trading costs and thresholds

---

## 🤝 Contributing

This is an academic project for Tel Aviv University. Contributions, suggestions, and feedback are welcome!

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. 

- Cryptocurrency trading involves significant risk
- Past performance does not guarantee future results
- This is NOT financial advice
- Always do your own research before trading
- Be aware of regulatory requirements in your jurisdiction

---

## 📧 Contact

**Author:** Guy Harem  
**Institution:** Tel Aviv University  
**Repository:** [github.com/guyHarem/TAU_DS_Project](https://github.com/guyHarem/TAU_DS_Project)

---

## 📜 License

This project is for academic use. Please contact the author for licensing information.

---

**Last Updated:** December 2025
