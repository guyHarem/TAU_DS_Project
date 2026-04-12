<a id="readme-top"></a>

<br/>
<div align="center">
<h3 align="center"><a href="https://github.com/guyHarem/TAU_DS_Project">Cryptocurrency Cross-Exchange Arbitrage Analysis</a></h3>
  <p align="center">
    A comprehensive data science project for identifying and analyzing cryptocurrency arbitrage opportunities across multiple exchanges in real-time.
  </p>
</div>

<details>
<summary>Table of Contents</summary>

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Architecture](#project-architecture)
- [System Workflow](#system-workflow)
- [Module 1: Data Retrieval](#module-1-data-retrieval)
  - [Overview](#data-retrieval-overview)
  - [Supported Exchanges](#supported-exchanges)
  - [Usage](#data-retrieval-usage)
- [Module 2: Data Analysis & Feature Engineering](#module-2-data-analysis--feature-engineering)
  - [Overview](#data-analysis-overview)
  - [Feature Layer Architecture](#feature-layer-architecture)
  - [Feature Categories](#feature-categories)
  - [Usage](#data-analysis-usage)
- [Module 3: Machine Learning Models](#module-3-machine-learning-models)
  - [Overview](#models-overview)
  - [Model Architectures](#model-architectures)
  - [Training & Evaluation](#model-training--evaluation)
  - [Usage Examples](#usage-examples)
- [Orchestration & CLI](#orchestration--cli)
- [Configuration & Setup](#configuration--setup)
- [Documentation References](#documentation-references)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)
- [Collaborators](#collaborators)

</details>

---

## Overview

This project analyzes cross-exchange arbitrage opportunities in cryptocurrency markets by:

1. **Collecting** minute-by-minute OHLCV data from 6 major exchanges
2. **Engineering** 160+ features across 5 dependency layers
3. **Analyzing** spread dynamics, temporal patterns, exchange behaviors, and profitability
4. **Building** predictive models to forecast profitable opportunities

### Supported Cryptocurrencies

Bitcoin (BTC), Ethereum (ETH), Dogecoin (DOGE), Chainlink (LINK), Solana (SOL), Ripple (XRP)

### Supported Exchanges

Binance, Bitfinex, Coinbase, Gate.io, Kraken, MEXC

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/guyHarem/TAU_DS_Project.git
cd TAU_DS_Project
pip install -r requirements.txt
```

### 2. Data Collection
```bash
python arbitrage_oracle.py data fetch
```

### 3. Feature Engineering
```bash
python arbitrage_oracle.py features add
```

### 4. Analysis & Model Training
```bash
python arbitrage_oracle.py analysis run
python arbitrage_oracle.py models train --all
```

### 5. View Results
```bash
python arbitrage_oracle.py models evaluate
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Project Architecture

```
TAU_DS_Project/
├── arbitrage_oracle.py             # 🎯 MAIN CLI - Unified orchestration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── 📥 data_retrieve/               # MODULE 1: Data Collection
│   ├── data_retrieve.py            # Main orchestrator
│   ├── binance_api.py              # Binance API client
│   ├── bitfinex_api.py             # Bitfinex API client
│   ├── coinbase_api.py             # Coinbase API client
│   ├── gateio_api.py               # Gate.io API client
│   ├── kraken_api.py               # Kraken API client
│   └── mexc_api.py                 # MEXC API client
│
├── 🔧 data_analysis/               # MODULE 2: Feature Engineering & Analysis
│   ├── data_featuring.py           # 5-layer feature engineering pipeline
│   ├── data_analysis.py            # 11-section statistical analysis
│   ├── FEATURE_LIST.md             # Complete feature documentation
│   ├── trading_costs.md            # Trading cost & profitability analysis
│   └── data_analysis_results.txt   # Analysis output results
│
├── 🧠 models/                      # MODULE 3: Machine Learning
│   ├── model_lstm.py               # LSTM regression model
│   ├── model_gru.py                # GRU regression model
│   ├── model_transformer.py        # Transformer attention model
│   ├── model_linear.py             # Logistic regression classifier
│   ├── model_randomforest.py       # Random Forest classifier
│   ├── model_xgboost.py            # XGBoost classifier
│   ├── model_catboost.py           # CatBoost classifier
│   ├── plotter.py                  # Universal plotting utility (300 DPI)
│   └── ds_model/                   # Model artifacts & visualizations
│
├── 💾 data/                        # Data Storage
│   ├── raw_data/                   # Combined multi-exchange OHLCV data
│   └── featured_data/              # Engineered feature sets (160+ features)
│
├── ✅ tests/                       # Test Folder
│   ├── test_data_validation.py
│   ├── test_edge_cases.py
│   ├── test_model_outputs.py
│   ├── test_models_integration.py
│   ├── test_models.py
│   ├── test_performance_benchmarks.py
│   └── test_raw_data.py


```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## System Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    ARBITRAGE ORACLE (CLI)                   │
│              Unified Interface for All Operations            │
└─────────────────────────────────────────────────────────────┘
                             ↓
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │   Raw Data  │   │   Features  │   │ ML Training │
   │ Collection  │   │ Engineering │   │   & Eval    │
   └─────────────┘   └─────────────┘   └─────────────┘
        ↓                    ↓                    ↓
   6 Exchanges      160+ Features      7 Models
   6 Currencies     5 Layers           2 Paradigms
   1-min OHLCV      4 Hours            Real-time
   
   ┌─────────────────────────────────────────────────────────┐
   │         OUTPUT: Predictions & Visualizations            │
   │  (300 DPI PNG, PDF, CSV reports, model artifacts)       │
   └─────────────────────────────────────────────────────────┘
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

# MODULE 1: DATA RETRIEVAL

<a id="data-retrieval-overview"></a>

## Overview

Fetches 1-minute candlestick OHLCV data across 6 exchanges and 6 cryptocurrencies, merging into unified format for analysis.

### Data Flow

```
User Input (currencies, date range)
        ↓
[Dynamic Module Loading]
        ↓
[Parallel Data Fetching]
    ├─ Binance (1000 rec/request)
    ├─ Bitfinex (10k min windows)
    ├─ Coinbase (300 rec/request)
    ├─ Gate.io (daily .gz files)
    ├─ Kraken (custom rate limit)
    └─ MEXC (1000 rec/request)
        ↓
[Outer Join on Timestamp]
        ↓
data/raw_data/combined_{SYMBOL}_data.csv
```

<a id="supported-exchanges"></a>

## Supported Exchanges

| Exchange | Pair Format | Pagination | Rate Limit | Special Features |
|---|---|---|---|---|
| **Binance** | BTCUSDT (USD→USDT) | 1000/req | 0.1s | Pair reversal |
| **Bitfinex** | BTC:USD | 10k chunks | 0.1s | Pair validation |
| **Coinbase** | BTC-USD | 300/req | 0.1-1s | Newest-first |
| **Gate.io** | BTC_USDT | Daily .gz | None | No rate limits |
| **Kraken** | XBTUSDT | Variable | Budget-based | Asset mapping |
| **MEXC** | BTCUSDT | 1000/req | 0.2s | Pair reversal |

### Exchange Details

- **Binance & MEXC:** Map USD→USDT, handle reversed pairs, 1000 rec/request
- **Bitfinex:** Pair validation, 10k-minute chunking for large ranges
- **Coinbase:** ISO8601 timestamps, newest-first ordering
- **Gate.io:** Archive API (.csv.gz files) bypasses rate limits
- **Kraken:** Asset name mapping (BTC→XBT, DOGE→XDG), counter-based rate limit
- All exchanges handle errors gracefully without stopping other fetches

<a id="data-retrieval-usage"></a>

## Data Retrieval Usage

### Interactive Mode
```bash
python data_retrieve/data_retrieve.py
```

Prompts for currencies, date range, and confirmation.

### Programmatic
```python
from data_retrieve.data_retrieve import fetch_data_from_modules, merge_dataframes
all_data = fetch_data_from_modules("BTC", "USD", "2025-03-01 10:00", "2025-03-02 10:00")
combined = merge_dataframes(all_data)
```

### Output Format
```csv
time,BINANCE:open,BINANCE:high,BINANCE:low,BINANCE:close,BINANCE:volume,BITFINEX:open,...
2025-03-01 10:00,54230.5,54245.3,54220.1,54235.8,12.5,54240.2,...
```

### CLI
```bash
python arbitrage_oracle.py data fetch
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

# MODULE 2: DATA ANALYSIS & FEATURE ENGINEERING

<a id="data-analysis-overview"></a>

## Overview

Transforms raw multi-exchange data into **160+ engineered features** organized across **5 dependency layers**, then produces statistical analysis of opportunities, risks, and profitability.

<a id="feature-layer-architecture"></a>

## Feature Layer Architecture

```
Raw Data (6 currencies × 6 exchanges, 1-min OHLCV)
    ↓
[LAYER 2] Foundation (27 features)
    • Spreads, exchanges, time features, price ratios
    ↓
[LAYER 3] Volume & Volatility (38 features)
    • Buy/sell volumes, rolling stats, price momentum
    ↓
[LAYER 4] Technical Analysis (68 features)
    • Moving averages, z-scores, rate of change, flags
    ↓
[LAYER 5] Temporal & Bollinger (27 features)
    • Bollinger bands, lag features, derivatives
    ↓
Featured Data (160+ features, ML-ready)
```

<a id="feature-categories"></a>

## Feature Categories

### Layer 2: Foundation (27)
- **Spreads:** `min_close`, `max_close`, `spread_close_absolute`, `spread_close_pct`
- **Exchanges:** `buy_exchange`, `sell_exchange`, `num_exchanges_available`
- **Time:** `hour`, `minute`, `day_of_week`, `is_weekend`, `overlap_hours`
- **Volatility:** Per-exchange and aggregate metrics
- **Price Ratios:** Pairwise and aggregate price ratios across exchanges

### Layer 3: Volume & Volatility (38)
- **Exchange-Specific:** `volume_buy_exchange`, `volume_sell_exchange`
- **Momentum:** `price_change_buy_exchange`, `price_change_sell_exchange`
- **Spread Extensions:** `spread_highlow_absolute`, `opportunity_gap`
- **Rolling Stats:** Volume volatility per window (5, 15, 30 min)

### Layer 4: Technical Analysis (68)
- **Rolling Statistics:** `spread_rolling_std_{window}`, `spread_zscore_{window}`
- **Moving Averages:** SMA & EMA for spreads and volumes (windows: 5, 15, 30)
- **Rate of Change:** 1st & 2nd derivatives, percentage changes
- **Opportunity Flags:** `is_opportunity` (≥0.2%), `is_real_opportunity` (≥0.3%)

### Layer 5: Temporal & Bollinger (27)
- **Bollinger Bands:** Upper, lower, position for windows (5, 15, 30)
- **Lag Features:** Spreads, volumes, opportunities (lags: 1, 5, 10, 30)
- **Historical Pairs:** `buy_exchange_lag_1`, `sell_exchange_lag_1`

<a id="data-analysis-usage"></a>

## Data Analysis Usage

### Feature Engineering
```bash
python data_analysis/data_featuring.py
# Output: featured_*.csv (160+ columns each)
```

### Statistical Analysis
```bash
python data_analysis/data_analysis.py
# Output: data_analysis_results.txt (11 analysis sections)
```

### Analysis Sections (11 Total)
1. **Opportunity Frequency** - % of minutes with opportunities
2. **Average Spreads** - Profitability distribution
3. **Temporal Patterns** - Best hours/days
4. **Exchange Patterns** - Most profitable trading pairs
5. **Volume & Liquidity** - Tradeable volumes
6. **Risk Assessment** - Volatility analysis
7. **Profitability Estimation** - Expected profit per trade
8. **Momentum Indicators** - MA patterns
9. **Bollinger Bands** - Statistical extremes
10. **Persistence Patterns** - Autocorrelation
11. **Rolling Statistics** - Volatility clustering

### CLI
```bash
python arbitrage_oracle.py features add
python arbitrage_oracle.py features list
python arbitrage_oracle.py analysis run
```

### Configuration
```python
TRADING_COST_PCT = 0.2              # Basic trading costs
SAFETY_MARGIN_PCT = 0.1             # Safety buffer
REAL_OPPORTUNITY_THRESHOLD = 0.3    # Profitable spread threshold
windows = [5, 15, 30]               # Rolling window sizes (minutes)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

# MODULE 3: MACHINE LEARNING MODELS

<a id="models-overview"></a>

## Overview

**7 trained models** + **1 plotting utility** for predicting arbitrage opportunities and spread movements.

**Twin Paradigm:**
- **RNN Regressors** (LSTM, GRU, Transformer): Predict `spread_close_pct` (continuous)
- **Tree/Linear Classifiers** (Linear, RF, XGB, CatBoost): Predict `is_real_opportunity` (binary)

| Model | Task | Framework | Input | Output | Training Time |
|---|---|---|---|---|---|
| **LSTM** | Regression | TensorFlow | Sequences | Continuous | Slow |
| **GRU** | Regression | TensorFlow | Sequences | Continuous | Medium |
| **Transformer** | Regression | PyTorch | Sequences | Continuous | Medium |
| **Linear** | Classification | scikit-learn | Direct features | Probability | Very fast |
| **Random Forest** | Classification | scikit-learn | Direct features | Probability | Fast |
| **XGBoost** | Classification | XGBoost | Direct features | Probability | Fast |
| **CatBoost** | Classification | CatBoost | Direct features | Probability | Fast |

<a id="model-architectures"></a>

## Model Architectures

### RNN Regressors

#### LSTM (model_lstm.py)
- **Architecture:** Input → LSTM(units) → Dropout → Dense → Dense(sigmoid)
- **Output:** Continuous spread prediction [0, 1]
- **Hyperparameters:** lstm_units (64), dense_units (32), dropout_rate (0.2), sequence_length (10)
- **Scaling:** MinMaxScaler [0, 1]
- **Loss:** MSE, Optimizer: Adam

#### GRU (model_gru.py)
- **Architecture:** Similar to LSTM but with GRU layer (simpler, faster)
- **Advantage:** Less parameters than LSTM
- **Use When:** Speed is important or limited data

#### Transformer (model_transformer.py)
- **Framework:** PyTorch
- **Architecture:** Input → PositionalEncoding → TransformerEncoder → FC layers → Output
- **Components:** Multi-head self-attention, feedforward networks
- **Hyperparameters:** d_model (64), nhead (4), num_layers (2), dim_feedforward (256)
- **Advantage:** Interpretable attention weights, parallelizable

### Tree/Linear Classifiers

#### Linear Classifier (model_linear.py)
- **Type:** Logistic Regression with L1/L2 regularization
- **Options:** 'linear', 'ridge', 'lasso'
- **Scaling:** StandardScaler (z-score)
- **Use:** Baseline, interpretability, fast training

#### Random Forest (model_randomforest.py)
- **Type:** Ensemble of decision trees
- **Hyperparameters:** n_estimators (300), max_depth (20), class_weight ('balanced')
- **Scaling:** Not required (scale-invariant)
- **Feature Importance:** From tree splits
- **Use:** Nonlinear relationships, no hyperparameter tuning

#### XGBoost (model_xgboost.py)
- **Type:** Gradient Boosting
- **Hyperparameters:** n_estimators (600), learning_rate (0.03), max_depth (5)
- **Early Stopping:** On validation set
- **Feature Importance:** Gain metric
- **Use:** Production, high performance, feature interactions

#### CatBoost (model_catboost.py)
- **Type:** Categorical Boosting
- **Hyperparameters:** iterations (1000), learning_rate (0.03), depth (6)
- **Special:** Native categorical support, reduced overfitting
- **Methods:** bucket_classification_metrics(), opportunity_detection_metrics()
- **Use:** Native categorical features, production-ready

### Plotter Module (plotter.py)

**9 Plotting Functions** for model evaluation (300 DPI PNG + PDF):

| Function | Purpose | Models |
|---|---|---|
| `plot_results()` | 4-subplot regression analysis | LSTM, GRU, Transformer |
| `plot_prediction_hist()` | Prediction distribution | All |
| `plot_training_history()` | Loss curves over epochs | RNN models |
| `plot_feature_importance()` | Feature importance | All (model-aware) |
| `plot_pr_curve()` | Precision-Recall curve | Classifiers |
| `plot_threshold_metrics()` | Threshold analysis | Classifiers |
| `plot_prediction_history()` | Time-series predictions | All |
| `plot_xgb_feature_importance()` | XGBoost gain importance| XGBoost |
| `save_plot()` | PNG (300 DPI) + PDF saver | Helper |

<a id="model-training--evaluation"></a>

## Training & Evaluation

### Data Pipeline: RNN Models
```python
df = load_data('BTCUSD')                    # Load featured data
X, y = prepare_features(df)                 # Exclude time/exchanges/target
X_sequences = create_sequences(X)           # Rolling windows
X_scaled = scale_features(X_sequences)      # MinMaxScaler [0, 1]
```
- **Features Excluded:** time, buy_exchange, sell_exchange
- **Scaling:** MinMaxScaler (neural networks benefit from bounded inputs)

### Data Pipeline: Tree/Linear Models
```python
df = load_data('BTCUSD')                    # Load featured data
X, y = prepare_features(df)                 # Exclude time/categoricals/target
X_scaled = scale_features(X)                # StandardScaler (z-score)
```
- **Train/Test Split:** TimeSeriesSplit (no future leakage)
- **Scaling:** StandardScaler (linear models assume normalized)

### Evaluation Metrics

**Regression:** MSE, MAE, R², residual analysis
**Classification:** Accuracy, Precision, Recall, F1, ROC-AUC, PR curves

<a id="usage-examples"></a>

## Usage Examples

### Train LSTM
```bash
python models/model_lstm.py --symbol BTCUSD --seed 42
```

### Train XGBoost
```bash
python models/model_xgboost.py --symbol BTCUSD --n-estimators 600 --seed 42
```

### Train CatBoost
```bash
python models/model_catboost.py --symbol BTCUSD --iterations 1000 --seed 42
```

### Train All Models
```bash
python arbitrage_oracle.py models train --all
```

### View Results
```bash
python arbitrage_oracle.py models evaluate
python arbitrage_oracle.py models list-trained
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Orchestration & CLI

**arbitrage_oracle.py** is the main CLI providing unified access:

### Main Commands
```bash
# Data
python arbitrage_oracle.py data fetch

# Features
python arbitrage_oracle.py features add
python arbitrage_oracle.py features list

# Analysis
python arbitrage_oracle.py analysis run

# Models
python arbitrage_oracle.py models train --all
python arbitrage_oracle.py models train --model xgboost,catboost
python arbitrage_oracle.py models list
python arbitrage_oracle.py models evaluate
```

### Configuration
- **Symbols:** BTC, ETH, DOGE, SOL, XRP, LINK (vs USD)
- **Exchanges:** Binance, Bitfinex, Coinbase, Gate.io, Kraken, MEXC
- **Models:** lstm, gru, transformer, linear, randomforest, xgboost, catboost

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Configuration & Setup

### Installation
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- pandas, numpy (data processing)
- tensorflow, torch (deep learning)
- scikit-learn, xgboost, catboost (ML)
- matplotlib, seaborn (visualization)
- requests (API calls)

### API Configuration
- No API keys required for public exchange data
- Optional: Set exchange API keys for private data/high rate limits

### Model Output
Models save artifacts to `models/ds_model/{model_type}/{symbol}/`:
- PNG plots (300 DPI)
- PDF plots (vector)
- Model checkpoints (`.joblib`, `.pth`)
- Performance metrics

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Documentation References

### Feature Documentation
- **[FEATURE_LIST.md](FEATURE_LIST.md)** - All 160+ features by layer
  - Feature definitions and explanations
  - Why each feature matters
  - Missing data handling

### Trading Economics
- **[trading_costs.md](trading_costs.md)** - Cost breakdown & profitability
  - Fee structures by exchange
  - Slippage estimates
  - Profitability thresholds

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contributing

This is an academic project for Tel Aviv University. Contributions and feedback are welcome!

### Development
1. Fork repository
2. Create feature branch
3. Add tests
4. Commit and push
5. Open Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Disclaimer

**⚠️ For educational and research purposes only.**

- Cryptocurrency trading involves significant financial risk
- Past performance does NOT guarantee future results
- This is NOT financial advice
- Always conduct your own research before trading
- Be aware of regulatory requirements in your jurisdiction
- No warranty or liability for losses

**Use at your own risk.**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Collaborators

**Authors:** [Oz Cabiri](https://github.com/OzCabiri) & [Guy Harem](https://github.com/guyHarem)  
**Institution:** Tel Aviv University  
**Repository:** [github.com/guyHarem/TAU_DS_Project](https://github.com/guyHarem/TAU_DS_Project)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## License

For academic use. Contact authors for licensing information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
