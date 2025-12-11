# Trading Costs Reference

## Exchange-Specific Trading Fees

| Exchange | Maker Fee | Taker Fee | Notes |
|----------|-----------|-----------|-------|
| **Binance** | 0.10% | 0.10% | Can be lower with BNB (0.075%) |
| **Coinbase Pro** | 0.40% | 0.60% | High fees! |
| **Bitfinex** | 0.10% | 0.20% | Volume-based discounts |
| **Kraken** | 0.16% | 0.26% | Lower with volume |
| **Gate.io** | 0.15% | 0.15% | |
| **MEXC** | 0.00% | 0.20% | Maker rebate |

## Network Withdrawal/Deposit Fees

| Crypto | Typical Withdrawal Fee | Cost at $54k BTC | Percentage |
|--------|------------------------|------------------|------------|
| **BTC** | 0.0005 BTC | $27 | 0.05% |
| **ETH** | 0.005 ETH | Variable | 0.08% |
| **DOGE** | Variable | Low | 0.02% |
| **USDT** | $1-25 | Depends on network | 0.03% |

## Cost Breakdown by Scenario

### Best Case Scenario (Binance/Bitfinex, fast execution)

| Cost Component | Percentage | Notes |
|----------------|------------|-------|
| Trading fees (0.10% × 2) | 0.20% | Binance taker + Bitfinex taker |
| Transfer fees | 0.04% | One BTC transfer |
| Slippage | 0.05% | Minimal market impact |
| Time risk | 0.05% | Fast execution, low volatility |
| **TOTAL** | **0.34%** | **Minimum viable threshold** |

**Note:** Bid-ask spread is NOT included as a separate cost because `close` prices already reflect actual execution prices.

### Realistic Scenario (Average cross-exchange arbitrage)

| Cost Component | Percentage | Notes |
|----------------|------------|-------|
| Trading fees (0.15% × 2) | 0.30% | Average exchange fees |
| Transfer fees | 0.05% | One BTC transfer |
| Slippage | 0.10% | Moderate market impact |
| Time risk | 0.05% | Price change during 10-60 min transfer |
| **TOTAL** | **0.50%** | **Recommended threshold** |

### Pessimistic Scenario (High-fee exchanges, slow execution)

| Cost Component | Percentage | Notes |
|----------------|------------|-------|
| Trading fees (0.40% × 2) | 0.80% | Coinbase or other high-fee exchanges |
| Transfer fees | 0.10% | BTC transfer |
| Slippage | 0.15% | Large orders, low liquidity |
| Time risk | 0.10% | High volatility during transfer |
| **TOTAL** | **1.15%** | **Conservative threshold** |

## Current Analysis Uses

| Scenario | Threshold | Use Case |
|----------|-----------|----------|
| **Best Case** | 0.45% | Finding maximum opportunities |
| **Realistic** | 0.60% | Recommended for analysis |
| **Conservative** | 0.70% | Risk-averse modeling |
| **Pessimistic** | 1.20% | Worst-case scenario |

## Important Notes

### Why Bid-Ask Spread is NOT a Separate Cost

When using historical `close` prices from CSV data:
- ✅ `close` price = actual execution price (someone traded at this price)
- ✅ Bid-ask spread is already "baked in" to the close price
- ❌ **Do NOT subtract bid-ask spread separately** - this would double-count the cost

**Bid-ask spread only matters when:**
- Using live order book data (bid/ask quotes)
- Calculating from mid-prices
- Building real-time trading systems

### Cost Component Definitions

- **Trading fees**: Cost to execute buy and sell orders (2 trades per arbitrage)
- **Transfer fees**: Network cost to move crypto between exchanges (mandatory for cross-exchange arbitrage) - **Updated to 0.15%**
- **Slippage**: Price movement during order execution due to market impact
- **Time risk**: Risk of price change during BTC transfer (typically 10-60 minutes)

### Exchange-Specific Pair Costs

| Buy Exchange | Sell Exchange | Total Trading Fees | Recommended Threshold |
|--------------|---------------|-------------------|----------------------|
| Binance | Bitfinex | 0.30% | 0.55% |
| Binance | Coinbase | 0.70% | 0.95% |
| Coinbase | Bitfinex | 0.80% | 1.05% |

## Recommendations

1. **For initial exploration**: Use **0.45%** threshold to see maximum opportunities
2. **For realistic analysis**: Use **0.60%** threshold (recommended)
3. **For ML modeling**: Use **0.60-0.70%** to account for execution uncertainty
4. **For live trading**: Add 0.10% safety margin (use 0.70% threshold)

## Updates

- **Removed** "Optimistic same-exchange" scenario (not applicable to cross-exchange arbitrage)
- **Removed** bid-ask spread as separate cost (already in close prices)
- **Updated** transfer fees from 0.05% to 0.15%
- **Updated** realistic total from 0.50% to 0.60%
- **Updated** best-case total from 0.35% to 0.45%
- **Updated** pessimistic total from 1.15% to 1.20%