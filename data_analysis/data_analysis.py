import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

#region Hyper-parameters

TRADING_COST_PCT = 0.2
SAFETY_MARGIN_PCT = 0.1
REAL_OPPORTUNITY_THRESHOLD = TRADING_COST_PCT + SAFETY_MARGIN_PCT
TRADE_AMOUNT_USD = 1000  # Assume $1000 per trade

ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw_data"
FEATURED_DATA_PATH = DATA_PATH / "featured_data"

#endregion

#region BASIC ANALASYS METHODS
def analyze_opportunity_frequency(datasets):
    print("\n" + "="*60)
    print("=== OPPORTUNITY FREQUENCY (1) ===")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        total_rows = len(df)
        # Derive opportunity counts from spread thresholds to avoid stale labels.
        total_opportunities = (df['spread_close_pct'] > TRADING_COST_PCT).sum()
        real_opportunities = (df['spread_close_pct'] > REAL_OPPORTUNITY_THRESHOLD).sum()
        
        opportunity_pct = (total_opportunities / total_rows) * 100
        real_opportunity_pct = (real_opportunities / total_rows) * 100
        
        print(f"\n--- {name} Opportunity ({total_rows} minutes) ---")
        print(f"Opportunities (≥{TRADING_COST_PCT}%): {total_opportunities} ({opportunity_pct:.2f}%)")
        print(f"Real opportunities (≥{REAL_OPPORTUNITY_THRESHOLD}%): {real_opportunities} ({real_opportunity_pct:.2f}%)")
        
        # Opportunity duration analysis
        df['opportunity_group'] = (df['is_real_opportunity'] != df['is_real_opportunity'].shift()).cumsum()
        opportunity_durations = df[df['is_real_opportunity'] == 1].groupby('opportunity_group').size()
        
        if len(opportunity_durations) > 0:
            print(f"\nOpportunity Duration Statistics (>0.3%):")
            print(f"  Average duration: {opportunity_durations.mean():.2f} minutes")
            print(f"  Median duration: {opportunity_durations.median():.0f} minutes")
            print(f"  Max duration: {opportunity_durations.max():.0f} minutes")
            print(f"  Total opportunity events: {len(opportunity_durations)}")

def analyze_spreads(datasets):
    print("\n" + "="*60)
    print("=== AVERAGE SPREADS (2) ===")
    print("="*60 + "\n")

    for name, df in datasets.items():
        total_rows = len(df)
        avg_spread_all = df['spread_close_pct'].mean()
        avg_spread_opportunity = df[df['is_opportunity'] == 1]['spread_close_pct'].mean()
        avg_spread_real = df[df['is_real_opportunity'] == 1]['spread_close_pct'].mean()
        
        print(f"\n--- {name} Avg Spreads ({total_rows} minutes) ---")
        print(f"  All times: {avg_spread_all:.4f}%")
        print(f"  During opportunities: {avg_spread_opportunity:.4f}%")
        print(f"  During real opportunities: {avg_spread_real:.4f}%")

def analyze_temporal_patterns(datasets):
    print("\n" + "="*60)
    print("=== TEMPORAL PATTERN ANALYSIS (3) ===")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Temporal ---")
        
        # Hourly analysis
        hourly_opportunities = df.groupby('hour')['is_real_opportunity'].agg(['sum', 'mean', 'count'])
        hourly_opportunities['opportunity_rate'] = (hourly_opportunities['sum'] / hourly_opportunities['count']) * 100
        
        print("\nTop 3 Hours for Opportunities:")
        top_hours = hourly_opportunities.nlargest(3, 'opportunity_rate')
        for hour, row in top_hours.iterrows():
            print(f"  Hour {hour:02d}:00 - {row['opportunity_rate']:.2f}% ({int(row['sum'])} opportunities)")
        
        # Day of week analysis
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_opportunities = df.groupby('day_of_week')['is_real_opportunity'].agg(['sum', 'mean', 'count'])
        daily_opportunities['opportunity_rate'] = (daily_opportunities['sum'] / daily_opportunities['count']) * 100
        daily_opportunities['day_name'] = [day_names[i] for i in daily_opportunities.index]
        
        print("\nOpportunity Rate by Day of Week:")
        for idx, row in daily_opportunities.iterrows():
            print(f"  {row['day_name']}: {row['opportunity_rate']:.2f}% ({int(row['sum'])} opportunities)")
        
        # Weekend vs Weekday
        weekend_rate = df[df['is_weekend'] == 1]['is_real_opportunity'].mean() * 100
        weekday_rate = df[df['is_weekend'] == 0]['is_real_opportunity'].mean() * 100
        
        print(f"\nWeekend vs Weekday:")
        print(f"  Weekend opportunity rate: {weekend_rate:.2f}%")
        print(f"  Weekday opportunity rate: {weekday_rate:.2f}%")
        print(f"  Difference: {weekend_rate - weekday_rate:+.2f}%")
        
        # Market overlap hours
        overlap_rate = df[df['overlap_hours'] == 1]['is_real_opportunity'].mean() * 100
        non_overlap_rate = df[df['overlap_hours'] == 0]['is_real_opportunity'].mean() * 100
        
        print(f"\nMarket Overlap Hours (19:00-21:00 UTC):")
        print(f"  During overlap: {overlap_rate:.2f}%")
        print(f"  Outside overlap: {non_overlap_rate:.2f}%")
        print(f"  Difference: {overlap_rate - non_overlap_rate:+.2f}%")

def analyze_exchange_patterns(datasets):
    print("\n" + "="*60)
    print("=== EXCHANGE PATTERN ANALYSIS (4) ===")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Exchange ---")
        
        # Most common exchange pairs
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            df_opportunities['exchange_pair'] = df_opportunities['buy_exchange'] + ' → ' + df_opportunities['sell_exchange']
            pair_counts = df_opportunities['exchange_pair'].value_counts()
            
            print("\nTop 5 Most Profitable Exchange Pairs:")
            for pair, count in pair_counts.head(5).items():
                pct = (count / len(df_opportunities)) * 100
                avg_spread = df_opportunities[df_opportunities['exchange_pair'] == pair]['spread_close_pct'].mean()
                print(f"  {pair}: {count} times ({pct:.1f}%) - Avg spread: {avg_spread:.4f}%")
            
            # Buy exchange frequency
            print("\nMost Common Buy Exchanges (cheapest):")
            buy_counts = df_opportunities['buy_exchange'].value_counts()
            for exchange, count in buy_counts.items():
                pct = (count / len(df_opportunities)) * 100
                print(f"  {exchange}: {count} times ({pct:.1f}%)")
            
            # Sell exchange frequency
            print("\nMost Common Sell Exchanges (most expensive):")
            sell_counts = df_opportunities['sell_exchange'].value_counts()
            for exchange, count in sell_counts.items():
                pct = (count / len(df_opportunities)) * 100
                print(f"  {exchange}: {count} times ({pct:.1f}%)")

def analyze_volume_liquidity(datasets):
    print("\n" + "="*60)
    print("=== Volume & Liquidity (5) ===")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Volume ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            print(f"\nVolume Statistics During Opportunities:")
            print(f"  Average min_volume: {df_opportunities['min_volume'].mean():.2f}")
            print(f"  Median min_volume: {df_opportunities['min_volume'].median():.2f}")
            print(f"  25th percentile: {df_opportunities['min_volume'].quantile(0.25):.2f}")
            print(f"  75th percentile: {df_opportunities['min_volume'].quantile(0.75):.2f}")
            
            # Volume sufficiency
            volume_thresholds = [10, 50, 100, 500]
            print(f"\nOpportunities by Volume Threshold:")
            for threshold in volume_thresholds:
                count = (df_opportunities['min_volume'] >= threshold).sum()
                pct = (count / len(df_opportunities)) * 100
                print(f"  ≥ {threshold}: {count} opportunities ({pct:.1f}%)")
            
            # Volume ratio analysis
            print(f"\nVolume Ratio (sell/buy) Statistics:")
            print(f"  Average ratio: {df_opportunities['volume_ratio'].mean():.3f}")
            print(f"  Median ratio: {df_opportunities['volume_ratio'].median():.3f}")

def analyze_risk_factors(datasets):
    print("\n" + "="*60)
    print("=== RISK ASSESSMENT (6) ===")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Risk Assessment ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            # Volatility during opportunities
            print(f"\nVolatility During Opportunities:")
            print(f"  Average volatility: {df_opportunities['volatility_avg'].mean():.4f}%")
            print(f"  Median volatility: {df_opportunities['volatility_avg'].median():.4f}%")
            print(f"  Max volatility: {df_opportunities['volatility_avg'].max():.4f}%")
            
            # Opportunity gap (high-low vs close spread)
            print(f"\nOpportunity Gap Analysis:")
            print(f"  Average gap: {df_opportunities['opportunity_gap'].mean():.4f}%")
            print(f"  Median gap: {df_opportunities['opportunity_gap'].median():.4f}%")
            
            # High gap = less realistic to execute at close prices
            high_gap_pct = (df_opportunities['opportunity_gap'] > 0.1).sum() / len(df_opportunities) * 100
            print(f"  Opportunities with gap >0.1%: {high_gap_pct:.1f}%")

def estimate_profitability(datasets):
    print("\n" + "="*60)
    print("=== PROFITABILITY ESTIMATION (7) ===")
    print("="*60 + "\n")
        
    for name, df in datasets.items():
        print(f"\n--- {name} Profit Estimation ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            # Calculate profit per opportunity
            df_opportunities['profit_pct'] = df_opportunities['spread_close_pct'] - TRADING_COST_PCT
            df_opportunities['profit_usd'] = (df_opportunities['profit_pct'] / 100) * TRADE_AMOUNT_USD
            
            total_opportunities = len(df_opportunities)
            total_profit = df_opportunities['profit_usd'].sum()
            avg_profit_per_trade = df_opportunities['profit_usd'].mean()
            
            print(f"\nAssuming ${TRADE_AMOUNT_USD} per trade:")
            print(f"  Total real opportunities: {total_opportunities}")
            print(f"  Total potential profit: ${total_profit:.2f}")
            print(f"  Average profit per trade: ${avg_profit_per_trade:.2f}")
            print(f"  Profit per hour (if traded all): ${total_profit / (len(df) / 60):.2f}")
            
            # Conservative estimate (only high volume opportunities)
            df_high_volume = df_opportunities[df_opportunities['min_volume'] >= 50]
            if len(df_high_volume) > 0:
                conservative_profit = df_high_volume['profit_usd'].sum()
                print(f"\nConservative Estimate (volume ≥ 50):")
                print(f"  Opportunities: {len(df_high_volume)}")
                print(f"  Total profit: ${conservative_profit:.2f}")
                print(f"  Average profit per trade: ${df_high_volume['profit_usd'].mean():.2f}")

#endregion

#region ADVANCED ANALYSIS METHODS
def analyze_momentum_indicators(datasets):
    print("\n" + "="*60)
    print("=== MOMENTUM INDICATORS (8) ===")
    print("="*60 + "\n")
    
    windows = [5, 15, 30]
    
    for name, df in datasets.items():
        print(f"\n--- {name} Momentum Analysis ---")
        
        # Skip NaN rows for rolling features
        df_valid = df.dropna(subset=['spread_ma_5', 'spread_ema_5'])
        df_opportunities = df_valid[df_valid['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            print("\n1. Moving Average Patterns:")
            for window in windows:
                ma_col = f'spread_ma_{window}'
                ema_col = f'spread_ema_{window}'
                
                # Compare current spread vs MA during opportunities
                above_ma = (df_opportunities['spread_close_pct'] > df_opportunities[ma_col]).mean() * 100
                avg_distance_from_ma = (df_opportunities['spread_close_pct'] - df_opportunities[ma_col]).mean()
                
                print(f"\n  Window {window}:")
                print(f"    Opportunities above MA: {above_ma:.1f}%")
                print(f"    Avg distance from MA: {avg_distance_from_ma:.4f}%")
                print(f"    Avg MA during opportunities: {df_opportunities[ma_col].mean():.4f}%")
                print(f"    Avg EMA during opportunities: {df_opportunities[ema_col].mean():.4f}%")
            
            print("\n2. Rate of Change Analysis:")
            if 'spread_rate_change' in df_opportunities.columns:
                print(f"  Average rate of change: {df_opportunities['spread_rate_change'].mean():.4f}%")
                print(f"  Median rate of change: {df_opportunities['spread_rate_change'].median():.4f}%")
                
                # Positive vs negative rate of change
                positive_rate = (df_opportunities['spread_rate_change'] > 0).mean() * 100
                print(f"  Opportunities with positive rate: {positive_rate:.1f}%")
                
                if 'spread_rate_acceleration' in df_opportunities.columns:
                    print(f"  Average acceleration: {df_opportunities['spread_rate_acceleration'].mean():.4f}%")
                    positive_accel = (df_opportunities['spread_rate_acceleration'] > 0).mean() * 100
                    print(f"  Opportunities with positive acceleration: {positive_accel:.1f}%")
            
            print("\n3. Price Change Patterns:")
            if 'price_change_buy_exchange' in df_opportunities.columns:
                print(f"  Avg price change (buy exchange): {df_opportunities['price_change_buy_exchange'].mean():.4f}%")
                print(f"  Avg price change (sell exchange): {df_opportunities['price_change_sell_exchange'].mean():.4f}%")
                
                # Divergence in price movements
                divergence = df_opportunities['price_change_sell_exchange'] - df_opportunities['price_change_buy_exchange']
                print(f"  Avg price divergence: {divergence.mean():.4f}%")

def analyze_bollinger_patterns(datasets):
    print("\n" + "="*60)
    print("=== BOLLINGER BANDS (9) ===)")
    print("="*60 + "\n")
    
    windows = [5, 15, 30]
    
    for name, df in datasets.items():
        print(f"\n--- {name} Bollinger Bands Analysis ---")
        
        df_valid = df.dropna(subset=['spread_ma_5', 'spread_bb_position_5'])
        df_opportunities = df_valid[df_valid['is_real_opportunity'] == 1]
        
        if len(df_opportunities) > 0:
            print("\nBollinger Band Position Analysis:")
            for window in windows:
                position_col = f'spread_bb_position_{window}'
                upper_col = f'spread_bb_upper_{window}'
                lower_col = f'spread_bb_lower_{window}'
                
                if position_col in df_opportunities.columns:
                    avg_position = df_opportunities[position_col].mean()
                    median_position = df_opportunities[position_col].median()
                    
                    # Band breakouts
                    above_upper = (df_opportunities['spread_close_pct'] > df_opportunities[upper_col]).mean() * 100
                    below_lower = (df_opportunities['spread_close_pct'] < df_opportunities[lower_col]).mean() * 100
                    
                    print(f"\n  Window {window}:")
                    print(f"    Avg BB position: {avg_position:.3f} (0=lower band, 1=upper band)")
                    print(f"    Median BB position: {median_position:.3f}")
                    print(f"    Opportunities above upper band: {above_upper:.1f}%")
                    print(f"    Opportunities below lower band: {below_lower:.1f}%")
                    
                    # Band width during opportunities
                    band_width = df_opportunities[upper_col] - df_opportunities[lower_col]
                    print(f"    Avg band width: {band_width.mean():.4f}%")

def analyze_persistence_patterns(datasets):
    print("\n" + "="*60)
    print("=== PERSISTENCE & LAG PATTERNS (10) ===")
    print("="*60 + "\n")
    
    lags = [1, 5, 10, 30]
    windows = [5, 15, 30]
    
    for name, df in datasets.items():
        print(f"\n--- {name} Persistence Analysis ---")
        
        print("\n1. Opportunity Persistence (Autocorrelation):")
        for lag in lags:
            lag_col = f'is_real_opportunity_lag_{lag}'
            if lag_col in df.columns:
                df_valid = df.dropna(subset=[lag_col])
                
                # Conditional probability: P(opportunity now | opportunity lag_N ago)
                had_opportunity_before = df_valid[df_valid[lag_col] == 1]
                if len(had_opportunity_before) > 0:
                    prob_now = had_opportunity_before['is_real_opportunity'].mean() * 100
                    baseline = df_valid['is_real_opportunity'].mean() * 100
                    
                    print(f"  Lag {lag}: {prob_now:.2f}% (baseline: {baseline:.2f}%) - " + 
                          f"{'HIGHER' if prob_now > baseline else 'LOWER'} than baseline")
        
        print("\n2. Rolling Opportunity Counts:")
        df_opportunities = df[df['is_real_opportunity'] == 1]
        if not len(df_opportunities):
            print("  No real opportunities found.")
        else:
            for window in windows:
                count_col = f'real_opportunities_in_last_{window}'
                if count_col in df_opportunities.columns:
                    avg_count = df_opportunities[count_col].mean()
                    max_count = df_opportunities[count_col].max()
                    print(f"  Last {window} min: Avg={avg_count:.2f}, Max={int(max_count)}")
        
        print("\n3. Spread Lag Correlation:")
        for lag in [1, 5, 10]:
            lag_col = f'spread_lag_{lag}'
            if lag_col in df.columns:
                df_valid = df.dropna(subset=[lag_col, 'spread_close_pct'])
                correlation = df_valid['spread_close_pct'].corr(df_valid[lag_col])
                print(f"  Spread vs {lag}-min lag: {correlation:.4f}")
        
        print("\n4. Spread Change Patterns:")
        for lag in [1, 5]:
            diff_col = f'spread_diff_from_lag_{lag}'
            if diff_col in df.columns:
                df_opportunities = df[df['is_real_opportunity'] == 1]
                df_opportunities_valid = df_opportunities.dropna(subset=[diff_col])
                
                if len(df_opportunities_valid) > 0:
                    avg_diff = df_opportunities_valid[diff_col].mean()
                    increasing = (df_opportunities_valid[diff_col] > 0).mean() * 100
                    print(f"  {lag}-min diff during opportunities: {avg_diff:.4f}% " +
                          f"({increasing:.1f}% increasing)")

def analyze_rolling_statistics(datasets):
    print("\n" + "="*60)
    print("=== ROLLING STATISTICS (11) ===)")
    print("="*60 + "\n")
    
    windows = [5, 15, 30]
    
    for name, df in datasets.items():
        print(f"\n--- {name} Rolling Stats Analysis ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        print("\n1. Spread Volatility (Rolling Std):")
        for window in windows:
            std_col = f'spread_rolling_std_{window}'
            if std_col in df_opportunities.columns:
                df_valid = df_opportunities.dropna(subset=[std_col])
                if len(df_valid) > 0:
                    avg_std = df_valid[std_col].mean()
                    median_std = df_valid[std_col].median()
                    print(f"  Window {window}: Avg={avg_std:.4f}%, Median={median_std:.4f}%")
        
        print("\n2. Spread Range (Max - Min):")
        for window in windows:
            range_col = f'spread_range_{window}'
            if range_col in df_opportunities.columns:
                df_valid = df_opportunities.dropna(subset=[range_col])
                if len(df_valid) > 0:
                    avg_range = df_valid[range_col].mean()
                    print(f"  Window {window}: Avg range={avg_range:.4f}%")
        
        print("\n3. Z-Score Analysis:")
        for window in windows:
            zscore_col = f'spread_zscore_{window}'
            if zscore_col in df_opportunities.columns:
                df_valid = df_opportunities.dropna(subset=[zscore_col])
                if len(df_valid) > 0:
                    avg_zscore = df_valid[zscore_col].mean()
                    median_zscore = df_valid[zscore_col].median()
                    
                    # High z-score means unusual spread
                    high_zscore = (df_valid[zscore_col].abs() > 2).mean() * 100
                    
                    print(f"  Window {window}:")
                    print(f"    Avg z-score: {avg_zscore:.3f}, Median: {median_zscore:.3f}")
                    print(f"    Opportunities with |z-score| > 2: {high_zscore:.1f}%")
        
        print("\n4. Volume Rolling Statistics:")
        for window in [5, 15]:
            buy_std_col = f'volume_buy_rolling_std_{window}'
            sell_std_col = f'volume_sell_rolling_std_{window}'
            
            if buy_std_col in df_opportunities.columns:
                df_valid = df_opportunities.dropna(subset=[buy_std_col, sell_std_col])
                if len(df_valid) > 0:
                    avg_buy_std = df_valid[buy_std_col].mean()
                    avg_sell_std = df_valid[sell_std_col].mean()
                    print(f"  Window {window}: Buy Std={avg_buy_std:.2f}, Sell Std={avg_sell_std:.2f}")

def analyze_feature_correlations(datasets):
    print("\n" + "="*60)
    print("=== FEATURE CORRELATION (12) ===)")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Correlation Analysis ---")
        
        # Select numeric features only
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target variables and raw exchange data
        exclude_patterns = ['is_opportunity', 'is_real_opportunity', ':close', ':open', ':high', ':low', ':volume',
                           'min_close', 'max_close', 'min_low', 'max_high', 'time']
        
        features = [col for col in numeric_features if not any(pattern in col for pattern in exclude_patterns)]
        
        # Calculate correlations with target
        if 'is_real_opportunity' in df.columns and len(features) > 0:
            correlations = {}
            for feature in features:
                df_valid = df.dropna(subset=[feature, 'is_real_opportunity'])
                if len(df_valid) > 1:  # Need enough data
                    corr = df_valid[feature].corr(df_valid['is_real_opportunity'])
                    if not np.isnan(corr):
                        correlations[feature] = corr
            
            # Sort by absolute correlation
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print("\nTop 15 Features Correlated with Opportunities:")
            for i, (feature, corr) in enumerate(sorted_corr[:15], 1):
                print(f"  {i:2d}. {feature:40s}: {corr:+.4f}")
            
            print("\nBottom 10 (Least Correlated):")
            for i, (feature, corr) in enumerate(sorted_corr[-10:], 1):
                print(f"  {i:2d}. {feature:40s}: {corr:+.4f}")

def analyze_cross_exchange_ratios(datasets):
    print("\n" + "="*60)
    print("=== CROSS-EXCHANGE PRICE RATIO (13) ===")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Price Ratio Analysis ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if 'price_ratio_buy_sell' in df_opportunities.columns:
            df_valid = df_opportunities.dropna(subset=['price_ratio_buy_sell'])
            
            if len(df_valid) > 0:
                print("\n1. Buy-Sell Price Ratio:")
                avg_ratio = df_valid['price_ratio_buy_sell'].mean()
                median_ratio = df_valid['price_ratio_buy_sell'].median()
                min_ratio = df_valid['price_ratio_buy_sell'].min()
                max_ratio = df_valid['price_ratio_buy_sell'].max()
                
                print(f"  Average: {avg_ratio:.6f}")
                print(f"  Median: {median_ratio:.6f}")
                print(f"  Range: {min_ratio:.6f} - {max_ratio:.6f}")
                print(f"  Spread % = (ratio - 1) * 100: {(avg_ratio - 1) * 100:.4f}%")
        
        if 'avg_price_ratio' in df_opportunities.columns:
            df_valid = df_opportunities.dropna(subset=['avg_price_ratio', 'max_price_ratio', 
                                                       'min_price_ratio', 'price_ratio_std'])
            
            if len(df_valid) > 0:
                print("\n2. Cross-Exchange Ratio Statistics:")
                print(f"  Avg of all ratios: {df_valid['avg_price_ratio'].mean():.6f}")
                print(f"  Avg max ratio: {df_valid['max_price_ratio'].mean():.6f}")
                print(f"  Avg min ratio: {df_valid['min_price_ratio'].mean():.6f}")
                print(f"  Avg std of ratios: {df_valid['price_ratio_std'].mean():.6f}")
                
                # High std = high price dispersion across exchanges
                high_dispersion = (df_valid['price_ratio_std'] > df_valid['price_ratio_std'].median()).mean() * 100
                print(f"  Opportunities with high price dispersion: {high_dispersion:.1f}%")

def analyze_price_position(datasets):
    print("\n" + "="*60)
    print("=== PRICE POSITION ANALYSIS (14) ===")
    print("="*60 + "\n")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Price Position Analysis ---")
        
        df_opportunities = df[df['is_real_opportunity'] == 1]
        
        if 'price_position_buy_exchange' in df_opportunities.columns:
            df_valid = df_opportunities.dropna(subset=['price_position_buy_exchange', 
                                                       'price_position_sell_exchange'])
            
            if len(df_valid) > 0:
                buy_position = df_valid['price_position_buy_exchange'].mean()
                sell_position = df_valid['price_position_sell_exchange'].mean()
                
                print("\nPrice Position (0=low of day, 1=high of day):")
                print(f"  Buy exchange avg position: {buy_position:.3f}")
                print(f"  Sell exchange avg position: {sell_position:.3f}")
                print(f"  Position difference: {sell_position - buy_position:.3f}")
                
                # Categorize positions
                buy_near_low = (df_valid['price_position_buy_exchange'] < 0.3).mean() * 100
                buy_near_high = (df_valid['price_position_buy_exchange'] > 0.7).mean() * 100
                sell_near_low = (df_valid['price_position_sell_exchange'] < 0.3).mean() * 100
                sell_near_high = (df_valid['price_position_sell_exchange'] > 0.7).mean() * 100
                
                print(f"\n  Buy exchange near daily low (<0.3): {buy_near_low:.1f}%")
                print(f"  Buy exchange near daily high (>0.7): {buy_near_high:.1f}%")
                print(f"  Sell exchange near daily low (<0.3): {sell_near_low:.1f}%")
                print(f"  Sell exchange near daily high (>0.7): {sell_near_high:.1f}%")

#endregion

#region ADDITIONAL METHODS
def load_featured_data():
    print("=== LOADING FEATURED DATA ===\n")
        
    datasets = {}
    crypto_currencies = ['BTCUSD', 'ETHUSD', 'DOGEUSD', 'LINKUSD', 'SOLUSD', 'XRPUSD']
    
    for currency in crypto_currencies:
        file_path = f'{FEATURED_DATA_PATH}/featured_{currency}_data.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df['time'])
            datasets[currency] = df
            print(f"✅ Loaded {currency}: {len(df)} rows, {len(df.columns)} columns")
        else:
            print(f"❌ File not found: {file_path}")
    
    print(f"\n✅ Loaded {len(datasets)} datasets\n")
    return datasets

def parse_analysis_selection(user_input, valid_ids):
    if user_input is None:
        return []

    cleaned = user_input.strip()
    if not cleaned:
        return []

    if cleaned == "0":
        return sorted(valid_ids)

    selected = []
    tokens = cleaned.replace(',', ' ').split()
    for token in tokens:
        if not token.isdigit():
            continue
        method_id = int(token)
        if method_id in valid_ids and method_id not in selected:
            selected.append(method_id)

    return selected

def select_and_run_analyses(datasets):
    methods = {
        1: ("Opportunity Frequency", analyze_opportunity_frequency),
        2: ("Spread Overview", analyze_spreads),
        3: ("Temporal Patterns", analyze_temporal_patterns),
        4: ("Exchange Patterns", analyze_exchange_patterns),
        5: ("Volume & Liquidity", analyze_volume_liquidity),
        6: ("Risk Factors", analyze_risk_factors),
        7: ("Profitability Estimation", estimate_profitability),
        8: ("Momentum Indicators", analyze_momentum_indicators),
        9: ("Bollinger Patterns", analyze_bollinger_patterns),
        10: ("Persistence Patterns", analyze_persistence_patterns),
        11: ("Rolling Statistics", analyze_rolling_statistics),
        12: ("Feature Correlations", analyze_feature_correlations),
        13: ("Cross-Exchange Ratios", analyze_cross_exchange_ratios),
        14: ("Price Position", analyze_price_position),
    }

    print("\n=== SELECT ANALYSES ===\n")
    for method_id, (label, _) in methods.items():
        print(f"{method_id:2d}. {label}")
    print("\n0. Run all")

    user_input = input("\nEnter analysis numbers (e.g., 1,3,7): ")
    selected_ids = parse_analysis_selection(user_input, set(methods.keys()))

    if not selected_ids:
        print("\nNo valid analysis numbers were provided.")
        return False

    print("\n=== RUNNING SELECTED ANALYSES ===")
    for method_id in selected_ids:
        label, method = methods[method_id]
        # print(f"\n[{method_id}] {label}")
        method(datasets)

    return True

def should_continue_analysis():
    while True:
        print("\n" + "="*60)
        print("Analysis run finished.")
        choice = input("Type 'c' to continue with another analysis or 'q' to quit: ").strip().lower()

        if choice in {"c", "continue"}:
            return True
        if choice in {"q", "quit", "exit"}:
            return False

        print("Invalid choice. Please enter 'c' or 'q'.")

#endregion

def main():

    datasets = load_featured_data()
    if not datasets:
        print("❌ No featured data found. Run feature engineering first!")
        return

    while True:
        select_and_run_analyses(datasets)
        if not should_continue_analysis():
            print("\nClosing data analysis.")
            break


if __name__ == "__main__":
    main()
