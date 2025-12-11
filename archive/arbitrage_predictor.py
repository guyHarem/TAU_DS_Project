import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data(csv_file):
    """Load the combined cryptocurrency data from CSV"""
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    return df

def engineer_features(df, exchange1='BINANCE', exchange2='BITFINEX', window_sizes=[5, 10, 20]):
    """
    Create features for predicting arbitrage opportunities
    
    Features:
    - Price difference between exchanges
    - Moving averages of prices
    - Price volatility
    - Rolling statistics
    - Time-based features
    """
    col1_high = f'{exchange1}:high'
    col2_high = f'{exchange2}:high'
    col1_low = f'{exchange1}:low'
    col2_low = f'{exchange2}:low'
    col1_close = f'{exchange1}:close'
    col2_close = f'{exchange2}:close'
    col1_volume = f'{exchange1}:volume'
    col2_volume = f'{exchange2}:volume'
    
    # Filter rows with both exchanges
    df_filtered = df[[
        'time', col1_high, col2_high, col1_low, col2_low, 
        col1_close, col2_close, col1_volume, col2_volume
    ]].dropna().copy()
    
    print(f"Data points with both {exchange1} and {exchange2}: {len(df_filtered)}")
    
    # Basic price difference features
    df_filtered['price_diff'] = df_filtered[col1_high] - df_filtered[col2_high]
    df_filtered['price_diff_pct'] = (df_filtered['price_diff'] / df_filtered[col2_high]) * 100
    
    # Spread features (high - low)
    df_filtered[f'{exchange1}_spread'] = df_filtered[col1_high] - df_filtered[col1_low]
    df_filtered[f'{exchange2}_spread'] = df_filtered[col2_high] - df_filtered[col2_low]
    df_filtered['spread_diff'] = df_filtered[f'{exchange1}_spread'] - df_filtered[f'{exchange2}_spread']
    
    # Volume ratio
    df_filtered['volume_ratio'] = df_filtered[col1_volume] / (df_filtered[col2_volume] + 1e-10)
    
    # Moving averages and rolling statistics
    for window in window_sizes:
        # Price moving averages
        df_filtered[f'{exchange1}_ma_{window}'] = df_filtered[col1_close].rolling(window=window).mean()
        df_filtered[f'{exchange2}_ma_{window}'] = df_filtered[col2_close].rolling(window=window).mean()
        
        # Price difference moving average
        df_filtered[f'price_diff_ma_{window}'] = df_filtered['price_diff'].rolling(window=window).mean()
        
        # Volatility (standard deviation)
        df_filtered[f'{exchange1}_volatility_{window}'] = df_filtered[col1_close].rolling(window=window).std()
        df_filtered[f'{exchange2}_volatility_{window}'] = df_filtered[col2_close].rolling(window=window).std()
        
        # Price momentum (rate of change)
        df_filtered[f'{exchange1}_momentum_{window}'] = df_filtered[col1_close].pct_change(window)
        df_filtered[f'{exchange2}_momentum_{window}'] = df_filtered[col2_close].pct_change(window)
    
    # Time-based features
    df_filtered['hour'] = df_filtered['time'].dt.hour
    df_filtered['minute'] = df_filtered['time'].dt.minute
    df_filtered['day_of_week'] = df_filtered['time'].dt.dayofweek
    
    # Target variable: Use STRICT threshold based on profitability
    # Trading fees: ~0.1% per trade x 2 trades = 0.2% total
    # Slippage and other costs: ~0.1%
    # Minimum profit target: 0.5% (to make it worthwhile)
    # Total threshold: 0.2% + 0.1% + 0.5% = 0.8%
    
    min_profit_threshold = 0.8  # Must be at least 0.8% difference to be profitable
    
    abs_price_diff_pct = abs(df_filtered['price_diff_pct'])
    df_filtered['arbitrage_opportunity'] = (abs_price_diff_pct >= min_profit_threshold).astype(int)
    
    # Drop NaN values created by rolling windows
    df_filtered = df_filtered.dropna()
    
    print(f"Data points after feature engineering: {len(df_filtered)}")
    print(f"Arbitrage opportunities (>={min_profit_threshold}% difference): {df_filtered['arbitrage_opportunity'].sum()} ({df_filtered['arbitrage_opportunity'].mean()*100:.2f}%)")
    print(f"Threshold used: {min_profit_threshold}% price difference")
    
    # Show some statistics about price differences
    print(f"\nPrice Difference Statistics:")
    print(f"  Mean: {df_filtered['price_diff_pct'].mean():.4f}%")
    print(f"  Std: {df_filtered['price_diff_pct'].std():.4f}%")
    print(f"  Min: {df_filtered['price_diff_pct'].min():.4f}%")
    print(f"  Max: {df_filtered['price_diff_pct'].max():.4f}%")
    print(f"  Median: {df_filtered['price_diff_pct'].median():.4f}%")
    print(f"  75th percentile: {abs_price_diff_pct.quantile(0.75):.4f}%")
    print(f"  90th percentile: {abs_price_diff_pct.quantile(0.90):.4f}%")
    print(f"  95th percentile: {abs_price_diff_pct.quantile(0.95):.4f}%")
    print(f"  99th percentile: {abs_price_diff_pct.quantile(0.99):.4f}%")
    
    return df_filtered

def train_model(df_filtered, test_size=0.2, random_state=42):
    """
    Train a Random Forest Classifier to predict arbitrage opportunities
    
    Why Random Forest?
    - Handles non-linear relationships well
    - Robust to outliers
    - Can capture complex patterns in price movements
    - Provides feature importance rankings
    - Works well with time-series features
    """
    # Separate features and target
    feature_cols = [col for col in df_filtered.columns 
                   if col not in ['time', 'arbitrage_opportunity', 'price_diff', 'price_diff_pct']]
    
    X = df_filtered[feature_cols]
    y = df_filtered['arbitrage_opportunity']
    
    print(f"\nFeatures used: {len(feature_cols)}")
    
    # Check class distribution
    print(f"\nClass Distribution:")
    print(f"  No Opportunity (0): {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%)")
    print(f"  Opportunity (1): {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")
    
    # Check if we have enough samples of both classes
    if (y == 1).sum() < 10:
        print("\n⚠️  WARNING: Very few arbitrage opportunities found!")
        print(f"   This means price differences between exchanges are very small.")
        print("   Consider:")
        print("   - Using more data (longer time period)")
        print("   - Trying different exchange pairs")
        print("   - Lowering the threshold (but be aware of fees)")
        return None, None, None, None, None, None
    
    # Use stratified split to ensure both classes in test set
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as e:
        print(f"\n⚠️  ERROR: Cannot split data - {e}")
        print("   Not enough samples in minority class for stratified split.")
        return None, None, None, None, None, None
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training set arbitrage rate: {y_train.mean()*100:.2f}%")
    print(f"Test set arbitrage rate: {y_test.mean()*100:.2f}%")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Classifier
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*70)
    
    model = RandomForestClassifier(
        n_estimators=100,        # Number of trees
        max_depth=10,            # Maximum depth of trees
        min_samples_split=10,    # Minimum samples to split a node
        min_samples_leaf=5,      # Minimum samples in a leaf
        random_state=random_state,
        n_jobs=-1,               # Use all CPU cores
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Prediction probabilities
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluation
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    print("\nTraining Set Performance:")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    
    print("\nTest Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['No Opportunity', 'Opportunity'], zero_division=0))
    
    # Cross-validation
    print("\nRunning 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'feature_importance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFeature importance plot saved to: {output_file}")
    plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Opportunity', 'Opportunity'],
                yticklabels=['No Opportunity', 'Opportunity'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'confusion_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_file}")
    plt.show()
    
    # Show prediction probabilities distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[:, 1], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Predicted Probability of Arbitrage Opportunity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)')
    plt.legend()
    plt.tight_layout()
    
    output_file = Path(__file__).parent / 'probability_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Probability distribution saved to: {output_file}")
    plt.show()
    
    # Save model and scaler
    model_file = Path(__file__).parent / 'arbitrage_model.pkl'
    scaler_file = Path(__file__).parent / 'scaler.pkl'
    
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(feature_cols, Path(__file__).parent / 'feature_cols.pkl')
    
    print(f"\nModel saved to: {model_file}")
    print(f"Scaler saved to: {scaler_file}")
    
    return model, scaler, feature_cols, X_test, y_test, y_pred_test

def main():
    print("="*70)
    print("ARBITRAGE OPPORTUNITY PREDICTOR")
    print("="*70)
    print("\nModel: Random Forest Classifier")
    print("Exchanges: BINANCE vs BITFINEX")
    print("\nProfit Threshold: 0.8%")
    print("  - Trading fees: 0.2% (0.1% x 2 trades)")
    print("  - Slippage: ~0.1%")
    print("  - Minimum profit: 0.5%")
    print("  = Total threshold: 0.8%")
    print("\nWhy Random Forest?")
    print("  - Excellent for non-linear relationships in financial data")
    print("  - Handles multiple features and their interactions")
    print("  - Resistant to overfitting with proper parameters")
    print("  - Provides feature importance for interpretability")
    print("  - No assumptions about data distribution")
    print("="*70)
    
    # Load data
    csv_file = Path(__file__).parent.parent / 'data_retrieve' / 'combined_BTCUSD_data.csv'
    
    if not csv_file.exists():
        print(f"\nError: CSV file not found at {csv_file}")
        print("Please run data_retrieve.py first to generate the data.")
        return
    
    print(f"\nLoading data from: {csv_file}")
    df = load_data(csv_file)
    
    print(f"Total records: {len(df)}")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    
    # Engineer features
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    df_filtered = engineer_features(df, 'BINANCE', 'BITFINEX', window_sizes=[5, 10, 20])
    
    # Train model
    result = train_model(df_filtered)
    if result[0] is None:
        print("\n⚠️  Model training stopped due to insufficient data.")
        print("   Please collect more data or try different parameters.")
        return
    
    model, scaler, feature_cols, X_test, y_test, y_pred_test = result
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nThe model predicts opportunities with >= 0.8% price difference.")
    print("This ensures profitability after accounting for:")
    print("  - Trading fees (0.2%)")
    print("  - Slippage (0.1%)")
    print("  - Minimum profit target (0.5%)")
    print("\nThe model uses these features to predict:")
    print("  - Current price differences between BINANCE and BITFINEX")
    print("  - Price trends and momentum")
    print("  - Volatility patterns")
    print("  - Volume ratios")
    print("  - Time of day")
    print("\nUse this model to decide when to execute arbitrage trades!")

if __name__ == "__main__":
    main()