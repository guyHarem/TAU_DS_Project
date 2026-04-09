"""LSTM regression model for predicting cryptocurrency spread movements.

This module implements an LSTM (Long Short-Term Memory) neural network to predict
the next-minute spread between buy and sell prices across exchanges (is_real_opportunity).

Architecture:
    - Input: Temporal sequences of features (shape: [seq_length, num_features])
    - LSTM layer: Learns long-range temporal dependencies
    - Dropout: Regularization to prevent overfitting
    - Dense layers: Feature transformation and output scaling
    - Output: Sigmoid-activated spread prediction [0, 1]

Data Pipeline:
    1. Load featured data from CSV
    2. Prepare features (exclude time, exchanges, target)
    3. Create rolling window sequences
    4. Scale features using MinMaxScaler [0, 1]
    5. Build and train recurrent model
    6. Generate predictions and visualizations

Key Hyperparameters:
    - lstm_units: Size of LSTM hidden state (default: 64)
    - dense_units: Size of fully-connected layers (default: 32)
    - dropout_rate: Dropout probability (default: 0.2)
    - sequence_length: Temporal window size (default: 10)
    - split_ratio: Train/test chronological split (default: 0.6)

Output:
    - Predictions on test set
    - Training history (loss curves)
    - Visualizations via plotter module

Typical Usage:
    model = LSTMSpreadModel(lstm_units=64, dropout_rate=0.2)
    X_train, X_test, y_train, y_test = model.prepare_features(df)
    history = model.train(X_train_scaled, y_train_scaled)
    y_pred = model.predict(X_test_scaled)

Author: TAU DS Project | Arbitrage team
Date: 2026
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import argparse

# Import plotting functions from plotter
from plotter import plot_results, plot_prediction_hist, plot_training_history

#Surpress warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LSTMSpreadModel:
    
    def __init__(self, threshold, lstm_units, dense_units, dropout_rate, sequence_length, split_ratio):
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.model = models.Sequential()
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.split_ratio = split_ratio
        self.scaler = MinMaxScaler()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_seq = None
        self.X_test_seq = None
        self.feature_names = None
        self.target_name = 'is_real_opportunity'
        self.is_fitted = False
        self.history = None
        
    def load_data(self, symbol):
        """Load featured data from CSV file"""
        base_path = Path(__file__).parent.parent
        data_path = base_path / 'data' / 'featured_data'
        file_path = data_path / f'featured_{symbol}_data.csv'
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
    def prepare_features(self):
        """Prepare features for training"""
        default_exclude = [
            'time',
            'buy_exchange',
            'sell_exchange',
            'high_exchange',
            'low_exchange',
            'num_exchanges_available',
            'buy_exchange_lag_1',
            'sell_exchange_lag_1',
            self.target_name

        ]
        
        # Drop exclude features
        X = self.df.drop(columns=default_exclude, errors='ignore')
        self.feature_names = X.columns.tolist()
        
        # Define target name as next minute prediction
        y = self.df[self.target_name].shift(-1)
        
        # drop null rows created by shift(-1) AND any NaN in X
        mask = y.notna() & X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
        
        # split_ratio chronological split
        split_idx = int(len(X) * self.split_ratio)
        self.X_train = X.iloc[:split_idx]
        self.y_train = y.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_test = y.iloc[split_idx:]
        
        print(f"\nFeatures prepared: {len(self.feature_names)} features")
        print(f"Train samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        
        
    def create_sequences(self, X, y):
        """Create sequences for LSTM input"""
        X_array = X.to_numpy()
        y_array = y.to_numpy()
        X_seq = list()
        y_seq = list()
        
        for i in range(len(X_array) - self.sequence_length):
            X_seq.append(X_array[i:i+self.sequence_length])
            y_seq.append(y_array[i+self.sequence_length])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        return X_seq, y_seq
    
    
    def scale_features(self, X_train_seq, X_test_seq):
        """Scale features using MinMaxScaler"""
        n_train, seq_len, n_features = X_train_seq.shape
        n_test = X_test_seq.shape[0]
        
        X_train_seq = X_train_seq.reshape(-1, n_features)
        X_test_seq = X_test_seq.reshape(-1, n_features)
                                       
        self.scaler.fit(X_train_seq)
        
        X_train_seq = self.scaler.transform(X_train_seq)
        X_test_seq = self.scaler.transform(X_test_seq)
        
        X_train_seq = X_train_seq.reshape(n_train, seq_len, n_features)
        X_test_seq = X_test_seq.reshape(n_test, seq_len, n_features)
        
        return X_train_seq, X_test_seq
    
    
    def build_model(self, n_features):
        """Build LSTM model architecture"""
        self.model.add(layers.LSTM(self.lstm_units, input_shape=(self.sequence_length, n_features)))
        self.model.add(layers.Dropout(self.dropout_rate))
        self.model.add(layers.Dense(self.dense_units, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='mse')
        
        
    def train(self, epochs, batch_size):
        """Train the LSTM model"""
        # Step 1 - Create sequences
        print("Creating sequences...")
        self.X_train_seq, self.y_train = self.create_sequences(self.X_train, self.y_train)
        self.X_test_seq, self.y_test = self.create_sequences(self.X_test, self.y_test)
        
        print(f"Train sequences: {len(self.X_train_seq)}")
        print(f"Test sequences: {len(self.X_test_seq)}")
        
        # Step 2 - Scale features
        print("Scaling features...")
        self.X_train_seq, self.X_test_seq = self.scale_features(self.X_train_seq, self.X_test_seq)
        
        # Step 3 - Build model
        print("Building model...")
        self.build_model(len(self.feature_names))
        
        # Step 4 - Train the model
        print("Training model...")
        self.history = self.model.fit(self.X_train_seq, self.y_train, 
                                      epochs=epochs, batch_size=batch_size, 
                                      validation_data=(self.X_test_seq, self.y_test), 
                                      verbose=1)
        
        self.is_fitted = True
        
        
    def predict(self, X_seq):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet, train it before running predictions.")
        
        predictions = self.model.predict(X_seq)
        return predictions
    
    
    def evaluate(self):
        """Evaluate the model"""
        y_pred = self.predict(self.X_test_seq)
        y_pred_binary = (y_pred > self.threshold).astype(int)
        
        
        MSE = mean_squared_error(self.y_test, y_pred)
        MAE = mean_absolute_error(self.y_test, y_pred)
        R2 = r2_score(self.y_test, y_pred)
        F1 = f1_score(self.y_test, y_pred_binary)
        
        print(f"\nTest Results:")
        print(f"  MSE: {MSE:.6f}")
        print(f"  MAE: {MAE:.6f}")
        print(f"  R² Score: {R2:.4f}")
        print(f"  F1 Score: {F1:.4f}")
        print(f"  Threshold: {self.threshold}")
        
        return MSE, MAE, R2, F1

        
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LSTM model for spread prediction')
    parser.add_argument('--symbol', type=str, default='BTCUSD',
                        help='Cryptocurrency symbol (default: BTCUSD)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Opportunity threshold (default: 0.3)')
    parser.add_argument('--lstm-units', type=int, default=64,
                        help='Number of LSTM units (default: 64)')
    parser.add_argument('--dense-units', type=int, default=32,
                        help='Number of dense units (default: 32)')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Dropout rate for regularization (default: 0.2)')
    return parser.parse_args()

        
def main():
    """Main function to train and evaluate the LSTM model"""
    
    #Parse CLI arguments
    args = parse_args()
    
    #Hardcoded params
    sequence_length = 20
    batch_size = 32
    epochs = 50
    split_ratio = 0.6

    
    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Get base path
    base_path = Path(__file__).parent.parent
    
    # Get symbol
    symbol = args.symbol
    
    print(f"\n{'='*60}")
    print(f"Training LSTM for {symbol}")
    print(f"{'='*60}\n")
    
    print(f"Configuration:")
    print(f"  Seed (numPy and TensorFlow): {args.seed}")
    print(f"  Threshold: {args.threshold}")
    print(f"  LSTM units: {args.lstm_units}")
    print(f"  Dense units: {args.dense_units}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Split ratio (Train part): {split_ratio}\n")
    
    # Initialize model
    model = LSTMSpreadModel(threshold = args.threshold,
                            lstm_units=args.lstm_units,
                            dense_units=args.dense_units,
                            dropout_rate=args.dropout_rate,
                            sequence_length=sequence_length,
                            split_ratio=split_ratio
                            )
    
    # Load and prepare data
    model.load_data(symbol)
    model.prepare_features()
    
    # Train the model
    model.train(epochs=epochs, batch_size=batch_size)
    
    # Evaluate
    model.evaluate()
    
    # Make predictions
    y_pred = model.predict(model.X_test_seq).flatten()
    
    # Create output directory
    output_dir = base_path / 'models' / 'ds_model' / 'lstm' / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use plotter functions instead of model methods
    plot_results(model.y_test, y_pred, model_name='LSTM',
                 save_path=output_dir / f'lstm_{symbol}_results.png')
    plot_prediction_hist(y_pred, model_name='LSTM',
                        save_path=output_dir / f'lstm_{symbol}_prediction_hist.png')
    plot_training_history(model.history,
                         model_name='LSTM',
                         save_path=output_dir / f'lstm_{symbol}_training_history.png')
    
    # Save the model
    model_path = output_dir / f"lstm_{symbol}_model.h5"
    model.model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
