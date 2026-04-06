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

class GRUSpreadModel:
    
    def __init__(self, threshold, gru_units, dense_units, dropout_rate, sequence_length, split_ratio):
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.model = models.Sequential()
        self.gru_units = gru_units
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
        """Create sequences for GRU input"""
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
        """Build GRU model architecture"""
        self.model.add(layers.GRU(self.gru_units, input_shape=(self.sequence_length, n_features)))
        self.model.add(layers.Dropout(self.dropout_rate))
        self.model.add(layers.Dense(self.dense_units, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='mse')
        
        
    def train(self, epochs, batch_size):
        """Train the GRU model"""
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
    parser = argparse.ArgumentParser(description='Train GRU model for spread prediction')
    parser.add_argument('--symbol', type=str, default='BTCUSD', 
                        help='Cryptocurrency symbol (default: BTCUSD)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Opportunity threshold (default: 0.3)')
    parser.add_argument('--gru-units', type=int, default=64,
                        help='Number of GRU units (default: 64)')
    parser.add_argument('--dense-units', type=int, default=32,
                        help='Number of dense units (default: 32)')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Dropout rate for regularization (default: 0.2)')
    return parser.parse_args()
        
def main():
    # Parse CLI arguments
    args = parse_args()
    
    # Hardcoded arguments
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
    print(f"Training GRU for {symbol}")
    print(f"{'='*60}\n")
    
    print(f"Configuration:")
    print(f"  Seed (numPy and TensorFlow): {args.seed}")
    print(f"  Threshold: {args.threshold}")
    print(f"  GRU units: {args.gru_units}")
    print(f"  Dense units: {args.dense_units}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Split ratio (Train part): {split_ratio}\n")
    
    # Initialize model
    model = GRUSpreadModel(threshold=args.threshold,
                            gru_units=args.gru_units,
                            dense_units=args.dense_units,
                            dropout_rate=args.dropout_rate,
                            sequence_length=sequence_length,
                            split_ratio=split_ratio)
    
    # Load and prepare data
    model.load_data(symbol)
    model.prepare_features()
    
    # DEBUG: Check data quality BEFORE sequences
    print(f"\n=== DEBUG: Data Quality Check ===")
    print(f"X_train has NaN: {model.X_train.isna().any().any()}")
    print(f"X_test has NaN: {model.X_test.isna().any().any()}")
    print(f"y_train has NaN: {model.y_train.isna().any()}")
    print(f"y_test has NaN: {model.y_test.isna().any()}")
    print(f"y_train unique values: {model.y_train.unique()}")
    print(f"y_train value counts:\n{model.y_train.value_counts()}")
    print(f"X_train has inf: {np.isinf(model.X_train.values).any()}")
    print(f"X_test has inf: {np.isinf(model.X_test.values).any()}")
    print(f"X_train min: {model.X_train.min().min()}, max: {model.X_train.max().max()}")
    print("=" * 40 + "\n")
    
    # Train the model
    model.train(epochs=epochs, batch_size=batch_size)
    
    # Evaluate
    model.evaluate()
    
    # Make predictions
    y_pred = model.predict(model.X_test_seq).flatten()
    
    # Create output directory
    output_dir = base_path / 'models' / 'ds_model' / 'gru' / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use plotter functions instead of model methods
    plot_results(model.y_test, y_pred, model_name='GRU', 
                 save_path=output_dir / f'gru_{symbol}_results.png')
    plot_prediction_hist(y_pred, model_name='GRU',
                        save_path=output_dir / f'gru_{symbol}_prediction_hist.png')
    plot_training_history(model.history, 
                         model_name='GRU',
                         save_path=output_dir / f'gru_{symbol}_training_history.png')
    
    # Save the model
    model_path = output_dir / f"gru_{symbol}_model.h5"
    model.model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()