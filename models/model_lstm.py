import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import argparse

# Import plotting functions from plotter
from models.plotter import plot_results, plot_prediction_hist, plot_training_history

warnings.filterwarnings('ignore')

class LSTMSpreadModel:
    
    def __init__(self, sequence_length=20, lstm_units=64, dense_units=32, dropout_rate=0.2, random_state=42):
        self.sequence_length = sequence_length
        self.model = models.Sequential()
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_seq = None
        self.X_test_seq = None
        self.feature_names = None
        self.target_name = 'spread_close_pct'
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
        
    def prepare_features(self, exclude_features=None):
        """Prepare features for training"""
        default_exclude = [
            'time', 
            self.target_name,
            'spread_close_absolute',
            'is_opportunity',
            'is_real_opportunity',
            'buy_exchange',
            'sell_exchange',
            'buy_exchange_lag_1',
            'sell_exchange_lag_1',
            'high_exchange',
            'low_exchange',
            'min_close',
            'max_close',
            'price_ratio_buy_sell',
            'opportunity_gap',
            'spread_diff_from_lag_1',
            'spread_diff_from_lag_5',
            'spread_rate_change',
            'spread_rate_change_pct',
            'spread_rate_acceleration'
        ]

        X = self.df.drop(columns=default_exclude, errors='ignore')
        self.feature_names = X.columns.tolist()
        y = self.df[self.target_name]
        
        # 80/20 chronological split
        split_idx = int(len(X) * 0.8)
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
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        
        
    def train(self, epochs=50, batch_size=32):
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
        
        MSE = mean_squared_error(self.y_test, y_pred)
        MAE = mean_absolute_error(self.y_test, y_pred)
        R2 = r2_score(self.y_test, y_pred)
        
        print(f"\nTest Results:")
        print(f"  MSE: {MSE:.6f}")
        print(f"  MAE: {MAE:.6f}")
        print(f"  R² Score: {R2:.4f}")
        
        return MSE, MAE, R2

        
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LSTM model for spread prediction')
    parser.add_argument('--symbol', type=str, default='BTCUSD',
                        help='Cryptocurrency symbol (default: BTCUSD)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Opportunity threshold (default: 0.3)')
    parser.add_argument('--seq-length', type=int, default=20,
                        help='Sequence length for LSTM input (default: 20)')
    parser.add_argument('--units', type=int, default=64,
                        help='Number of LSTM units (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    return parser.parse_args()

        
def main():
    """Main function to train and evaluate the LSTM model"""
    # Parse arguments
    args = parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Get base path
    base_path = Path(__file__).parent.parent
    symbol = args.symbol
    
    print(f"\n{'='*60}")
    print(f"Training LSTM for {symbol}")
    print(f"{'='*60}\n")
    
    print(f"Configuration:")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  LSTM units: {args.units}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}\n")
    
    # Initialize model
    model = LSTMSpreadModel(sequence_length=args.seq_length,
                           lstm_units=args.units,
                           dense_units=32,
                           dropout_rate=0.2,
                           random_state=args.seed)
    
    # Load and prepare data
    model.load_data(symbol)
    model.prepare_features()
    
    # Train the model
    model.train(epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate
    model.evaluate()
    
    # Make predictions
    y_pred = model.predict(model.X_test_seq)
    
    # Create output directory
    output_dir = base_path / 'models' / 'ds_model' / 'lstm' / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use plotter functions instead of model methods
    plot_results(model.y_test, y_pred, model_name='LSTM',
                 save_path=output_dir / f'lstm_{symbol}_results.png')
    plot_prediction_hist(y_pred, model_name='LSTM',
                        save_path=output_dir / f'lstm_{symbol}_prediction_hist.png')
    plot_training_history(model.history.history['loss'],
                         model.history.history.get('val_loss', model.history.history['loss']),
                         model_name='LSTM',
                         save_path=output_dir / f'lstm_{symbol}_training_history.png')
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()









































































