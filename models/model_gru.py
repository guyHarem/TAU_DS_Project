import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import argparse
warnings.filterwarnings('ignore')

class GRUSpreadModel:
    
    def __init__(self, sequence_length=20,gru_units=64, dense_units=32, dropout_rate=0.2, random_state=42):
        self.sequence_length = sequence_length
        self.model = models.Sequential()
        self.gru_units = gru_units
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
        
        
    def load_data(self,symbol):
        
        base_path = Path(__file__).parent.parent
        data_path = base_path / 'data' / 'featured_data'
        file_path = data_path / f'featured_{symbol}_data.csv'
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path)
        
    def prepare_features(self, exclude_features=None):
        
        default_exclude = [
        'time', 
        self.target_name,
        'spread_close_absolute',  # Direct calculation from target
        'is_opportunity',  # Target-related
        'is_opportunity_flag',  # Target-related
        'is_real_opportunity',  # Target-related
        'buy_exchange',  # Categorical
        'sell_exchange',  # Categorical
        'buy_exchange_lag_1',  # Categorical
        'sell_exchange_lag_1',  # Categorical
        'high_exchange',  # Categorical
        'low_exchange',  # Categorical
        'min_close',  # Used to calculate target
        'max_close',  # Used to calculate target
        'price_ratio_buy_sell',  # Directly related to spread
        'opportunity_gap',  # Directly related to spread
            # These features are derived from spread and cause perfect prediction
            'spread_diff_from_lag_1',  # Derived from spread
            'spread_diff_from_lag_5',  # Derived from spread
            'spread_rate_change',  # Derived from spread
            'spread_rate_change_pct',  # Derived from spread
            'spread_rate_acceleration'  # Derived from spread
        ]

        # Verify labels
        X = self.df.drop(columns=default_exclude, errors = 'ignore')
        self.feature_names = X.columns.tolist()
        y = self.df[self.target_name]
        
        
        # 80/20 split
        split_idx = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_idx]
        self.y_train = y.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_test = y.iloc[split_idx:]
        
        
    def create_sequences(self, X, y):
        
        # Declaring new variables and empty lists
        X_array = X.to_numpy()
        y_array = y.to_numpy()
        X_seq = list()
        y_seq = list()
        
        # Iterating the numpy arrays to add to the sequences 
        for i in range(len(X_array) - self.sequence_length):
            X_seq.append(X_array[i:i+self.sequence_length])
            y_seq.append(y_array[i+self.sequence_length])
            
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        return X_seq, y_seq
    
    
    def scale_features(self, X_train_seq, X_test_seq):
        
        n_train, seq_len, n_features = X_train_seq.shape
        
        n_test = X_test_seq.shape[0]
        
        X_train_seq = X_train_seq.reshape(-1,n_features)
        
        X_test_seq = X_test_seq.reshape(-1,n_features)
                                       
        self.scaler.fit(X_train_seq)
        
        X_train_seq = self.scaler.transform(X_train_seq)
        
        X_test_seq = self.scaler.transform(X_test_seq)
        
        X_train_seq = X_train_seq.reshape(n_train,seq_len, n_features)
        
        X_test_seq = X_test_seq.reshape(n_test, seq_len, n_features)
        
        return X_train_seq, X_test_seq
    
    
    
    def build_model(self, n_features):
        
        self.model.add(layers.GRU(self.gru_units, input_shape = (self.sequence_length, n_features)))
        
        self.model.add(layers.Dropout(self.dropout_rate))
        
        self.model.add(layers.Dense(self.dense_units, activation = 'relu'))
        
        self.model.add(layers.Dense(1))
        
        self.model.compile(optimizer='adam', loss='mse')
        
        
        
    def train(self, epochs=50, batch_size=32):
        
        
        # Step 1 - Create sequences
        print("Creating sequences...")
        self.X_train_seq, self.y_train = self.create_sequences(self.X_train, self.y_train)
        self.X_test_seq, self.y_test = self.create_sequences(self.X_test, self.y_test)
        
        # Step 2 - Scale features
        print("Scaling features...")
        self.X_train_seq, self.X_test_seq = self.scale_features(self.X_train_seq, self.X_test_seq)
        
        # Step 3 - build model
        print("Building model...")
        self.build_model(len(self.feature_names))
        
        # Step 4 - Fit(Train) the model
        print("Training model...")
        self.history = self.model.fit(self.X_train_seq, self.y_train, epochs=epochs, batch_size=batch_size, validation_data = (self.X_test_seq, self.y_test), verbose=1)
        
        # Step 4.1 - Mark as fitted
        self.is_fitted = True
        
        
    def predict(self, X_seq):
        
        if self.is_fitted == False:
            raise ValueError("Model not fitted yet, train it before running predictions.")
        
        predictions = self.model.predict(X_seq)
        return predictions
    
    
    def evaluate(self):
        
        y_pred = self.predict(self.X_test_seq)
        
        MSE = mean_squared_error(self.y_test, y_pred)
        MAE = mean_absolute_error(self.y_test, y_pred)
        R2 = r2_score(self.y_test, y_pred)
        
        print(f"MSE: {MSE:.4f}")
        print(f"MAE: {MAE:.4f}")
        print(f"R² Score: {R2:.4f}")
        
        return MSE, MAE, R2
    
    
    def plot_results(self, y_test, y_pred, save_path = None):
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        residuals = y_test - y_pred
        abs_error = np.abs(residuals)

        #axes[0, 0] - Actual Vs predicted 
        axes[0 ,0].scatter(y_test, y_pred, alpha = 0.5, s = 10)
        axes[0 ,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw = 2, label = 'Perfect Prediction')
        axes[0 ,0].set_xlabel('Actual spread_close_pct', fontsize = 12)
        axes[0 ,0].set_ylabel('Predicted spread_close_pct', fontsize = 12)
        axes[0 ,0].set_title('Actual vs Predicted', fontsize = 14, fontweight = 'bold')
        axes[0 ,0].legend()
        axes[0 ,0].grid(True, alpha = 0.3)
        
        #axes[0, 1] - Residuals
        axes[0 ,1].scatter(y_pred, residuals, alpha = 0.5, s = 10)
        axes[0 ,1].axhline(y = 0, color = 'r', linestyle = '--', lw = 2)
        axes[0 ,1].set_xlabel('Predicted spread_close_pct', fontsize = 12)
        axes[0 ,1].set_ylabel('Residuals', fontsize = 12)
        axes[0 ,1].set_title('Residual Plot', fontsize = 14, fontweight = 'bold')
        axes[0 ,1].grid(True, alpha = 0.3)
        
        #axes[1, 0] - Residual Distribution
        axes[1 ,0].hist(residuals, bins = 50, edgecolor = 'black', alpha = 0.7)
        axes[1 ,0].axvline(x = 0, color = 'r', linestyle = '--', lw = 2)
        axes[1 ,0].set_xlabel('Residuals', fontsize = 12)
        axes[1 ,0].set_ylabel('Frequency', fontsize = 12)
        axes[1 ,0].set_title('Residual Distribution', fontsize = 14, fontweight = 'bold')
        axes[1 ,0].grid(True, alpha = 0.3)
        
        #axes[1, 1] - Absolute Error Distribution
        axes[1 ,1].hist(abs_error, bins = 50, edgecolor = 'black', alpha = 0.7, color = 'orange')
        axes[1 ,1].set_xlabel('Absolute Error', fontsize = 12)
        axes[1 ,1].set_ylabel('Frequency', fontsize = 12)
        axes[1 ,1].set_title('Absolute Error Distribution', fontsize = 14, fontweight = 'bold')
        axes[1 ,1].grid(True, alpha = 0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
            print(f"Results plot saved to {save_path}")
        else:
            plt.show()  
            
        plt.close()
        
        
    def plot_prediction_hist(self, y_pred, save_path = None):

        plt.figure(figsize = (10, 6))
        plt.hist(y_pred, bins = 40, edgecolor = 'black', alpha = 0.75)
        plt.xlabel('Predicted spread_close_pct', fontsize = 12)
        plt.ylabel('Frequency', fontsize = 12)
        plt.title('Prediction Histogram', fontsize = 14, fontweight = 'bold')
        plt.grid(True, alpha = 0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
            print(f"Prediction histogram saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        
    def plot_training_history(self, save_path=None):
        
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        plt.figure(figsize=(10,6))
        plt.plot(train_loss, label= 'Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches = 'tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        
        
def args_parse():
    
    parser = argparse.ArgumentParser(description='Train GRU model for spread prediction')
    parser.add_argument('--crypto', type=str, default='BTCUSD', help='Cryptocurreny symbol (default: BTUSD)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default:42)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default:50)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default:32)')
    return parser.parse_args()
        
def main():
    
    # Parse arguments
    args = args_parse()
    
    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Get base path
    base_path = Path(__file__).parent.parent
    
    # Get crypto symbol
    crypto = args.crypto
    
    print(f"\n{'='*60}")
    print(f"Training GRU for {crypto}")
    print(f"{'='*60}\n")
    
    # Declare model
    model = GRUSpreadModel(sequence_length=20, gru_units=64, dense_units=32, dropout_rate=0.2, random_state=args.seed)
    
    # Load and prepare model
    model.load_data(crypto)
    model.prepare_features()
    
    # Train the model
    model.train(epochs=args.epochs, batch_size= args.batch_size)
    
    # Evaluate
    model.evaluate()
    
    # Predictions
    y_pred = model.predict(model.X_test_seq)
    
    # Create output directory
    output_dir = base_path / 'models' / 'ds_model' / 'gru' / crypto
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot
    model.plot_results(model.y_test, y_pred, save_path=output_dir / f'gru_{crypto}_results.png')
    model.plot_prediction_hist(y_pred, save_path=output_dir / f'gru_{crypto}_prediction_hist.png')
    model.plot_training_history(save_path=output_dir / f'gru_{crypto}_training_history.png')
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_dir}")
    print("="*60)
    
    
      
    
    
    
    
    
    
    
    

                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    
        
        
        
    
        
    
        
        
        
        
































