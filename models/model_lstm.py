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
warnings.filterwarnings('ignore')

class LSTMSpreadModel:
    
    def __init__(self, sequence_length=20,lstm_units=64, dense_units=32, dropout_rate=0.2, random_state=42):
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
        
        print("hello")