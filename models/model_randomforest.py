import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Ignore warnings
warnings.filterwarnings('ignore')


class RandomForestSpreadModel:
    
    def __init__(self, n_estimators=100, max_depth=None, random_state = 42):
        """"""
        
        
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
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

        X = self.df.drop(columns=default_exclude, errors = 'ignore')
        self.feature_names = X.columns.tolist()
        y = self.df[self.target_name]
        
        split_idx = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_idx]
        self.y_train = y.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_test = y.iloc[split_idx
    
    
