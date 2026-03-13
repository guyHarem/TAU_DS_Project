"""Train a Transformer regression model to predict `spread_close_pct`.

Usage (example):
    python models/model_transformer.py --symbol BTCUSD --seq-length 60 --seed 42
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import plotting functions from plotter
from models.plotter import (
    plot_results,
    plot_prediction_hist,
    plot_training_history
)

warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """Dataset for time series prediction with transformer"""
    
    def __init__(self, features, targets, seq_length):
        """
        Parameters:
        -----------
        features : np.array
            Feature array (n_samples, n_features)
        targets : np.array
            Target array (n_samples,)
        seq_length : int
            Length of input sequences
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return X, y


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """Transformer model for spread prediction"""
    
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()


class TransformerModel:
    """Wrapper class for transformer model training and evaluation"""
    
    def __init__(self, seq_length=60, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.1, learning_rate=0.001,
                 batch_size=32, epochs=50, random_state=42):
        self.seq_length = seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = StandardScaler()
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
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def prepare_features(self, df, exclude_features=None):
        """Prepare features for training"""
        default_exclude = [
            'time', 
            self.target_name,
            'spread_close_absolute',
            'is_opportunity',
            'is_opportunity_flag',
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
            'spread_rate_acceleration',
        ]
        
        if exclude_features:
            default_exclude.extend(exclude_features)
        
        exclude_cols = list(set(default_exclude))
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        df_clean = df[feature_cols + [self.target_name]].copy()
        df_clean = df_clean.dropna(subset=[self.target_name])
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        for col in feature_cols:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
        
        df_clean = df_clean.dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean[self.target_name].values
        
        self.feature_names = feature_cols
        
        print(f"\nFeatures prepared: {len(feature_cols)} features")
        print(f"Samples: {X.shape[0]}")
        print(f"Target range: [{y.min():.6f}, {y.max():.6f}]")
        
        return X, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the transformer model"""
        print(f"\nTraining Transformer model...")
        print(f"Configuration:")
        print(f"  Sequence length: {self.seq_length}")
        print(f"  Model dimension: {self.d_model}")
        print(f"  Attention heads: {self.nhead}")
        print(f"  Encoder layers: {self.num_layers}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}\n")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train, self.seq_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_dataset = TimeSeriesDataset(X_val_scaled, y_val, self.seq_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None
        
        n_features = X_train.shape[1]
        self.model = TransformerPredictor(
            n_features=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.6f}")
        
        self.is_fitted = True
        self.history = {'train_loss': train_losses, 'val_loss': val_losses}
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        dataset = TimeSeriesDataset(X_scaled, np.zeros(len(X)), self.seq_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print("\nEvaluating model...")
        
        y_pred = self.predict(X_test)
        y_test_aligned = y_test[self.seq_length:]
        
        metrics = {
            'mse': mean_squared_error(y_test_aligned, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_aligned, y_pred)),
            'mae': mean_absolute_error(y_test_aligned, y_pred),
            'r2': r2_score(y_test_aligned, y_pred),
            'mape': np.mean(np.abs((y_test_aligned - y_pred) / (y_test_aligned + 1e-9))) * 100,
        }
        
        print(f"Test R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        
        return metrics, y_pred, y_test_aligned


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Transformer model for crypto spread prediction')
    parser.add_argument('--symbol', type=str, default='BTCUSD',
                        help='Cryptocurrency symbol (default: BTCUSD)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Opportunity threshold (default: 0.3)')
    parser.add_argument('--seq-length', type=int, default=60,
                        help='Sequence length for transformer input (default: 60)')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Dimension of model embeddings (default: 128)')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of transformer layers (default: 3)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    return parser.parse_args()


def main():
    """Main function to train and evaluate the transformer model"""
    args = parse_args()
    
    symbol = args.symbol
    seed = args.seed
    threshold = args.threshold
    seq_length = args.seq_length
    d_model = args.d_model
    nhead = args.nhead
    num_layers = args.num_layers
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    base_path = Path(__file__).parent.parent
    
    print(f"\n{'='*60}")
    print(f"Training Transformer for {symbol}")
    print(f"{'='*60}\n")
    
    # Initialize model
    model = TransformerModel(
        seq_length=seq_length,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=lr,
        random_state=seed
    )
    
    # Load data
    df = model.load_data(symbol)
    
    # Prepare features
    X, y = model.prepare_features(df)
    
    # Chronological split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Further split train into train/val
    val_split_idx = int(len(X_train) * 0.9)
    X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    
    print(f"\nTrain set size: {X_train_final.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model.train(X_train_final, y_train_final, X_val, y_val)
    
    # Evaluate model
    metrics, y_pred, y_test_aligned = model.evaluate(X_test, y_test)
    
    # Output directory
    output_path = base_path / 'models' / 'ds_model' / 'transformer' / symbol
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output_path}")
    
    # Use plotter functions for plots
    plot_training_history(
        model.history['train_loss'],
        model.history['val_loss'],
        model_name='Transformer',
        save_path=output_path / f'transformer_{symbol}_training_history.png'
    )
    
    plot_results(
        y_test_aligned,
        y_pred,
        model_name='Transformer',
        save_path=output_path / f'transformer_{symbol}_results.png'
    )
    
    plot_prediction_hist(
        y_pred,
        model_name='Transformer',
        save_path=output_path / f'transformer_{symbol}_prediction_hist.png'
    )
    
    # Save model
    torch.save(model.model.state_dict(), output_path / f'transformer_{symbol}_model.pth')
    print(f"Model saved to {output_path / f'transformer_{symbol}_model.pth'}")
    
    print(f"\n{'='*60}")
    print("Model training completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
