"""Transformer attention-based model for predicting cryptocurrency spread movements.

This module implements a Transformer encoder architecture to predict the next-minute
spread between buy and sell prices across exchanges (is_real_opportunity).

Transformer Advantage:
    - Self-attention mechanism learns long-range dependencies without recurrence
    - Fully parallelizable (faster training than LSTM/GRU)
    - Attention weights provide interpretability
    - PyTorch-based for flexibility

Architecture:
    - Input: Feature sequences (shape: [seq_length, num_features])
    - Positional Encoding: Sine/cosine position embeddings
    - Linear Projection: Input features → d_model dimension
    - Transformer Encoder: Multi-head self-attention layers
    - Output FC layers: d_model → 64 → 1 (sigmoid)
    - Output: Sigmoid-activated spread prediction [0, 1]

Key Components:
    - PositionalEncoding: Adds position information
    - TransformerPredictor: Main model class (nn.Module)
    - TransformerEncoderLayer: Multi-head attention + feedforward
    - DataLoader: PyTorch batch training

Key Hyperparameters:
    - d_model: Feature embedding dimension (default: 64)
    - nhead: Number of attention heads (default: 4)
    - num_layers: Number of encoder layers (default: 2)
    - dim_feedforward: Feedforward hidden dimension (default: 256)
    - dropout_rate: Dropout probability (default: 0.1)
    - sequence_length: Temporal window size

Data Pipeline:
    1. Load featured data from CSV
    2. Prepare features (exclude time, exchanges, target)
    3. Create sequences using time windows
    4. Scale features with MinMaxScaler [0, 1]
    5. Build TransformerPredictor
    6. Train with PyTorch DataLoader
    7. Generate predictions and visualizations

Output:
    - Predictions on test set
    - Attention weights (for interpretability)
    - Training metrics and visualizations

Typical Usage:
    model = TransformerPredictor(d_model=64, nhead=4, num_layers=2)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32)
    train(model, train_loader, epochs=50)
    y_pred = model(X_test_tensor)

Author: TAU DS Project | Arbitrage team
Date: 2026
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score


# Import plotting functions from plotter
from plotter import (
    plot_results,
    plot_prediction_hist,
    plot_training_history
)

#Surpress warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        
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
    
    def __init__(self, n_features, d_model, nhead, num_layers, 
                 dim_feedforward, dropout_rate):
        super().__init__()
        
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)     
        return x.squeeze()


class TransformerModel:
    """Wrapper class for transformer model training and evaluation"""
    
    def __init__(self,
                threshold,
                d_model,
                num_layers,
                dropout_rate,
                sequence_length,
                nhead,
                dim_feedforward,
                learning_rate,
                split_ratio):
        
        self.threshold = threshold
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.learning_rate = learning_rate
        self.split_ratio = split_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = None
        self.scaler = MinMaxScaler()
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
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def prepare_features(self, df):
        """Prepare features and split data (60% train, 20% val, 20% test)"""
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
        X = df.drop(columns=default_exclude, errors='ignore')
        self.feature_names = X.columns.tolist()
        
        # Define target name as next minute prediction
        y = df[self.target_name].shift(-1)
        
        # drop null rows created by shift(-1) AND any NaN in X
        mask = y.notna() & X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
        
        X = X.values
        y = y.values
        
        print(f"\nFeatures prepared: {len(self.feature_names)} features")
        print(f"Total samples: {X.shape[0]}")
        print(f"Target range: [{y.min():.6f}, {y.max():.6f}]")
        
        # Chronological split: 60% train, 20% val, 20% test
        train_idx = int(len(X) * self.split_ratio)
        val_idx = int(len(X) * (self.split_ratio+0.2))
        
        X_train = X[:train_idx]
        X_val = X[train_idx:val_idx]
        X_test = X[val_idx:]
        
        y_train = y[:train_idx]
        y_val = y[train_idx:val_idx]
        y_test = y[val_idx:]
        
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def create_sequences(self, X, y):
        """
        Create sequences for time series prediction
        
        Parameters:
        -----------
        X : np.array
            Feature array (n_samples, n_features)
        y : np.array
            Target array (n_samples,)
            
        Returns:
        --------
        X_seq : np.array
            Sequence features (n_sequences, sequence_length, n_features)
        y_seq : np.array
            Sequence targets (n_sequences,)
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)   
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """Train the transformer model"""
        
        print(f"\nTraining Transformer model...")
        print(f"Configuration:")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Model dimension: {self.d_model}")
        print(f"  Attention heads: {self.nhead}")
        print(f"  Encoder layers: {self.num_layers}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}\n")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val)
            val_dataset = TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        n_features = X_train.shape[1]
        self.model = TransformerPredictor(
            n_features=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
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
                
                if (epoch + 1) % 1 == 0:  # Print every epoch
                    if val_loader is not None:
                        print(f"Epoch [{epoch+1:3d}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    else:
                        print(f"Epoch [{epoch+1:3d}/{epochs}] - Train Loss: {train_loss:.6f}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}")
    
        self.is_fitted = True
        self.history = {'loss': train_losses, 'val_loss': val_losses}
    
    def predict(self, X, batch_size):
        """Make predictions on test data"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        X_scaled = self.scaler.transform(X)
        
        # Create sequences (y can be dummy values, only X matters)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        dataset = TensorDataset(torch.FloatTensor(X_seq))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for X_batch, in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)

    def evaluate(self, X_test, y_test, batch_size):
        """Evaluate the model by predicting on test data"""
        print("\nEvaluating model...")
        
        # Call predict internally
        y_pred = self.predict(X_test, batch_size)
        
        # Align test data with sequence length
        y_test_aligned = y_test[self.sequence_length:]
        
        # Create binary predictions using threshold
        y_pred_binary = (y_pred > self.threshold).astype(int)
        y_test_binary = (y_test_aligned > self.threshold).astype(int)
        
        metrics = {
            'mse': mean_squared_error(y_test_aligned, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_aligned, y_pred)),
            'mae': mean_absolute_error(y_test_aligned, y_pred),
            'r2': r2_score(y_test_aligned, y_pred),
            'mape': np.mean(np.abs((y_test_aligned - y_pred) / (y_test_aligned + 1e-9))) * 100,
            'f1': f1_score(y_test_binary, y_pred_binary),
        }
        
        print(f"Test R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Threshold: {self.threshold}")
        
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
    parser.add_argument('--d-model', type=int, default=128,
                        help='Dimension of model embeddings (default: 128)')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of transformer layers (default: 3)')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Dropout rate for regularization (default: 0.2)')
    
    return parser.parse_args()


def main():
    """Main function to train and evaluate the transformer model"""
    
    # Parse CLI agruments
    args = parse_args()
    
    # Hardcoded arguments
    sequence_length = 20 # Matched to lstm/gru, was 60, is it better?
    nhead = 8
    dim_feedforward = 4*args.d_model
    batch_size = 32
    epochs = 50    
    split_ratio = 0.6 #split is 60/20/20
    learning_rate = 0.001
        
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get base path
    base_path = Path(__file__).parent.parent
    
    # Get symbol
    symbol = args.symbol
        
    print(f"\n{'='*60}")
    print(f"Training Transformer for {symbol}")
    print(f"{'='*60}\n")
    
    print(f"Configuration:")
    print(f"  Seed (numPy and PyTorch): {args.seed}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Model Dimension: {args.d_model}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Number of heads: {nhead}")
    print(f"  Dimensions forward: {dim_feedforward}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Split ratio (Train part): {split_ratio}\n")
    
    # Initialize model
    model = TransformerModel(
        threshold = args.threshold,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        sequence_length=sequence_length,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        learning_rate=learning_rate,
        split_ratio = split_ratio
    )
    
    # Load and prepare data
    df = model.load_data(symbol)    
    X_train, X_val, X_test, y_train, y_val, y_test  = model.prepare_features(df)
       
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    
    # Evaluate model
    metrics, y_pred, y_test_aligned = model.evaluate(X_test, y_test, batch_size=batch_size)
    
    # Make predictions again for plotting
    y_pred = model.predict(X_test, batch_size=batch_size).flatten()
    
    # Output directory
    output_path = base_path / 'models' / 'ds_model' / 'transformer' / symbol
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output_path}")
    
    # Use plotter functions for plots
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
    
    plot_training_history(
        model.history,
        model_name='Transformer',
        save_path=output_path / f'transformer_{symbol}_training_history.png'
    )
    
    
    # Save model
    torch.save(model.model.state_dict(), output_path / f'transformer_{symbol}_model.pth')
    print(f"Model saved to {output_path / f'transformer_{symbol}_model.pth'}")
    
    print(f"\n{'='*60}")
    print("Model training completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
