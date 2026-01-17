import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """Dataset for time series prediction"""
    
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
        # Return sequence of past features and next minute's target
        X = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return X, y


class RNNPredictor(nn.Module):
    """RNN/LSTM/GRU model for spread prediction"""
    
    def __init__(self, n_features, hidden_dim=64, num_layers=2, output_dim=1, 
                 model_type='LSTM', dropout=0.1):
        """
        Parameters:
        -----------
        n_features : int
            Number of input features
        hidden_dim : int
            Number of features in the hidden state
        num_layers : int
            Number of recurrent layers
        output_dim : int
            Dimension of output
        model_type : str
            'LSTM', 'GRU', or 'RNN'
        dropout : float
            Dropout probability
        """
        super(RNNPredictor, self).__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the recurrent layer
        if model_type == 'GRU':
            self.rnn = nn.GRU(n_features, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout)
        elif model_type == 'RNN':
            self.rnn = nn.RNN(n_features, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout)
        else: # LSTM
            self.rnn = nn.LSTM(n_features, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout)
            
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Parameters:
        -----------
        x : Tensor
            Shape (batch_size, seq_len, n_features)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate RNN
        if self.model_type == 'LSTM':
            # Initialize cell state
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
            
        # Decode the hidden state of the last time step
        # out shape: (batch_size, seq_len, hidden_dim)
        out = out[:, -1, :]
        out = self.fc(out)
        
        return out.squeeze()


class RNNModel:
    """Wrapper class for RNN/LSTM/GRU model training and evaluation"""
    
    def __init__(self, model_type='LSTM', seq_length=60, hidden_dim=64, num_layers=2,
                 dropout=0.1, learning_rate=0.001, batch_size=32, epochs=50):
        """
        Initialize the RNN model wrapper
        """
        self.model_type = model_type
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = 'spread_close_pct'
        self.is_fitted = False
        
    def load_data(self, file_path):
        """Load featured data from CSV file"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def prepare_features(self, df, exclude_features=None):
        """Prepare features for training"""
        # Columns to exclude (to prevent data leakage)
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
        
        # Clean data
        df_clean = df[feature_cols + [self.target_name]].copy()
        df_clean = df_clean.dropna(subset=[self.target_name])
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values
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
        """Train the RNN model"""
        print(f"\nTraining {self.model_type} model...")
        print(f"Sequence length: {self.seq_length}")
        print(f"Hidden dimension: {self.hidden_dim}")
        print(f"Layers: {self.num_layers}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train, self.seq_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_dataset = TimeSeriesDataset(X_val_scaled, y_val, self.seq_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Initialize model
        n_features = X_train.shape[1]
        self.model = RNNPredictor(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            model_type=self.model_type,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
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
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.6f}")
        
        self.is_fitted = True
        return train_losses, val_losses
    
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
    
    def evaluate(self, X_test, y_test, tolerance=0.002):
        """Evaluate the model"""
        print("\nEvaluating model...")
        
        # Get predictions (accounting for sequence length)
        y_pred = self.predict(X_test)
        y_test_aligned = y_test[self.seq_length:]
        
        # Calculate absolute errors
        abs_errors = np.abs(y_test_aligned - y_pred)
        
        # Hit rate: predictions within tolerance
        within_tolerance = abs_errors <= tolerance
        hit_rate = np.mean(within_tolerance)
        
        metrics = {
            'mse': mean_squared_error(y_test_aligned, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_aligned, y_pred)),
            'mae': mean_absolute_error(y_test_aligned, y_pred),
            'r2': r2_score(y_test_aligned, y_pred),
            'mape': np.mean(np.abs((y_test_aligned - y_pred) / (y_test_aligned + 1e-9))) * 100,
            'hit_rate': hit_rate,
            'tolerance': tolerance
        }
        
        print(f"Test R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"Hit Rate (within ±{tolerance}): {hit_rate:.2%} ({int(within_tolerance.sum())}/{len(y_test_aligned)} predictions)")
        
        return metrics, y_pred, y_test_aligned
    
    def plot_training_history(self, train_losses, val_losses=None, save_path=None):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title(f'Training History ({self.model_type})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_results(self, y_test, y_pred, save_path=None):
        """Plot prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual spread_close_pct', fontsize=12)
        axes[0, 0].set_ylabel('Predicted spread_close_pct', fontsize=12)
        axes[0, 0].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted spread_close_pct', fontsize=12)
        axes[0, 1].set_ylabel('Residuals', fontsize=12)
        axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Time series comparison (last 200 points)
        n_points = min(200, len(y_test))
        axes[1, 0].plot(range(n_points), y_test[-n_points:], label='Actual', alpha=0.7)
        axes[1, 0].plot(range(n_points), y_pred[-n_points:], label='Predicted', alpha=0.7)
        axes[1, 0].set_xlabel('Time Step', fontsize=12)
        axes[1, 0].set_ylabel('spread_close_pct', fontsize=12)
        axes[1, 0].set_title(f'Prediction vs Actual (Last {n_points} points)', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residual Distribution
        axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Residuals', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {save_path}")
        else:
            plt.show()
        plt.close()


def main():
    """Main function to train and evaluate the RNN model"""
    parser = argparse.ArgumentParser(description='Train RNN/LSTM/GRU model for crypto spread prediction')
    parser.add_argument('--crypto', type=str, default='BTCUSD',
                        help='Cryptocurrency to model (default: BTCUSD)')
    parser.add_argument('--model-type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'],
                        help='Type of recurrent model (default: LSTM)')
    parser.add_argument('--seq-length', type=int, default=60,
                        help='Sequence length for input (default: 60)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension size (default: 64)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of recurrent layers (default: 2)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define paths
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data' / 'featured_data'
    file_path = data_path / f'featured_{args.crypto}_data.csv'
    
    # Initialize model
    print(f"\nInitializing {args.model_type} model for {args.crypto}...")
    model = RNNModel(
        model_type=args.model_type,
        seq_length=args.seq_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Load data
    df = model.load_data(file_path)
    
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
    train_losses, val_losses = model.train(X_train_final, y_train_final, X_val, y_val)
    
    # Evaluate model
    metrics, y_pred, y_test_aligned = model.evaluate(X_test, y_test)
    
    # Output directory
    output_path = base_path / 'models' / 'ds_model' / 'rnn' / args.crypto
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_path}")
    
    # Save plots
    model.plot_training_history(
        train_losses, 
        val_losses, 
        save_path=output_path / f'{args.model_type.lower()}_{args.crypto}_training_history.png'
    )
    
    model.plot_results(
        y_test_aligned,
        y_pred,
        save_path=output_path / f'{args.model_type.lower()}_{args.crypto}_results.png'
    )
    
    # Save model
    torch.save(model.model.state_dict(), output_path / f'{args.model_type.lower()}_{args.crypto}_model.pth')
    print(f"Model saved to {output_path / f'{args.model_type.lower()}_{args.crypto}_model.pth'}")
    
    print("\n" + "="*60)
    print("Model training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
