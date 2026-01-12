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
        # Return sequence of past features and next minute's target
        X = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return X, y


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Parameters:
        -----------
        x : Tensor
            Shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """Transformer model for spread prediction"""
    
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1):
        """
        Parameters:
        -----------
        n_features : int
            Number of input features
        d_model : int
            Dimension of model embeddings
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer encoder layers
        dim_feedforward : int
            Dimension of feedforward network
        dropout : float
            Dropout rate
        """
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        Parameters:
        -----------
        x : Tensor
            Shape (batch_size, seq_len, n_features)
            
        Returns:
        --------
        Tensor
            Shape (batch_size, 1)
        """
        # x: (batch_size, seq_len, n_features)
        batch_size, seq_len, n_features = x.shape
        
        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Transpose for transformer: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (seq_len, batch_size, d_model)
        
        # Use the last timestep's output
        x = x[-1, :, :]  # (batch_size, d_model)
        
        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze()


class TransformerModel:
    """Wrapper class for transformer model training and evaluation"""
    
    def __init__(self, seq_length=60, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.1, learning_rate=0.001,
                 batch_size=32, epochs=50):
        """
        Initialize the transformer model
        
        Parameters:
        -----------
        seq_length : int
            Number of past timesteps to use for prediction
        d_model : int
            Dimension of model embeddings
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer encoder layers
        dim_feedforward : int
            Dimension of feedforward network
        dropout : float
            Dropout rate
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
        epochs : int
            Number of training epochs
        """
        self.seq_length = seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
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
    
    def create_dataloaders(self, X_train, y_train, X_test, y_test):
        """Create PyTorch dataloaders"""
        train_dataset = TimeSeriesDataset(X_train, y_train, self.seq_length)
        test_dataset = TimeSeriesDataset(X_test, y_test, self.seq_length)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                 shuffle=False)  # Don't shuffle for time series
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                shuffle=False)
        
        print(f"\nTrain sequences: {len(train_dataset)}")
        print(f"Test sequences: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the transformer model"""
        print(f"\nTraining Transformer model...")
        print(f"Sequence length: {self.seq_length}")
        print(f"Model dimension: {self.d_model}")
        print(f"Attention heads: {self.nhead}")
        print(f"Encoder layers: {self.num_layers}")
        
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
        self.model = TransformerPredictor(
            n_features=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
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
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
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
        plt.title('Training History', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def analyze_opportunity_performance(self, y_test, y_pred, is_opportunity, opp_thresh=0.30):
        """Analyze model performance specifically on opportunity rows"""
        print("\n" + "="*60)
        print("OPPORTUNITY-SPECIFIC PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Align opportunity flags with predictions (account for seq_length)
        is_opp_aligned = is_opportunity[self.seq_length:]
        
        # Separate by opportunity status
        opp_mask = is_opp_aligned == 1
        non_opp_mask = ~opp_mask
        
        n_opp = opp_mask.sum()
        n_non_opp = non_opp_mask.sum()
        
        print(f"\nDataset composition:")
        print(f"  Opportunity rows: {n_opp} ({100*n_opp/len(is_opp_aligned):.2f}%)")
        print(f"  Non-opportunity rows: {n_non_opp} ({100*n_non_opp/len(is_opp_aligned):.2f}%)")
        
        if n_opp == 0:
            print("\nNo opportunity rows found in test set!")
            return None
        
        # Metrics for opportunity rows
        y_test_opp = y_test[opp_mask]
        y_pred_opp = y_pred[opp_mask]
        
        mae_opp = mean_absolute_error(y_test_opp, y_pred_opp)
        rmse_opp = np.sqrt(mean_squared_error(y_test_opp, y_pred_opp))
        r2_opp = r2_score(y_test_opp, y_pred_opp)
        
        # Metrics for non-opportunity rows
        y_test_non_opp = y_test[non_opp_mask]
        y_pred_non_opp = y_pred[non_opp_mask]
        
        mae_non_opp = mean_absolute_error(y_test_non_opp, y_pred_non_opp)
        rmse_non_opp = np.sqrt(mean_squared_error(y_test_non_opp, y_pred_non_opp))
        r2_non_opp = r2_score(y_test_non_opp, y_pred_non_opp)
        
        print(f"\nPerformance on OPPORTUNITY rows (n={n_opp}):")
        print(f"  MAE: {mae_opp:.6f}")
        print(f"  RMSE: {rmse_opp:.6f}")
        print(f"  R²: {r2_opp:.4f}")
        print(f"  Actual spread range: [{y_test_opp.min():.4f}, {y_test_opp.max():.4f}]")
        print(f"  Predicted spread range: [{y_pred_opp.min():.4f}, {y_pred_opp.max():.4f}]")
        print(f"  Mean actual: {y_test_opp.mean():.4f}")
        print(f"  Mean predicted: {y_pred_opp.mean():.4f}")
        
        print(f"\nPerformance on NON-OPPORTUNITY rows (n={n_non_opp}):")
        print(f"  MAE: {mae_non_opp:.6f}")
        print(f"  RMSE: {rmse_non_opp:.6f}")
        print(f"  R²: {r2_non_opp:.4f}")
        print(f"  Actual spread range: [{y_test_non_opp.min():.4f}, {y_test_non_opp.max():.4f}]")
        print(f"  Predicted spread range: [{y_pred_non_opp.min():.4f}, {y_pred_non_opp.max():.4f}]")
        print(f"  Mean actual: {y_test_non_opp.mean():.4f}")
        print(f"  Mean predicted: {y_pred_non_opp.mean():.4f}")
        
        # Opportunity detection capability
        y_pred_is_opp = y_pred >= opp_thresh
        y_test_is_opp = y_test >= opp_thresh
        
        tp = np.sum(y_test_is_opp & y_pred_is_opp)
        fp = np.sum(~y_test_is_opp & y_pred_is_opp)
        fn = np.sum(y_test_is_opp & ~y_pred_is_opp)
        tn = np.sum(~y_test_is_opp & ~y_pred_is_opp)
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        
        print(f"\nOpportunity Detection (threshold={opp_thresh}):")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Negatives: {tn}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        
        # Check if model is just predicting mean/constant
        pred_std = np.std(y_pred)
        pred_range = y_pred.max() - y_pred.min()
        print(f"\nPrediction diversity check:")
        print(f"  Prediction std: {pred_std:.6f}")
        print(f"  Prediction range: {pred_range:.6f}")
        if pred_std < 0.001:
            print("  ⚠️  WARNING: Model predictions have very low variance!")
            print("  ⚠️  Model may be collapsing to constant predictions.")
        
        return {
            'opp_metrics': {'mae': mae_opp, 'rmse': rmse_opp, 'r2': r2_opp},
            'non_opp_metrics': {'mae': mae_non_opp, 'rmse': rmse_non_opp, 'r2': r2_non_opp},
            'detection': {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        }
    
    def calculate_threshold_metrics(self, y_test, y_pred, opp_thresh=0.30, tol=0.002):
        """Calculate precision, recall, F1, and hit-rate across different thresholds"""
        thresholds = np.linspace(0.02, 0.20, 50)
        precisions = []
        recalls = []
        f1_scores = []
        hit_rates = []
        
        y_test_is_real_opp = y_test >= opp_thresh
        
        for thresh in thresholds:
            y_pred_is_opp = y_pred >= thresh
            
            tp = np.sum(y_test_is_real_opp & y_pred_is_opp)
            fp = np.sum(~y_test_is_real_opp & y_pred_is_opp)
            fn = np.sum(y_test_is_real_opp & ~y_pred_is_opp)
            
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            
            # Hit-rate: proportion of real opportunities predicted within tolerance
            if y_test_is_real_opp.sum() > 0:
                hit_rate = np.mean((np.abs(y_test - y_pred) <= tol)[y_test_is_real_opp])
            else:
                hit_rate = 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            hit_rates.append(hit_rate)
        
        return thresholds, precisions, recalls, f1_scores, hit_rates
    
    def plot_threshold_metrics(self, y_test, y_pred, opp_thresh=0.30, tol=0.002, save_path=None):
        """Plot precision, recall, F1, and hit-rate vs prediction threshold"""
        thresholds, precisions, recalls, f1_scores, hit_rates = self.calculate_threshold_metrics(
            y_test, y_pred, opp_thresh, tol
        )
        
        # Find best F1 threshold
        best_f1_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(thresholds, precisions, label='Precision', marker='o', 
                markersize=3, linewidth=2, alpha=0.8)
        ax.plot(thresholds, recalls, label='Recall', marker='s', 
                markersize=3, linewidth=2, alpha=0.8)
        ax.plot(thresholds, f1_scores, label='F1 Score', marker='^', 
                markersize=3, linewidth=2, alpha=0.8)
        ax.plot(thresholds, hit_rates, label=f'Hit-rate (±{tol})', marker='d', 
                markersize=3, linewidth=2, alpha=0.8)
        
        # Mark best F1 point
        ax.axvline(x=best_thresh, color='red', linestyle='--', alpha=0.5, 
                   label=f'Best F1 threshold: {best_thresh:.4f}')
        ax.plot(best_thresh, best_f1, 'r*', markersize=15, 
                label=f'Best F1: {best_f1:.3f}')
        
        # Mark the opp_thresh line
        ax.axvline(x=opp_thresh, color='gray', linestyle=':', alpha=0.5, 
                   label=f'Real opp threshold: {opp_thresh}')
        
        ax.set_xlabel('Prediction Threshold', fontsize=13)
        ax.set_ylabel('Metric Value', fontsize=13)
        ax.set_title('Opportunity Detection Metrics vs Threshold', 
                     fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold metrics plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
        
        print(f"\nBest operating point: threshold={best_thresh:.4f}")
        print(f"  Precision: {precisions[best_f1_idx]:.3f}")
        print(f"  Recall: {recalls[best_f1_idx]:.3f}")
        print(f"  F1: {best_f1:.3f}")
        print(f"  Hit-rate: {hit_rates[best_f1_idx]:.3f}")
        
        return best_thresh
    
    def plot_opportunity_comparison(self, y_test, y_pred, is_opportunity, save_path=None):
        """Plot comparison of performance on opportunity vs non-opportunity rows"""
        # Align opportunity flags
        is_opp_aligned = is_opportunity[self.seq_length:]
        opp_mask = is_opp_aligned == 1
        non_opp_mask = ~opp_mask
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot colored by opportunity
        axes[0, 0].scatter(y_test[non_opp_mask], y_pred[non_opp_mask], 
                          alpha=0.3, s=10, label='Non-opportunity', color='blue')
        axes[0, 0].scatter(y_test[opp_mask], y_pred[opp_mask], 
                          alpha=0.6, s=20, label='Opportunity', color='red')
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'k--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual spread_close_pct', fontsize=12)
        axes[0, 0].set_ylabel('Predicted spread_close_pct', fontsize=12)
        axes[0, 0].set_title('Predictions: Opportunity vs Non-Opportunity', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error distribution comparison
        errors_opp = np.abs(y_test[opp_mask] - y_pred[opp_mask])
        errors_non_opp = np.abs(y_test[non_opp_mask] - y_pred[non_opp_mask])
        
        axes[0, 1].hist(errors_non_opp, bins=50, alpha=0.5, label='Non-opportunity', 
                       color='blue', edgecolor='black')
        axes[0, 1].hist(errors_opp, bins=50, alpha=0.5, label='Opportunity', 
                       color='red', edgecolor='black')
        axes[0, 1].set_xlabel('Absolute Error', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Box plot comparison
        box_data = [errors_non_opp, errors_opp]
        bp = axes[1, 0].boxplot(box_data, labels=['Non-Opportunity', 'Opportunity'],
                                patch_artist=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('red')
        axes[1, 0].set_ylabel('Absolute Error', fontsize=12)
        axes[1, 0].set_title('Error Distribution (Box Plot)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Metrics comparison bar chart
        mae_opp = np.mean(errors_opp)
        mae_non_opp = np.mean(errors_non_opp)
        rmse_opp = np.sqrt(np.mean((y_test[opp_mask] - y_pred[opp_mask])**2))
        rmse_non_opp = np.sqrt(np.mean((y_test[non_opp_mask] - y_pred[non_opp_mask])**2))
        
        x = np.arange(2)
        width = 0.35
        axes[1, 1].bar(x - width/2, [mae_non_opp, rmse_non_opp], width, 
                      label='Non-Opportunity', color='blue', alpha=0.7)
        axes[1, 1].bar(x + width/2, [mae_opp, rmse_opp], width, 
                      label='Opportunity', color='red', alpha=0.7)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['MAE', 'RMSE'])
        axes[1, 1].set_ylabel('Error', fontsize=12)
        axes[1, 1].set_title('Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Opportunity comparison plot saved to {save_path}")
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
    """Main function to train and evaluate the transformer model"""
    parser = argparse.ArgumentParser(description='Train Transformer model for crypto spread prediction')
    parser.add_argument('--crypto', type=str, default='BTCUSD',
                        help='Cryptocurrency to model (default: BTCUSD)')
    parser.add_argument('--seq-length', type=int, default=60,
                        help='Sequence length for transformer input (default: 60)')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Dimension of model embeddings (default: 128)')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of transformer layers (default: 3)')
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
    print(f"\nInitializing Transformer model for {args.crypto}...")
    model = TransformerModel(
        seq_length=args.seq_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Load data
    df = model.load_data(file_path)
    
    # Extract opportunity flags before preparing features (they get excluded)
    is_real_opportunity = df['is_real_opportunity'].values if 'is_real_opportunity' in df.columns else None
    
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
    
    # Analyze opportunity-specific performance
    if is_real_opportunity is not None:
        is_real_opp_test = is_real_opportunity[split_idx:]
        opp_analysis = model.analyze_opportunity_performance(
            y_test_aligned, 
            y_pred, 
            is_real_opp_test,
            opp_thresh=0.30
        )
    
    # Output directory
    output_path = base_path / 'models' / 'ds_model' / 'transformer' / args.crypto
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_path}")
    
    # Save plots
    model.plot_training_history(
        train_losses, 
        val_losses, 
        save_path=output_path / f'transformer_{args.crypto}_training_history.png'
    )
    
    model.plot_results(
        y_test_aligned,
        y_pred,
        save_path=output_path / f'transformer_{args.crypto}_results.png'
    )
    
    # Plot opportunity-specific comparison
    if is_real_opportunity is not None:
        is_real_opp_test = is_real_opportunity[split_idx:]
        model.plot_opportunity_comparison(
            y_test_aligned,
            y_pred,
            is_real_opp_test,
            save_path=output_path / f'transformer_{args.crypto}_opportunity_comparison.png'
        )
        
        # Plot threshold metrics (precision, recall, F1, hit-rate)
        model.plot_threshold_metrics(
            y_test_aligned,
            y_pred,
            opp_thresh=0.30,
            tol=0.002,
            save_path=output_path / f'transformer_{args.crypto}_threshold_metrics.png'
        )
    
    # Save model
    torch.save(model.model.state_dict(), output_path / f'transformer_{args.crypto}_model.pth')
    print(f"Model saved to {output_path / f'transformer_{args.crypto}_model.pth'}")
    
    print("\n" + "="*60)
    print("Model training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
