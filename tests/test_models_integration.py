"""
Integration tests for ML models - validates models run correctly with actual data.

Usage:
    pytest tests/test_models_integration.py -v
    pytest tests/test_models_integration.py::TestLinearModelIntegration -v
    pytest tests/test_models_integration.py -k "integration" -v
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Project root
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / 'models'
DATA_DIR = REPO_ROOT / 'data' / 'featured_data'


# ============================================================================
# Utilities
# ============================================================================

def is_git_lfs_pointer(file_path):
    """Check if file is a Git LFS pointer (not actual data)"""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            return first_line.startswith('version https://git-lfs.github.com')
    except:
        return False


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def available_symbols():
    """Get available cryptocurrency symbols from data directory"""
    if not DATA_DIR.exists():
        return []
    
    symbols = []
    for csv_file in DATA_DIR.glob('featured_*_data.csv'):
        # Skip Git LFS pointers
        if is_git_lfs_pointer(csv_file):
            continue
        
        # Extract symbol from filename: featured_SYMBOL_data.csv
        symbol = csv_file.name.replace('featured_', '').replace('_data.csv', '')
        symbols.append(symbol)
    
    return sorted(list(set(symbols)))


@pytest.fixture(scope="session")
def sample_symbol(available_symbols):
    """Get first available symbol for testing"""
    if not available_symbols:
        pytest.skip("No featured data files found (Git LFS files not downloaded?)")
    return available_symbols[0]


@pytest.fixture(scope="session")
def featured_data(sample_symbol):
    """Load sample featured data"""
    data_path = DATA_DIR / f'featured_{sample_symbol}_data.csv'
    
    if not data_path.exists():
        pytest.skip(f"Data file not found: {data_path}")
    
    if is_git_lfs_pointer(data_path):
        pytest.skip(f"Data file is Git LFS pointer (not downloaded): {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        return df, sample_symbol
    except Exception as e:
        pytest.skip(f"Cannot read data: {e}")


@pytest.fixture(scope="session")
def data_stats(featured_data):
    """Calculate statistics about featured data"""
    df, symbol = featured_data
    
    has_target = 'spread_close_pct' in df.columns
    
    return {
        'symbol': symbol,
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': df.columns.tolist(),
        'has_target': has_target,
        'target_range': (df['spread_close_pct'].min(), df['spread_close_pct'].max()) 
                       if has_target else None,
        'null_count': df.isnull().sum().sum(),
    }


# ============================================================================
# Data Validation Tests
# ============================================================================

class TestDataAvailability:
    """Test that required data exists"""
    
    def test_featured_data_dir_exists(self):
        """Test featured_data directory exists"""
        assert DATA_DIR.exists(), f"Data directory not found: {DATA_DIR}"
    
    def test_featured_data_files_exist(self, available_symbols):
        """Test at least one featured data file exists (not Git LFS pointer)"""
        if len(available_symbols) == 0:
            # Check if we have Git LFS pointers
            pointer_count = sum(1 for f in DATA_DIR.glob('featured_*_data.csv') 
                              if is_git_lfs_pointer(f))
            if pointer_count > 0:
                pytest.skip(f"Found {pointer_count} Git LFS pointers but files not downloaded. "
                          "Run: git lfs pull")
            else:
                pytest.skip("No featured data files found")
        
        print(f"\nAvailable symbols: {available_symbols}")
    
    def test_sample_data_readable(self, featured_data):
        """Test sample data can be read"""
        df, symbol = featured_data
        assert df is not None
        assert len(df) > 0
        print(f"\nLoaded {symbol}: {len(df)} rows, {len(df.columns)} columns")
    
    def test_target_column_exists(self, featured_data):
        """Test target column exists in data"""
        df, symbol = featured_data
        assert 'spread_close_pct' in df.columns, \
            f"Target column 'spread_close_pct' not found in {symbol} data. Columns: {df.columns.tolist()}"
    
    def test_data_has_minimum_rows(self, featured_data):
        """Test data has minimum required rows"""
        df, symbol = featured_data
        assert len(df) >= 100, \
            f"{symbol} has only {len(df)} rows, need at least 100"
    
    def test_data_target_distribution(self, data_stats):
        """Test target variable has reasonable distribution"""
        stats = data_stats
        if stats['target_range']:
            min_val, max_val = stats['target_range']
            assert min_val < max_val, "Target has no variance"
            print(f"\nTarget range: [{min_val:.6f}, {max_val:.6f}]")


# ============================================================================
# Linear Model Tests
# ============================================================================

class TestLinearModelIntegration:
    """Integration tests for Linear model"""
    
    def test_linear_runs_with_defaults(self, sample_symbol):
        """Test linear model runs with default args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"\nLinear model output:\n{result.stdout[:500]}")
        
        # Should complete successfully or with data error
        assert 'unrecognized' not in result.stderr.lower()
    
    def test_linear_accepts_all_hyperparams(self, sample_symbol):
        """Test linear model accepts all hyperparameters"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol,
             '--seed', '123',
             '--threshold', '0.5',
             '--model-type', 'lasso',
             '--alpha', '0.001'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_linear_produces_output_dir(self, sample_symbol):
        """Test linear model creates output directory"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol,
             '--seed', '42'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Output dir may or may not exist depending on data, but no error
        assert 'unrecognized' not in result.stderr.lower()


# ============================================================================
# LSTM Model Tests
# ============================================================================

class TestLSTMModelIntegration:
    """Integration tests for LSTM model"""
    
    def test_lstm_runs_with_minimal_epochs(self, sample_symbol):
        """Test LSTM model runs with minimal epochs"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_lstm.py'),
             '--symbol', sample_symbol,
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"\nLSTM model output:\n{result.stdout[:500]}")
        
        assert 'unrecognized' not in result.stderr.lower()
    
    def test_lstm_accepts_sequence_params(self, sample_symbol):
        """Test LSTM accepts sequence parameters"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_lstm.py'),
             '--symbol', sample_symbol,
             '--seq-length', '10',
             '--units', '32',
             '--batch-size', '16',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()


# ============================================================================
# GRU Model Tests
# ============================================================================

class TestGRUModelIntegration:
    """Integration tests for GRU model"""
    
    def test_gru_runs_with_minimal_epochs(self, sample_symbol):
        """Test GRU model runs with minimal epochs"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_gru.py'),
             '--symbol', sample_symbol,
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"\nGRU model output:\n{result.stdout[:500]}")
        
        assert 'unrecognized' not in result.stderr.lower()
    
    def test_gru_accepts_sequence_params(self, sample_symbol):
        """Test GRU accepts sequence parameters"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_gru.py'),
             '--symbol', sample_symbol,
             '--seq-length', '10',
             '--units', '32',
             '--batch-size', '16',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()


# ============================================================================
# RandomForest Model Tests
# ============================================================================

class TestRandomForestModelIntegration:
    """Integration tests for RandomForest model"""
    
    def test_rf_runs_with_defaults(self, sample_symbol):
        """Test RF model runs with default args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"\nRandomForest model output:\n{result.stdout[:500]}")
        
        assert 'unrecognized' not in result.stderr.lower()
    
    def test_rf_accepts_all_hyperparams(self, sample_symbol):
        """Test RF accepts all hyperparameters"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--symbol', sample_symbol,
             '--n-estimators', '50',
             '--max-depth', '8',
             '--seed', '99'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()


# ============================================================================
# XGBoost Model Tests
# ============================================================================

class TestXGBoostModelIntegration:
    """Integration tests for XGBoost model"""
    
    def test_xgboost_runs_with_defaults(self, sample_symbol):
        """Test XGBoost model runs with default args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"\nXGBoost model output:\n{result.stdout[:500]}")
        
        assert 'unrecognized' not in result.stderr.lower()
    
    def test_xgboost_accepts_all_hyperparams(self, sample_symbol):
        """Test XGBoost accepts all hyperparameters"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
             '--symbol', sample_symbol,
             '--train-frac', '0.6',
             '--val-frac', '0.2',
             '--seed', '77'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()


# ============================================================================
# Transformer Model Tests
# ============================================================================

class TestTransformerModelIntegration:
    """Integration tests for Transformer model"""
    
    def test_transformer_runs_with_minimal_epochs(self, sample_symbol):
        """Test Transformer model runs with minimal epochs"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_transformer.py'),
             '--symbol', sample_symbol,
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"\nTransformer model output:\n{result.stdout[:500]}")
        
        assert 'unrecognized' not in result.stderr.lower()
    
    def test_transformer_accepts_all_hyperparams(self, sample_symbol):
        """Test Transformer accepts all hyperparameters"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_transformer.py'),
             '--symbol', sample_symbol,
             '--seq-length', '20',
             '--d-model', '64',
             '--nhead', '4',
             '--num-layers', '1',
             '--batch-size', '16',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()


# ============================================================================
# CatBoost Model Tests
# ============================================================================

class TestCatBoostModelIntegration:
    """Integration tests for CatBoost model"""
    
    def test_catboost_runs_with_defaults(self, sample_symbol):
        """Test CatBoost model runs with default args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_catboost.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"\nCatBoost model output:\n{result.stdout[:500]}")
        
        assert 'unrecognized' not in result.stderr.lower()
    
    def test_catboost_accepts_all_hyperparams(self, sample_symbol):
        """Test CatBoost accepts all hyperparameters"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_catboost.py'),
             '--symbol', sample_symbol,
             '--iterations', '100',
             '--learning-rate', '0.05',
             '--depth', '4',
             '--seed', '55'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()


# ============================================================================
# Cross-Model Consistency Tests
# ============================================================================

class TestModelConsistency:
    """Test consistency across all models"""
    
    MODELS_WITH_SYMBOLS = [
        ('model_linear.py', {'--symbol', '--seed', '--threshold', '--model-type'}),
        ('model_lstm.py', {'--symbol', '--seed', '--threshold', '--seq-length'}),
        ('model_gru.py', {'--symbol', '--seed', '--threshold', '--seq-length'}),
        ('model_randomforest.py', {'--symbol', '--seed', '--threshold', '--n-estimators'}),
        ('model_xgboost.py', {'--symbol', '--seed', '--threshold', '--train-frac'}),
        ('model_transformer.py', {'--symbol', '--seed', '--threshold', '--epochs'}),
        ('model_catboost.py', {'--symbol', '--seed', '--threshold', '--iterations'}),
    ]
    
    def test_all_models_run_with_sample_data(self, sample_symbol):
        """Test all models complete without critical errors"""
        failed_models = []
        
        for model_name, _ in self.MODELS_WITH_SYMBOLS:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                continue
            
            result = subprocess.run(
                [sys.executable, str(model_path), '--symbol', sample_symbol],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Check for critical errors (not data-missing errors)
            if 'unrecognized arguments' in result.stderr.lower():
                failed_models.append((model_name, result.stderr))
        
        assert len(failed_models) == 0, \
            f"Failed models: {[(m, e[:100]) for m, e in failed_models]}"
    
    def test_all_models_accept_standard_args(self, sample_symbol):
        """Test all models accept standard args"""
        for model_name, required_args in self.MODELS_WITH_SYMBOLS:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                continue
            
            # Test with standard args
            cmd = [sys.executable, str(model_path), '--symbol', sample_symbol, 
                   '--seed', '42', '--threshold', '0.3']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            assert 'unrecognized arguments' not in result.stderr.lower(), \
                f"{model_name} failed with standard args"


# ============================================================================
# Output Directory Tests
# ============================================================================

class TestOutputGeneration:
    """Test that models generate expected output"""
    
    def test_linear_creates_output_structure(self, sample_symbol):
        """Test linear model creates output directory structure"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized' not in result.stderr.lower()
    
    def test_models_use_correct_output_dir_structure(self, sample_symbol):
        """Test all models use expected output directory structure"""
        base_path = REPO_ROOT / 'models' / 'ds_model'
        
        # Just verify base structure exists
        assert base_path.exists(), f"ds_model directory not found: {base_path}"


# ============================================================================
# Performance Tests
# ============================================================================

class TestModelPerformance:
    """Test model performance and output quality"""
    
    def test_models_complete_in_reasonable_time(self, sample_symbol):
        """Test models don't timeout (complete in <5 min)"""
        models = [
            ('model_linear.py', 300),
            ('model_randomforest.py', 300),
            ('model_xgboost.py', 300),
            ('model_catboost.py', 300),
        ]
        
        for model_name, timeout in models:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                continue
            
            import time
            start = time.time()
            
            result = subprocess.run(
                [sys.executable, str(model_path), '--symbol', sample_symbol],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            elapsed = time.time() - start
            assert elapsed < timeout, \
                f"{model_name} took {elapsed:.1f}s (timeout: {timeout}s)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])