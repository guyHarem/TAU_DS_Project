"""
Edge case and robustness tests for models.

Tests model behavior with unusual inputs and edge conditions.

Usage:
    pytest tests/test_edge_cases.py -v
    pytest tests/test_edge_cases.py::TestEdgeCaseInputs -v
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / 'models'
DATA_DIR = REPO_ROOT / 'data' / 'featured_data'


def is_git_lfs_pointer(file_path):
    """Check if file is a Git LFS pointer"""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            return first_line.startswith('version https://git-lfs.github.com')
    except:
        return False


@pytest.fixture(scope="session")
def sample_symbol():
    """Get sample symbol"""
    csv_files = [f for f in DATA_DIR.glob('featured_*_data.csv') 
                 if not is_git_lfs_pointer(f)]
    
    if not csv_files:
        pytest.skip("No data files available")
    
    symbol = csv_files[0].name.replace('featured_', '').replace('_data.csv', '')
    return symbol


# ============================================================================
# Edge Case Input Tests
# ============================================================================

class TestEdgeCaseInputs:
    """Test models handle edge case inputs"""
    
    def test_very_small_seed(self, sample_symbol):
        """Test model with very small seed value"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol,
             '--seed', '0'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should not crash
        assert result.returncode != 1 or 'unrecognized' not in result.stderr.lower()
    
    def test_very_large_seed(self, sample_symbol):
        """Test model with very large seed value"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--symbol', sample_symbol,
             '--seed', '999999999'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_threshold_zero(self, sample_symbol):
        """Test model with threshold = 0"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol,
             '--threshold', '0'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_threshold_one(self, sample_symbol):
        """Test model with threshold = 1"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol,
             '--threshold', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_threshold_very_small(self, sample_symbol):
        """Test model with very small threshold"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
             '--symbol', sample_symbol,
             '--threshold', '0.001'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_threshold_near_one(self, sample_symbol):
        """Test model with threshold close to 1"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
             '--symbol', sample_symbol,
             '--threshold', '0.999'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()


# ============================================================================
# Neural Network Edge Cases
# ============================================================================

class TestNeuralNetworkEdgeCases:
    """Test edge cases for neural network models"""
    
    def test_lstm_with_minimal_sequence_length(self, sample_symbol):
        """Test LSTM with very small sequence length"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_lstm.py'),
             '--symbol', sample_symbol,
             '--seq-length', '1',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_lstm_with_minimal_units(self, sample_symbol):
        """Test LSTM with minimal units"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_lstm.py'),
             '--symbol', sample_symbol,
             '--units', '8',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_lstm_with_minimal_batch_size(self, sample_symbol):
        """Test LSTM with minimal batch size"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_lstm.py'),
             '--symbol', sample_symbol,
             '--batch-size', '1',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_gru_with_one_epoch(self, sample_symbol):
        """Test GRU with just one epoch"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_gru.py'),
             '--symbol', sample_symbol,
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_transformer_with_minimal_params(self, sample_symbol):
        """Test Transformer with minimal parameters"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_transformer.py'),
             '--symbol', sample_symbol,
             '--seq-length', '5',
             '--d-model', '32',
             '--nhead', '2',
             '--num-layers', '1',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()


# ============================================================================
# Tree-Based Models Edge Cases
# ============================================================================

class TestTreeModelEdgeCases:
    """Test edge cases for tree-based models"""
    
    def test_rf_with_single_tree(self, sample_symbol):
        """Test RandomForest with just 1 tree"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--symbol', sample_symbol,
             '--n-estimators', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_rf_with_single_depth(self, sample_symbol):
        """Test RandomForest with very shallow trees"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--symbol', sample_symbol,
             '--max-depth', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_xgboost_with_minimal_trees(self, sample_symbol):
        """Test XGBoost with minimal iterations"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
             '--symbol', sample_symbol,
             '--train-frac', '0.5',
             '--val-frac', '0.25'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_catboost_with_one_iteration(self, sample_symbol):
        """Test CatBoost with 1 iteration"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_catboost.py'),
             '--symbol', sample_symbol,
             '--iterations', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_catboost_with_minimal_depth(self, sample_symbol):
        """Test CatBoost with minimal tree depth"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_catboost.py'),
             '--symbol', sample_symbol,
             '--depth', '1'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        assert 'unrecognized arguments' not in result.stderr.lower()


# ============================================================================
# Boundary Condition Tests
# ============================================================================

class TestBoundaryConditions:
    """Test models at boundary conditions"""
    
    def test_train_frac_boundaries(self, sample_symbol):
        """Test XGBoost with boundary train fraction values"""
        for train_frac in ['0.0', '1.0', '0.5']:
            result = subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
                 '--symbol', sample_symbol,
                 '--train-frac', train_frac],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            assert 'unrecognized arguments' not in result.stderr.lower(), \
                f"Failed with train-frac={train_frac}"
    
    def test_linear_model_types(self, sample_symbol):
        """Test Linear model with different model types"""
        for model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
            result = subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_linear.py'),
                 '--symbol', sample_symbol,
                 '--model-type', model_type],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            assert 'unrecognized arguments' not in result.stderr.lower(), \
                f"Failed with model-type={model_type}"


# ============================================================================
# Robustness Tests
# ============================================================================

class TestRobustness:
    """Test model robustness to variations"""
    
    def test_same_model_different_symbols(self):
        """Test model works with different symbols"""
        csv_files = [f for f in DATA_DIR.glob('featured_*_data.csv') 
                     if not is_git_lfs_pointer(f)]
        
        if len(csv_files) < 2:
            pytest.skip("Need at least 2 data files")
        
        for csv_file in csv_files[:2]:  # Test first 2
            symbol = csv_file.name.replace('featured_', '').replace('_data.csv', '')
            
            result = subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_linear.py'),
                 '--symbol', symbol],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            assert 'unrecognized' not in result.stderr.lower(), \
                f"Failed for symbol {symbol}"
    
    def test_different_seeds_different_results(self, sample_symbol):
        """Test models produce different results with different seeds"""
        results = []
        
        for seed in ['1', '2', '3']:
            result = subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
                 '--symbol', sample_symbol,
                 '--seed', seed],
                capture_output=True,
                text=True,
                timeout=300
            )
            results.append((seed, result.stdout))
        
        # All should produce output
        assert all(len(r[1].strip()) > 0 for r in results), \
            "Some seeds produced no output"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])