"""
CLI tests for all machine learning models.

Usage:
    pytest tests/test_models.py -v
    pytest tests/test_models.py::TestLinearModel -v
    pytest tests/test_models.py -k "xgboost" -v
    pytest tests/test_models.py --tb=short -v
"""

import subprocess
import sys
from pathlib import Path
import pytest

# Project root (tests is one level down from root)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / 'models'

# Ensure paths exist
assert REPO_ROOT.exists(), f"REPO_ROOT not found: {REPO_ROOT}"
assert MODELS_DIR.exists(), f"MODELS_DIR not found: {MODELS_DIR}"


class TestLinearModel:
    """Tests for model_linear.py CLI"""
    
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        """Check that model file exists"""
        model_path = MODELS_DIR / 'model_linear.py'
        if not model_path.exists():
            pytest.skip(f"model_linear.py not found at {model_path}")
    
    def test_linear_args_parsing(self):
        """Test that linear model accepts standard args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'), 
             '--symbol', 'BTCUSD', '--seed', '42', 
             '--threshold', '0.3', '--model-type', 'ridge'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_linear_default_args(self):
        """Test linear model with default args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py')],
            capture_output=True,
            text=True,
            timeout=120
        )
        # Should attempt to run (may fail if data missing)
        assert 'unrecognized' not in result.stderr.lower()


class TestLSTMModel:
    """Tests for model_lstm.py CLI"""
    
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        """Check that model file exists"""
        model_path = MODELS_DIR / 'model_lstm.py'
        if not model_path.exists():
            pytest.skip(f"model_lstm.py not found at {model_path}")
    
    def test_lstm_args_parsing(self):
        """Test that LSTM model accepts standard args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_lstm.py'),
             '--symbol', 'BTCUSD', '--seed', '42',
             '--threshold', '0.3', '--seq-length', '20',
             '--units', '64', '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_lstm_hyphenated_args(self):
        """Test LSTM uses hyphenated args (not underscored)"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_lstm.py'),
             '--seq-length', '20', '--batch-size', '32'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert 'unrecognized argument' not in result.stderr.lower()


class TestGRUModel:
    """Tests for model_gru.py CLI"""
    
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        """Check that model file exists"""
        model_path = MODELS_DIR / 'model_gru.py'
        if not model_path.exists():
            pytest.skip(f"model_gru.py not found at {model_path}")
    
    def test_gru_args_parsing(self):
        """Test that GRU model accepts standard args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_gru.py'),
             '--symbol', 'BTCUSD', '--seed', '42',
             '--threshold', '0.3', '--seq-length', '20',
             '--units', '64', '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert 'unrecognized arguments' not in result.stderr.lower()


class TestRandomForestModel:
    """Tests for model_randomforest.py CLI"""
    
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        """Check that model file exists"""
        model_path = MODELS_DIR / 'model_randomforest.py'
        if not model_path.exists():
            pytest.skip(f"model_randomforest.py not found at {model_path}")
    
    def test_rf_args_parsing(self):
        """Test that RandomForest accepts standard args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--symbol', 'BTCUSD', '--seed', '42',
             '--threshold', '0.3', '--n-estimators', '50',
             '--max-depth', '10'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_rf_hyphenated_args(self):
        """Test RandomForest uses hyphenated args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--n-estimators', '100', '--max-depth', '20'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert 'unrecognized argument' not in result.stderr.lower()


class TestXGBoostModel:
    """Tests for model_xgboost.py CLI"""
    
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        """Check that model file exists"""
        model_path = MODELS_DIR / 'model_xgboost.py'
        if not model_path.exists():
            pytest.skip(f"model_xgboost.py not found at {model_path}")
    
    def test_xgboost_args_parsing(self):
        """Test that XGBoost accepts standard args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
             '--symbol', 'BTCUSD', '--seed', '42',
             '--threshold', '0.3', '--train-frac', '0.7',
             '--val-frac', '0.15'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert 'unrecognized arguments' not in result.stderr.lower()


class TestTransformerModel:
    """Tests for model_transformer.py CLI"""
    
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        """Check that model file exists"""
        model_path = MODELS_DIR / 'model_transformer.py'
        if not model_path.exists():
            pytest.skip(f"model_transformer.py not found at {model_path}")
    
    def test_transformer_args_parsing(self):
        """Test that Transformer accepts standard args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_transformer.py'),
             '--symbol', 'BTCUSD', '--seed', '42',
             '--threshold', '0.3', '--seq-length', '60',
             '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert 'unrecognized arguments' not in result.stderr.lower()


class TestCatBoostModel:
    """Tests for model_catboost.py CLI"""
    
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        """Check that model file exists"""
        model_path = MODELS_DIR / 'model_catboost.py'
        if not model_path.exists():
            pytest.skip(f"model_catboost.py not found at {model_path}")
    
    def test_catboost_args_parsing(self):
        """Test that CatBoost accepts standard args"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_catboost.py'),
             '--symbol', 'BTCUSD', '--seed', '42',
             '--threshold', '0.3', '--iterations', '100',
             '--learning-rate', '0.03', '--depth', '6'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert 'unrecognized arguments' not in result.stderr.lower()


class TestStandardizationAcrossModels:
    """Tests for consistency across all models"""
    
    MODELS = [
        'model_linear.py',
        'model_lstm.py',
        'model_gru.py',
        'model_randomforest.py',
        'model_xgboost.py',
        'model_transformer.py',
        'model_catboost.py',
    ]
    
    def test_all_models_exist(self):
        """Test all models are present"""
        for model_name in self.MODELS:
            model_path = MODELS_DIR / model_name
            assert model_path.exists(), f"{model_name} not found at {model_path}"
    
    def test_all_models_accept_symbol_arg(self):
        """Test all models accept --symbol argument"""
        for model_name in self.MODELS:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                pytest.skip(f"{model_name} not found")
            
            result = subprocess.run(
                [sys.executable, str(model_path), '--symbol', 'TESTBTC'],
                capture_output=True,
                text=True,
                timeout=120
            )
            assert 'unrecognized arguments' not in result.stderr.lower(), \
                f"{model_name} doesn't accept --symbol"
    
    def test_all_models_accept_seed_arg(self):
        """Test all models accept --seed argument"""
        for model_name in self.MODELS:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                pytest.skip(f"{model_name} not found")
            
            result = subprocess.run(
                [sys.executable, str(model_path), '--seed', '42'],
                capture_output=True,
                text=True,
                timeout=120
            )
            assert 'unrecognized arguments' not in result.stderr.lower(), \
                f"{model_name} doesn't accept --seed"
    
    def test_all_models_accept_threshold_arg(self):
        """Test all models accept --threshold argument"""
        for model_name in self.MODELS:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                pytest.skip(f"{model_name} not found")
            
            result = subprocess.run(
                [sys.executable, str(model_path), '--threshold', '0.3'],
                capture_output=True,
                text=True,
                timeout=120
            )
            assert 'unrecognized arguments' not in result.stderr.lower(), \
                f"{model_name} doesn't accept --threshold"


class TestOracleIntegration:
    """Tests for oracle integration"""
    
    def test_oracle_exists(self):
        """Test that arbitrage_oracle.py exists"""
        oracle_path = REPO_ROOT / 'arbitrage_oracle.py'
        assert oracle_path.exists(), f"arbitrage_oracle.py not found at {oracle_path}"


class TestFileStructure:
    """Tests for project file structure"""
    
    def test_models_dir_exists(self):
        """Test that models directory exists"""
        assert MODELS_DIR.exists(), f"Models directory not found: {MODELS_DIR}"
    
    def test_plotter_exists(self):
        """Test that plotter.py exists"""
        plotter_path = MODELS_DIR / 'plotter.py'
        assert plotter_path.exists(), f"plotter.py not found at {plotter_path}"
    
    def test_data_dir_exists(self):
        """Test that data directory exists"""
        data_path = REPO_ROOT / 'data'
        assert data_path.exists(), f"Data directory not found: {data_path}"
    
    def test_featured_data_dir_exists(self):
        """Test that featured_data directory exists"""
        featured_data_path = REPO_ROOT / 'data' / 'featured_data'
        assert featured_data_path.exists(), f"Featured data directory not found: {featured_data_path}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])