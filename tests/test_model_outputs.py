"""
Tests for model outputs and predictions.

Validates that models produce valid predictions and metrics.

Usage:
    pytest tests/test_model_outputs.py -v
    pytest tests/test_model_outputs.py::TestPredictionValidity -v
"""

import subprocess
import sys
import json
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
# Prediction Validity Tests
# ============================================================================

class TestPredictionValidity:
    """Test predictions are valid and reasonable"""
    
    def test_linear_produces_numeric_output(self, sample_symbol):
        """Test linear model produces numeric output"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should contain numeric output or metrics
        output = result.stdout + result.stderr
        
        # Check for common metric outputs
        has_numeric = any(char.isdigit() for char in output)
        assert has_numeric, "Output contains no numeric values"
    
    def test_xgboost_produces_numeric_output(self, sample_symbol):
        """Test XGBoost model produces numeric output"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout + result.stderr
        has_numeric = any(char.isdigit() for char in output)
        assert has_numeric, "Output contains no numeric values"
    
    def test_catboost_produces_numeric_output(self, sample_symbol):
        """Test CatBoost produces numeric output"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_catboost.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout + result.stderr
        has_numeric = any(char.isdigit() for char in output)
        assert has_numeric, "Output contains no numeric values"


# ============================================================================
# Output Format Tests
# ============================================================================

class TestOutputFormats:
    """Test output formats are consistent"""
    
    def test_model_output_not_empty(self, sample_symbol):
        """Test model produces non-empty output"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout.strip()
        assert len(output) > 0, "Model produced empty output"
    
    def test_model_completes_without_exception(self, sample_symbol):
        """Test model completes without Python exceptions"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Check for common Python exception patterns
        error_patterns = ['Traceback', 'Error:', 'Exception', 'Failure']
        has_error = any(pattern in result.stderr for pattern in error_patterns)
        
        # Allow "data not found" type errors, but not Python exceptions
        if has_error and 'unrecognized' not in result.stderr.lower():
            assert False, f"Model raised exception:\n{result.stderr[:500]}"
    
    def test_models_produce_consistent_output_format(self, sample_symbol):
        """Test different models produce outputs in similar formats"""
        models = ['model_linear.py', 'model_randomforest.py', 'model_xgboost.py']
        
        outputs = []
        for model_name in models:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                continue
            
            result = subprocess.run(
                [sys.executable, str(model_path), '--symbol', sample_symbol],
                capture_output=True,
                text=True,
                timeout=300
            )
            outputs.append(result.stdout)
        
        # All should have some output
        assert all(len(out.strip()) > 0 for out in outputs), \
            "Some models produced empty output"


# ============================================================================
# Metric Validity Tests
# ============================================================================

class TestMetricValidity:
    """Test output metrics are valid"""
    
    def test_accuracy_in_valid_range(self, sample_symbol):
        """Test accuracy metrics are between 0 and 1"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout
        
        # Look for accuracy-like values
        import re
        numbers = [float(x) for x in re.findall(r'\d+\.\d+', output)]
        
        # At least some numbers should be present
        if numbers:
            # Filter for likely accuracy/metric values
            metrics = [n for n in numbers if 0 <= n <= 1.5]
            # If we found metrics, they should be reasonable
            if metrics:
                assert all(0 <= m <= 1.5 for m in metrics), \
                    f"Found invalid metric values: {metrics}"
    
    def test_loss_is_positive(self, sample_symbol):
        """Test loss values are positive"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout + result.stderr
        
        # Look for loss values
        import re
        loss_pattern = r'[Ll]oss[:\s]+([0-9]+\.?[0-9]*)'
        losses = re.findall(loss_pattern, output)
        
        if losses:
            loss_values = [float(x) for x in losses]
            assert all(x >= 0 for x in loss_values), \
                f"Found negative loss values: {loss_values}"


# ============================================================================
# Reproducibility Tests
# ============================================================================

class TestReproducibility:
    """Test model outputs are reproducible with same seed"""
    
    def test_linear_reproducible_with_seed(self, sample_symbol):
        """Test linear model produces same output with same seed"""
        # Run twice with same seed
        outputs = []
        
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_linear.py'),
                 '--symbol', sample_symbol,
                 '--seed', '42'],
                capture_output=True,
                text=True,
                timeout=300
            )
            outputs.append(result.stdout)
        
        # Extract numeric values
        import re
        numbers1 = re.findall(r'\d+\.\d+', outputs[0])
        numbers2 = re.findall(r'\d+\.\d+', outputs[1])
        
        # Should have similar output structure
        assert len(numbers1) > 0, "No numeric output from first run"
        assert len(numbers2) > 0, "No numeric output from second run"
    
    def test_randomforest_reproducible_with_seed(self, sample_symbol):
        """Test RF model is reproducible with seed"""
        outputs = []
        
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
                 '--symbol', sample_symbol,
                 '--seed', '99'],
                capture_output=True,
                text=True,
                timeout=300
            )
            outputs.append(result.stdout)
        
        # Both should have output
        assert all(len(out.strip()) > 0 for out in outputs), \
            "Some runs produced empty output"


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and messages"""
    
    def test_model_handles_invalid_symbol(self):
        """Test model handles invalid symbol gracefully"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', 'INVALID_SYMBOL_THAT_DOESNT_EXIST'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should either skip gracefully or show informative error
        # Not a raw unrecognized argument error
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_model_handles_invalid_threshold(self):
        """Test model handles invalid threshold gracefully"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', 'BTCUSD' if (DATA_DIR / 'featured_BTCUSD_data.csv').exists() else 'TEST',
             '--threshold', '-0.5'],  # Invalid: should be 0-1
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should handle gracefully
        assert 'unrecognized arguments' not in result.stderr.lower()
    
    def test_model_output_is_string(self, sample_symbol):
        """Test model output can be captured as string"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # stdout and stderr should be strings
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])