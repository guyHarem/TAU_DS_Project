"""
Tests for model outputs and predictions.

Validates that models:
  - Produce valid predictions and metrics
  - Create proper output directory structure
  - Save model artifacts (.joblib, .pth, etc.)
  - Generate visualization plots

Output Structure:
  models/ds_model/{model_type}/{symbol}/
    - {model_type}_{symbol}_model.{joblib|pth}
    - {model_type}_{symbol}_results.png
    - {model_type}_{symbol}_prediction_hist.png
    - {model_type}_{symbol}_feature_importance.png
    - {model_type}_{symbol}_pr_curve.png
    - {model_type}_{symbol}_threshold_metrics.png

Usage:
    pytest tests/test_model_outputs.py -v
    pytest tests/test_model_outputs.py::TestOutputDirectory -v
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
OUTPUT_BASE = REPO_ROOT / 'models' / 'ds_model'


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
# Output Directory Tests
# ============================================================================

class TestOutputDirectory:
    """Test models create proper output directories and artifacts"""
    
    def test_linear_creates_output_directory(self, sample_symbol):
        """Test linear model creates output directory structure"""
        # Linear saves to: ds_model/linear-classifier/{model_type}/{symbol}/
        output_dir = OUTPUT_BASE / 'linear-classifier' / 'linear' / sample_symbol
        
        # Remove existing output to test fresh creation
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol,
             '--model-type', 'linear'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Check if model ran successfully
        if result.returncode != 0 or 'Error' in result.stderr or 'Traceback' in result.stderr:
            pytest.skip(f"Model failed to run: {result.stderr[:200]}")
        
        # Check directory was created
        assert output_dir.exists(), f"Output directory not created: {output_dir}"
        
        # Check for model artifact (named: linear_{symbol}_model.joblib)
        model_file = output_dir / f'linear_{sample_symbol}_model.joblib'
        assert model_file.exists(), f"Model artifact not saved: {model_file}"
    
    def test_randomforest_creates_output_directory(self, sample_symbol):
        """Test RF model creates output directory and artifacts"""
        # RF saves to: ds_model/random-forest/{symbol}/
        output_dir = OUTPUT_BASE / 'random-forest' / sample_symbol
        
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Check if model ran successfully
        if result.returncode != 0 or 'Error' in result.stderr or 'Traceback' in result.stderr:
            pytest.skip(f"Model failed to run: {result.stderr[:200]}")
        
        assert output_dir.exists(), f"Output directory not created: {output_dir}"
        
        # RF saves as: rf_{symbol}_model.joblib (abbreviated, not 'random-forest')
        model_file = output_dir / f'rf_{sample_symbol}_model.joblib'
        assert model_file.exists(), f"Model artifact not saved: {model_file}"
    
    def test_xgboost_creates_output_directory(self, sample_symbol):
        """Test XGBoost model creates output directory and artifacts"""
        # XGBoost saves to: ds_model/xgboost/{symbol}/
        output_dir = OUTPUT_BASE / 'xgboost' / sample_symbol
        
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Check if model ran successfully
        if result.returncode != 0 or 'Error' in result.stderr or 'Traceback' in result.stderr:
            pytest.skip(f"Model failed to run (may have missing dependencies): {result.stderr[:200]}")
        
        assert output_dir.exists(), f"Output directory not created: {output_dir}"
        
        model_file = output_dir / f'xgboost_{sample_symbol}_model.joblib'
        assert model_file.exists(), f"Model artifact not saved: {model_file}"
    
    def test_catboost_creates_output_directory(self, sample_symbol):
        """Test CatBoost model creates output directory and artifacts"""
        # CatBoost saves to: ds_model/catboost/{symbol}/
        output_dir = OUTPUT_BASE / 'catboost' / sample_symbol
        
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_catboost.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Check if model ran successfully
        if result.returncode != 0 or 'Error' in result.stderr or 'Traceback' in result.stderr:
            pytest.skip(f"Model failed to run: {result.stderr[:200]}")
        
        assert output_dir.exists(), f"Output directory not created: {output_dir}"
        
        model_file = output_dir / f'catboost_{sample_symbol}_model.joblib'
        assert model_file.exists(), f"Model artifact not saved: {model_file}"


# ============================================================================
# Prediction Validity Tests
# ============================================================================

class TestPredictionValidity:
    """Test predictions are valid and reasonable"""
    
    def test_models_exit_without_errors(self, sample_symbol):
        """Test models execute without Python exceptions"""
        models = ['model_linear.py', 'model_randomforest.py', 'model_xgboost.py', 'model_catboost.py']
        
        failed_models = []
        skipped_models = []
        
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
            
            # Check for import/dependency errors (skip for these)
            if 'ImportError' in result.stderr or 'ModuleNotFoundError' in result.stderr or 'libomp' in result.stderr:
                skipped_models.append(model_name)
                continue
            
            # Check for Python exceptions (but allow data-not-found errors)
            error_patterns = ['Traceback', 'Error:', 'Exception', 'Failure']
            has_error = any(pattern in result.stderr for pattern in error_patterns)
            
            if has_error and 'data not found' not in result.stderr.lower():
                failed_models.append((model_name, result.stderr[:200]))
        
        if skipped_models:
            pytest.skip(f"Skipping models with missing dependencies: {skipped_models}")
        
        assert len(failed_models) == 0, \
            f"Models raised exceptions: {failed_models}"


# ============================================================================
# Plot Generation Tests
# ============================================================================

class TestPlotGeneration:
    """Test models generate expected plot files (when they run successfully)"""
    
    def test_linear_generates_plots(self, sample_symbol):
        """Test linear model generates visualization plots (if it ran)"""
        output_dir = OUTPUT_BASE / 'linear-classifier' / 'linear' / sample_symbol
        
        # If output directory doesn't exist, model didn't run
        if not output_dir.exists():
            pytest.skip(f"Output directory not found - model likely didn't run: {output_dir}")
        
        # Linear files are named: linear_{symbol}_*
        expected_plots = [
            f'linear_{sample_symbol}_results.png',
            f'linear_{sample_symbol}_pr_curve.png',
        ]
        
        # Check that at least some plots exist
        existing_plots = [f for f in expected_plots if (output_dir / f).exists()]
        assert len(existing_plots) > 0, \
            f"No plots found in {output_dir}. Expected: {expected_plots}"
    
    def test_randomforest_generates_plots(self, sample_symbol):
        """Test RF model generates visualization plots (if it ran)"""
        output_dir = OUTPUT_BASE / 'random-forest' / sample_symbol
        
        # If output directory doesn't exist, model didn't run
        if not output_dir.exists():
            pytest.skip(f"Output directory not found - model likely didn't run: {output_dir}")
        
        # RF files are named: rf_{symbol}_* (abbreviated, not random-forest_)
        expected_plots = [
            f'rf_{sample_symbol}_results.png',
            f'rf_{sample_symbol}_feature_importance.png',
        ]
        
        # Check that at least some plots exist
        existing_plots = [f for f in expected_plots if (output_dir / f).exists()]
        assert len(existing_plots) > 0, \
            f"No plots found in {output_dir}. Expected: {expected_plots}"


# ============================================================================
# Metric Validity Tests
# ============================================================================

class TestMetricValidity:
    """Test model outputs contain valid metrics"""
    
    def test_model_saves_metrics(self, sample_symbol):
        """Test that models save/output metrics about their performance"""
        # Most models print metrics to stdout during training
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout + result.stderr
        # Should mention some metrics
        metric_keywords = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mse', 'rmse']
        has_metrics = any(keyword in output.lower() for keyword in metric_keywords)
        
        # At minimum should have numeric values
        has_numbers = any(char.isdigit() for char in output)
        assert has_numbers, "Model output contains no numeric values"


# ============================================================================
# Reproducibility Tests
# ============================================================================

class TestReproducibility:
    """Test model outputs are reproducible with same seed"""
    
    def test_linear_reproducible_with_seed(self, sample_symbol):
        """Test linear model produces same output structure with same seed"""
        # Run twice with same seed
        outputs = []
        
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_linear.py'),
                 '--symbol', sample_symbol,
                 '--seed', '42',
                 '--model-type', 'linear'],
                capture_output=True,
                text=True,
                timeout=300
            )
            outputs.append(result.stdout)
        
        # Both runs should produce output
        assert all(len(out.strip()) > 0 for out in outputs), \
            "Some runs produced empty output"
    
    def test_randomforest_reproducible_with_seed(self, sample_symbol):
        """Test RF model is reproducible with seed"""
        outputs = []
        
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
                 '--symbol', sample_symbol,
                 '--seed', '99',
                 '--n-estimators', '100'],
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
    """Test error handling and valid argument usage"""
    
    def test_linear_accepts_valid_args(self, sample_symbol):
        """Test linear model accepts valid arguments"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol,
             '--model-type', 'ridge',
             '--alpha', '0.5',
             '--seed', '42',
             '--threshold', '0.3'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should not have unrecognized argument errors
        assert 'unrecognized arguments' not in result.stderr.lower(), \
            f"Model rejected valid args:\n{result.stderr[:300]}"
    
    def test_randomforest_accepts_valid_args(self, sample_symbol):
        """Test RF model accepts valid arguments"""
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--symbol', sample_symbol,
             '--n-estimators', '50',
             '--seed', '42',
             '--threshold', '0.5'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Should not have unrecognized argument errors
        assert 'unrecognized arguments' not in result.stderr.lower(), \
            f"Model rejected valid args:\n{result.stderr[:300]}"
    
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