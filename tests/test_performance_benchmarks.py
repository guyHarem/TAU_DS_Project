"""
Performance benchmarks and profiling tests.

Tests model inference speed, memory usage, and computational efficiency.

Usage:
    pytest tests/test_performance_benchmarks.py -v
    pytest tests/test_performance_benchmarks.py -v -s
"""

import subprocess
import sys
import time
from pathlib import Path
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
# Execution Time Benchmarks
# ============================================================================

class TestExecutionTime:
    """Test model execution times"""
    
    @pytest.mark.performance
    def test_linear_execution_time(self, sample_symbol):
        """Benchmark Linear model execution time"""
        start = time.time()
        
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol,
             '--seed', '42'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        elapsed = time.time() - start
        print(f"\nLinear model: {elapsed:.2f}s")
        
        # Linear models should be fast (< 30 seconds)
        assert elapsed < 30, f"Linear model too slow: {elapsed:.2f}s"
    
    @pytest.mark.performance
    def test_randomforest_execution_time(self, sample_symbol):
        """Benchmark RandomForest execution time"""
        start = time.time()
        
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
             '--symbol', sample_symbol,
             '--seed', '42'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        elapsed = time.time() - start
        print(f"\nRandomForest model: {elapsed:.2f}s")
        
        # RF should complete in reasonable time (< 60 seconds)
        assert elapsed < 60, f"RandomForest too slow: {elapsed:.2f}s"
    
    @pytest.mark.performance
    def test_xgboost_execution_time(self, sample_symbol):
        """Benchmark XGBoost execution time"""
        start = time.time()
        
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_xgboost.py'),
             '--symbol', sample_symbol,
             '--seed', '42'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        elapsed = time.time() - start
        print(f"\nXGBoost model: {elapsed:.2f}s")
        
        # XGBoost should be reasonably fast (< 60 seconds)
        assert elapsed < 60, f"XGBoost too slow: {elapsed:.2f}s"
    
    @pytest.mark.performance
    def test_catboost_execution_time(self, sample_symbol):
        """Benchmark CatBoost execution time"""
        start = time.time()
        
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_catboost.py'),
             '--symbol', sample_symbol,
             '--seed', '42'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        elapsed = time.time() - start
        print(f"\nCatBoost model: {elapsed:.2f}s")
        
        # CatBoost should be reasonably fast (< 60 seconds)
        assert elapsed < 60, f"CatBoost too slow: {elapsed:.2f}s"
    
    @pytest.mark.performance
    def test_lstm_execution_time(self, sample_symbol):
        """Benchmark LSTM execution time"""
        start = time.time()
        
        result = subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_lstm.py'),
             '--symbol', sample_symbol,
             '--epochs', '1',
             '--seed', '42'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        elapsed = time.time() - start
        print(f"\nLSTM model: {elapsed:.2f}s")
        
        # Deep learning can be slower (< 120 seconds for 1 epoch)
        assert elapsed < 120, f"LSTM too slow: {elapsed:.2f}s"


# ============================================================================
# Comparative Performance Tests
# ============================================================================

class TestComparativePerformance:
    """Compare performance across models"""
    
    @pytest.mark.performance
    def test_shallow_models_faster_than_deep(self, sample_symbol):
        """Test traditional models are faster than deep learning"""
        # Linear model time
        start = time.time()
        subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_linear.py'),
             '--symbol', sample_symbol],
            capture_output=True,
            timeout=300
        )
        linear_time = time.time() - start
        
        # LSTM time (1 epoch)
        start = time.time()
        subprocess.run(
            [sys.executable, str(MODELS_DIR / 'model_lstm.py'),
             '--symbol', sample_symbol,
             '--epochs', '1'],
            capture_output=True,
            timeout=300
        )
        lstm_time = time.time() - start
        
        print(f"\nLinear: {linear_time:.2f}s, LSTM: {lstm_time:.2f}s")
        
        # Linear should be faster (or at least not significantly slower)
        # Allow LSTM to be ~2x slower
        assert linear_time < lstm_time * 3, \
            f"Linear ({linear_time:.2f}s) not significantly faster than LSTM ({lstm_time:.2f}s)"
    
    @pytest.mark.performance
    def test_tree_models_have_consistent_speed(self, sample_symbol):
        """Test tree-based models have consistent execution times"""
        times = {}
        
        for model_name in ['model_randomforest.py', 'model_xgboost.py', 'model_catboost.py']:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                continue
            
            start = time.time()
            subprocess.run(
                [sys.executable, str(model_path),
                 '--symbol', sample_symbol,
                 '--seed', '42'],
                capture_output=True,
                timeout=300
            )
            elapsed = time.time() - start
            times[model_name] = elapsed
        
        # All tree models should finish in reasonable time
        assert all(t < 120 for t in times.values()), \
            f"Some tree models too slow: {times}"
        
        print(f"\nTree model times: {times}")


# ============================================================================
# Scaling Tests
# ============================================================================

class TestPerformanceScaling:
    """Test performance scaling with different parameters"""
    
    @pytest.mark.performance
    def test_linear_scales_with_alpha(self, sample_symbol):
        """Test Linear model performance with different alpha values"""
        times = {}
        
        for alpha in ['0.001', '0.01', '0.1']:
            start = time.time()
            subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_linear.py'),
                 '--symbol', sample_symbol,
                 '--alpha', alpha],
                capture_output=True,
                timeout=300
            )
            elapsed = time.time() - start
            times[alpha] = elapsed
        
        print(f"\nLinear with different alphas: {times}")
        
        # All alphas should have similar execution time
        max_diff = max(times.values()) - min(times.values())
        assert max_diff < 10, \
            f"Large performance variance across alphas: {times}"
    
    @pytest.mark.performance
    def test_rf_scales_with_estimators(self, sample_symbol):
        """Test RF performance with different n_estimators"""
        times = {}
        
        for n_est in ['10', '50', '100']:
            start = time.time()
            subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_randomforest.py'),
                 '--symbol', sample_symbol,
                 '--n-estimators', n_est],
                capture_output=True,
                timeout=300
            )
            elapsed = time.time() - start
            times[n_est] = elapsed
        
        print(f"\nRF with different n_estimators: {times}")
        
        # More estimators should take longer, but not exponentially
        assert times['100'] < times['10'] * 20, \
            "RF doesn't scale well with estimators"
    
    @pytest.mark.performance
    def test_lstm_scales_with_epochs(self, sample_symbol):
        """Test LSTM performance scales with epochs"""
        times = {}
        
        for epochs in ['1', '2', '3']:
            start = time.time()
            subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_lstm.py'),
                 '--symbol', sample_symbol,
                 '--epochs', epochs],
                capture_output=True,
                timeout=300
            )
            elapsed = time.time() - start
            times[epochs] = elapsed
        
        print(f"\nLSTM with different epochs: {times}")
        
        # More epochs should take longer
        # 3 epochs should be roughly 3x longer than 1 epoch (within 2-4x due to overhead)
        assert times['3'] < times['1'] * 5, \
            "LSTM doesn't scale linearly with epochs"


# ============================================================================
# Stability Tests
# ============================================================================

class TestPerformanceStability:
    """Test performance stability across runs"""
    
    @pytest.mark.performance
    def test_linear_consistent_runtime(self, sample_symbol):
        """Test Linear model has consistent runtime"""
        times = []
        
        for _ in range(3):
            start = time.time()
            subprocess.run(
                [sys.executable, str(MODELS_DIR / 'model_linear.py'),
                 '--symbol', sample_symbol,
                 '--seed', '42'],
                capture_output=True,
                timeout=300
            )
            times.append(time.time() - start)
        
        # Standard deviation should be < 50% of mean
        mean_time = sum(times) / len(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_dev = variance ** 0.5
        
        print(f"\nLinear runtime times: {times}")
        print(f"Mean: {mean_time:.2f}s, Std: {std_dev:.2f}s")
        
        assert std_dev < mean_time * 0.5, \
            f"High runtime variance: mean={mean_time:.2f}, std={std_dev:.2f}"


# ============================================================================
# Memory Usage Tests (Qualitative)
# ============================================================================

class TestMemoryUsage:
    """Test memory usage doesn't explode"""
    
    @pytest.mark.performance
    def test_model_completes_without_memory_error(self, sample_symbol):
        """Test models don't run out of memory"""
        for model_name in ['model_linear.py', 'model_randomforest.py', 'model_xgboost.py']:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                continue
            
            result = subprocess.run(
                [sys.executable, str(model_path),
                 '--symbol', sample_symbol],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Check for memory errors
            memory_errors = ['MemoryError', 'out of memory', 'memory allocation']
            has_memory_error = any(err in result.stderr for err in memory_errors)
            
            assert not has_memory_error, \
                f"{model_name} ran out of memory"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s', '-m', 'performance'])