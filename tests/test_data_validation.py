"""
Data validation and preprocessing tests.

Tests for data quality, feature engineering, and data pipeline.

Usage:
    pytest tests/test_data_validation.py -v
    pytest tests/test_data_validation.py::TestDataQuality -v
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
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
def sample_data():
    """Load sample data for testing"""
    csv_files = [f for f in DATA_DIR.glob('featured_*_data.csv') 
                 if not is_git_lfs_pointer(f)]
    
    if not csv_files:
        pytest.skip("No data files available")
    
    df = pd.read_csv(csv_files[0])
    return df, csv_files[0].name


# ============================================================================
# Data Type Tests
# ============================================================================

class TestDataTypes:
    """Test data types are correct"""
    
    def test_target_is_numeric(self, sample_data):
        """Test target variable is numeric"""
        df, _ = sample_data
        assert pd.api.types.is_numeric_dtype(df['spread_close_pct']), \
            "Target 'spread_close_pct' must be numeric"
    
    def test_numeric_features_are_numeric(self, sample_data):
        """Test numeric features have correct types"""
        df, _ = sample_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 1, "Should have multiple numeric features"
    
    def test_no_object_columns_except_time(self, sample_data):
        """Test no unexpected object/string columns"""
        df, _ = sample_data
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Only allow 'time' column as object
        allowed = {'time', 'Time', 'timestamp', 'Timestamp', 'date', 'Date'}
        unexpected = [col for col in object_cols if col.lower() not in {a.lower() for a in allowed}]
        
        assert len(unexpected) == 0, f"Unexpected object columns: {unexpected}"


# ============================================================================
# Missing Data Tests
# ============================================================================

class TestMissingData:
    """Test handling of missing data"""
    
    def test_no_missing_target(self, sample_data):
        """Test target has no missing values"""
        df, _ = sample_data
        assert df['spread_close_pct'].isnull().sum() == 0, \
            f"Target has {df['spread_close_pct'].isnull().sum()} missing values"
    
    def test_missing_data_below_threshold(self, sample_data):
        """Test missing data is below 20% threshold"""
        df, _ = sample_data
        missing_ratio = (df.isnull().sum() / len(df) * 100)
        
        for col, ratio in missing_ratio.items():
            assert ratio < 20, f"{col} has {ratio:.2f}% missing data (threshold: 20%)"
    
    def test_critical_features_not_missing(self, sample_data):
        """Test critical features have no missing values"""
        df, _ = sample_data
        
        # Features that shouldn't be missing
        critical_cols = [col for col in df.columns 
                        if any(x in col.lower() for x in ['price', 'volume', 'spread'])]
        
        for col in critical_cols:
            assert df[col].isnull().sum() == 0, \
                f"Critical feature {col} has missing values"
    
    def test_total_missing_count(self, sample_data):
        """Test total missing values across dataset"""
        df, _ = sample_data
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (total_missing / total_cells) * 100
        
        assert missing_pct < 5, \
            f"Total missing data: {missing_pct:.2f}% (threshold: 5%)"


# ============================================================================
# Outlier Detection Tests
# ============================================================================

class TestOutliers:
    """Test data for extreme outliers"""
    
    def test_target_no_extreme_outliers(self, sample_data):
        """Test target variable has no extreme outliers"""
        df, _ = sample_data
        target = df['spread_close_pct']
        
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 10 * IQR  # 10 * IQR is extreme
        upper_bound = Q3 + 10 * IQR
        
        outliers = ((target < lower_bound) | (target > upper_bound)).sum()
        assert outliers == 0, f"Found {outliers} extreme outliers in target"
    
    def test_numeric_features_in_reasonable_range(self, sample_data):
        """Test numeric features are in reasonable ranges"""
        df, _ = sample_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'spread_close_pct':
                continue
            
            # Check for inf values
            assert not np.isinf(df[col]).any(), f"{col} contains inf values"
            
            # Check for extremely large values (> 1e6 for most features)
            if col not in ['buy_volume', 'sell_volume', 'trade_rate']:
                assert df[col].abs().max() < 1e6, f"{col} has extremely large values"
    
    def test_no_infinite_values(self, sample_data):
        """Test no infinite values in data"""
        df, _ = sample_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            assert inf_count == 0, f"{col} has {inf_count} inf values"


# ============================================================================
# Feature Distribution Tests
# ============================================================================

class TestFeatureDistribution:
    """Test feature distributions are reasonable"""
    
    def test_target_has_variance(self, sample_data):
        """Test target has sufficient variance"""
        df, _ = sample_data
        target = df['spread_close_pct']
        
        # Standard deviation should be > 0
        assert target.std() > 0, "Target has zero variance"
        
        # Coefficient of variation should be reasonable
        cv = target.std() / abs(target.mean()) if target.mean() != 0 else float('inf')
        assert cv < 100, f"Target has extreme coefficient of variation: {cv}"
    
    def test_target_range_reasonable(self, sample_data):
        """Test target is in expected range"""
        df, _ = sample_data
        target = df['spread_close_pct']
        
        # For percentage spread, should be between -100% and +100%
        assert target.min() >= -100, f"Target min too low: {target.min()}"
        assert target.max() <= 100, f"Target max too high: {target.max()}"
    
    def test_no_constant_features(self, sample_data):
        """Test no features are completely constant"""
        df, _ = sample_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            assert df[col].std() > 0, f"Feature {col} is constant (zero variance)"
    
    def test_price_features_positive(self, sample_data):
        """Test price features are positive"""
        df, _ = sample_data
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        
        for col in price_cols:
            assert (df[col] >= 0).all(), f"Price feature {col} has negative values"
    
    def test_volume_features_positive(self, sample_data):
        """Test volume features are positive or zero"""
        df, _ = sample_data
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        
        for col in volume_cols:
            assert (df[col] >= 0).all(), f"Volume feature {col} has negative values"


# ============================================================================
# Data Consistency Tests
# ============================================================================

class TestDataConsistency:
    """Test data consistency and logical rules"""
    
    def test_spreads_symmetric(self, sample_data):
        """Test bid-ask spreads follow logical consistency"""
        df, _ = sample_data
        
        if 'buy_price' in df.columns and 'sell_price' in df.columns:
            # Buy price should be <= sell price (typically)
            inconsistent = (df['buy_price'] > df['sell_price']).sum()
            total = len(df)
            ratio = inconsistent / total
            
            assert ratio < 0.1, \
                f"Buy price > sell price in {ratio*100:.2f}% of rows (threshold: 10%)"
    
    def test_volume_consistency(self, sample_data):
        """Test volume ratios are reasonable"""
        df, _ = sample_data
        
        if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
            # Ratio should be between 0.1 and 10
            ratio = df['buy_volume'] / (df['sell_volume'] + 1e-10)
            
            extreme_ratios = ((ratio < 0.01) | (ratio > 100)).sum()
            assert extreme_ratios / len(df) < 0.05, \
                "Volume ratios are unrealistic"
    
    def test_no_duplicate_timestamps(self, sample_data):
        """Test no duplicate timestamps"""
        df, _ = sample_data
        
        if 'time' in df.columns:
            duplicates = df['time'].duplicated().sum()
            assert duplicates == 0, f"Found {duplicates} duplicate timestamps"


# ============================================================================
# Feature Scaling Tests
# ============================================================================

class TestFeatureScaling:
    """Test features are in appropriate scales"""
    
    def test_features_reasonable_scale(self, sample_data):
        """Test features are in reasonable scale"""
        df, _ = sample_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            
            # Check for features with extreme scales
            if std > 0:
                # If mean >> std, might have scaling issues
                ratio = abs(mean) / std
                assert ratio < 1000, \
                    f"{col} has extreme scale ratio: {ratio:.2f}"


# ============================================================================
# Data Volume Tests
# ============================================================================

class TestDataVolume:
    """Test sufficient data volume"""
    
    def test_minimum_samples(self, sample_data):
        """Test minimum number of samples"""
        df, _ = sample_data
        assert len(df) >= 100, f"Only {len(df)} samples, need >= 100"
    
    def test_minimum_features(self, sample_data):
        """Test minimum number of features"""
        df, _ = sample_data
        assert len(df.columns) >= 5, f"Only {len(df.columns)} features, need >= 5"
    
    def test_sample_feature_ratio(self, sample_data):
        """Test samples-to-features ratio is reasonable"""
        df, _ = sample_data
        ratio = len(df) / len(df.columns)
        
        # At least 10 samples per feature
        assert ratio >= 10, \
            f"Sample-to-feature ratio {ratio:.2f} too low (need >= 10)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])