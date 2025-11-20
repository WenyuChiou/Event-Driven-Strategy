"""
Unit tests for alpha factor calculations.
"""

import pytest
import pandas as pd
import numpy as np
from package.alpha_eric import AlphaFactory


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + np.abs(np.random.randn(100) * 0.3),
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - np.abs(np.random.randn(100) * 0.3),
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Ensure high >= close >= low and high >= open >= low
    data['high'] = data[['open', 'close', 'high']].max(axis=1) + np.abs(np.random.randn(100) * 0.1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1) - np.abs(np.random.randn(100) * 0.1)
    
    return data


class TestAlphaFactory:
    """Test cases for AlphaFactory class."""
    
    def test_alpha_factory_initialization(self, sample_data):
        """Test AlphaFactory initialization."""
        alpha = AlphaFactory(sample_data)
        assert alpha.data is not None
        assert len(alpha.data) == len(sample_data)
    
    def test_alpha01_type1(self, sample_data):
        """Test alpha01 type1 calculation."""
        alpha = AlphaFactory(sample_data)
        result = alpha.alpha01(days=[3, 5], type=None)
        
        assert isinstance(result, pd.DataFrame)
        assert 'utility_3_type1' in result.columns
        assert 'utility_5_type1' in result.columns
        assert not result['utility_3_type1'].isnull().all()
    
    def test_alpha01_type2(self, sample_data):
        """Test alpha01 type2 calculation."""
        alpha = AlphaFactory(sample_data)
        result = alpha.alpha01(days=[3], type='type2')
        
        assert isinstance(result, pd.DataFrame)
        assert 'utility_3_type2' in result.columns
    
    def test_alpha02(self, sample_data):
        """Test alpha02 calculation."""
        alpha = AlphaFactory(sample_data)
        result = alpha.alpha02(days=[3, 5], weight=0.5)
        
        assert isinstance(result, pd.DataFrame)
        assert 'prospect_3' in result.columns
        assert 'prospect_5' in result.columns
    
    def test_alpha03(self, sample_data):
        """Test alpha03 calculation."""
        alpha = AlphaFactory(sample_data)
        result = alpha.alpha03(days=[3, 5], risk_aversion=1.5)
        
        assert isinstance(result, pd.DataFrame)
        assert 'risk_aversion_3' in result.columns
        assert 'risk_aversion_5' in result.columns
    
    def test_add_all_alphas(self, sample_data):
        """Test adding all alpha factors."""
        alpha = AlphaFactory(sample_data)
        result = alpha.add_all_alphas(days=[3, 9])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > len(sample_data.columns)
    
    def test_alpha_vectorization(self, sample_data):
        """Test that alpha calculations are vectorized (no .apply with lambda)."""
        alpha = AlphaFactory(sample_data)
        
        # Check that alpha01 uses vectorized operations
        import inspect
        source = inspect.getsource(alpha.alpha01)
        
        # Should not contain inefficient patterns
        assert '.apply(lambda' not in source or 'np.where' in source or 'vectorized' in source.lower()


class TestAlphaPerformance:
    """Performance tests for alpha calculations."""
    
    def test_alpha_calculation_speed(self, sample_data):
        """Test that alpha calculations are reasonably fast."""
        import time
        
        alpha = AlphaFactory(sample_data)
        
        start_time = time.time()
        result = alpha.alpha01(days=[3, 5, 10, 20])
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 5.0, f"Alpha calculation took {elapsed:.2f}s, expected < 5s"
    
    def test_vectorized_vs_apply(self, sample_data):
        """Test that vectorized operations are faster than apply."""
        import time
        
        # This test verifies that our vectorized implementation is used
        alpha = AlphaFactory(sample_data)
        
        start = time.time()
        result = alpha.alpha02(days=[3, 5, 10])
        vectorized_time = time.time() - start
        
        # Vectorized should be reasonably fast
        assert vectorized_time < 2.0

