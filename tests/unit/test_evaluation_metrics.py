"""
Unit tests for evaluation metrics.
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown
)


@pytest.fixture
def sample_returns():
    """Create sample returns for testing."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(100) * 0.01)
    return returns


@pytest.fixture
def positive_returns():
    """Create consistently positive returns."""
    return pd.Series([0.01] * 100)


@pytest.fixture
def negative_returns():
    """Create consistently negative returns."""
    return pd.Series([-0.01] * 100)


class TestSharpeRatio:
    """Test cases for Sharpe ratio calculation."""
    
    def test_sharpe_ratio_basic(self, sample_returns):
        """Test basic Sharpe ratio calculation."""
        sharpe = calculate_sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)
    
    def test_sharpe_ratio_with_risk_free_rate(self, sample_returns):
        """Test Sharpe ratio with risk-free rate."""
        sharpe = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        assert isinstance(sharpe, float)
    
    def test_sharpe_ratio_positive_returns(self, positive_returns):
        """Test Sharpe ratio with positive returns."""
        sharpe = calculate_sharpe_ratio(positive_returns)
        assert sharpe > 0
    
    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        returns = pd.Series([0.01] * 100)
        sharpe = calculate_sharpe_ratio(returns)
        # Should handle zero std gracefully
        assert isinstance(sharpe, float)


class TestSortinoRatio:
    """Test cases for Sortino ratio calculation."""
    
    def test_sortino_ratio_basic(self, sample_returns):
        """Test basic Sortino ratio calculation."""
        sortino = calculate_sortino_ratio(sample_returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
        assert not np.isinf(sortino)
    
    def test_sortino_ratio_with_risk_free_rate(self, sample_returns):
        """Test Sortino ratio with risk-free rate."""
        sortino = calculate_sortino_ratio(sample_returns, risk_free_rate=0.02)
        assert isinstance(sortino, float)
    
    def test_sortino_ratio_positive_returns(self, positive_returns):
        """Test Sortino ratio with positive returns."""
        sortino = calculate_sortino_ratio(positive_returns)
        assert sortino > 0


class TestCalmarRatio:
    """Test cases for Calmar ratio calculation."""
    
    def test_calmar_ratio_basic(self, sample_returns):
        """Test basic Calmar ratio calculation."""
        max_dd = calculate_max_drawdown(sample_returns)
        calmar = calculate_calmar_ratio(sample_returns, max_dd)
        assert isinstance(calmar, float)
    
    def test_calmar_ratio_zero_drawdown(self, positive_returns):
        """Test Calmar ratio with zero drawdown."""
        max_dd = 0.0
        calmar = calculate_calmar_ratio(positive_returns, max_dd)
        # Should handle zero drawdown gracefully
        assert isinstance(calmar, float)


class TestMaxDrawdown:
    """Test cases for maximum drawdown calculation."""
    
    def test_max_drawdown_basic(self, sample_returns):
        """Test basic max drawdown calculation."""
        max_dd = calculate_max_drawdown(sample_returns)
        assert isinstance(max_dd, float)
        assert max_dd >= 0  # Drawdown should be non-negative
    
    def test_max_drawdown_positive_returns(self, positive_returns):
        """Test max drawdown with positive returns."""
        max_dd = calculate_max_drawdown(positive_returns)
        assert max_dd >= 0
    
    def test_max_drawdown_negative_returns(self, negative_returns):
        """Test max drawdown with negative returns."""
        max_dd = calculate_max_drawdown(negative_returns)
        assert max_dd >= 0

