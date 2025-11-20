"""
Unit tests for evaluation metrics module.
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_trade_statistics
)


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    sharpe = calculate_sharpe_ratio(returns)
    
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)


def test_calculate_sortino_ratio():
    """Test Sortino ratio calculation."""
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    sortino = calculate_sortino_ratio(returns)
    
    assert isinstance(sortino, (float, np.floating))
    assert not np.isnan(sortino) and not np.isinf(sortino)


def test_calculate_calmar_ratio():
    """Test Calmar ratio calculation."""
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    calmar = calculate_calmar_ratio(returns)
    
    assert isinstance(calmar, (float, np.floating))


def test_calculate_max_drawdown():
    """Test maximum drawdown calculation."""
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    max_dd, duration, recovery = calculate_max_drawdown(returns)
    
    assert isinstance(max_dd, (float, np.floating))
    assert isinstance(duration, (float, np.floating))
    assert isinstance(recovery, (float, np.floating))
    assert max_dd >= 0


def test_calculate_trade_statistics():
    """Test trade statistics calculation."""
    trades = pd.DataFrame({
        'profit_loss': [10, -5, 15, -8, 20, -10],
        'date': pd.date_range('2023-01-01', periods=6, freq='1H')
    })
    
    stats = calculate_trade_statistics(trades)
    
    assert 'total_trades' in stats
    assert 'win_rate' in stats
    assert stats['total_trades'] == 6
    assert 0 <= stats['win_rate'] <= 1

