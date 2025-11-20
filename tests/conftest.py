"""
Pytest configuration and fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
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
    data['high'] = data[['open', 'close', 'high']].max(axis=1) + 0.1
    data['low'] = data[['open', 'close', 'low']].min(axis=1) - 0.1
    
    return data


@pytest.fixture
def sample_event_data(sample_ohlcv_data):
    """Create sample data with events."""
    data = sample_ohlcv_data.copy()
    
    # Add some technical indicators
    data['Lower_Band_Slope'] = np.random.randn(100) * 0.01
    data['Slope_Change'] = np.random.randn(100) * 0.01
    data['Rebound_Above_EMA'] = np.random.choice([True, False], 100)
    data['Break_Below_EMA'] = np.random.choice([True, False], 100)
    data['Average_Volatility_long'] = np.random.uniform(100, 1000, 100)
    
    # Add some events
    data['Event'] = 0
    data.loc[10:15, 'Event'] = 1  # Some long events
    data.loc[30:35, 'Event'] = -1  # Some short events
    data['Label'] = data['Event']
    
    return data

