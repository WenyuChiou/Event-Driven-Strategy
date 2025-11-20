"""
Unit tests for event detection module.
"""

import pytest
import pandas as pd
import numpy as np
from src.event_detection import detect_trading_events, analyze_trading_events


def test_detect_trading_events_basic(sample_ohlcv_data):
    """Test basic event detection functionality."""
    # Add required technical indicators
    data = sample_ohlcv_data.copy()
    data['Lower_Band_Slope'] = np.random.randn(100) * 0.01
    data['Slope_Change'] = np.random.randn(100) * 0.01
    data['Rebound_Above_EMA'] = np.random.choice([True, False], 100)
    data['Break_Below_EMA'] = np.random.choice([True, False], 100)
    data['Average_Volatility_long'] = np.random.uniform(100, 1000, 100)
    
    result = detect_trading_events(data)
    
    assert 'Event' in result.columns
    assert 'Label' in result.columns
    assert 'Event_Type' in result.columns
    assert len(result) == len(data)


def test_analyze_trading_events(sample_event_data):
    """Test event analysis functionality."""
    analysis = analyze_trading_events(sample_event_data)
    
    assert 'total_events' in analysis
    assert 'long_events' in analysis
    assert 'short_events' in analysis
    assert 'win_rate' in analysis
    assert 'profit_factor' in analysis


def test_analyze_trading_events_no_events(sample_ohlcv_data):
    """Test analysis with no events."""
    data = sample_ohlcv_data.copy()
    data['Event'] = 0
    data['Label'] = 0
    
    analysis = analyze_trading_events(data)
    
    assert 'error' in analysis or analysis['total_events'] == 0

