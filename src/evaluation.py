"""
Enhanced evaluation metrics module for trading strategies.

Provides comprehensive financial metrics including Sharpe ratio, Sortino ratio,
Calmar ratio, and other risk-adjusted performance measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio for a series of returns.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    risk_free_rate : float, default=0.0
        Risk-free rate (annualized)
    periods_per_year : int, default=252
        Number of trading periods per year
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (downside deviation only).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    risk_free_rate : float, default=0.0
        Risk-free rate (annualized)
    periods_per_year : int, default=252
        Number of trading periods per year
        
    Returns:
    --------
    float
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annual return / maximum drawdown).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    periods_per_year : int, default=252
        Number of trading periods per year
        
    Returns:
    --------
    float
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (peak - cumulative_returns) / peak
    max_drawdown = drawdown.max()
    
    if max_drawdown == 0:
        return float('inf') if returns.mean() > 0 else 0.0
    
    annual_return = returns.mean() * periods_per_year
    return annual_return / abs(max_drawdown)


def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Information ratio (active return / tracking error).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns
    periods_per_year : int, default=252
        Number of trading periods per year
        
    Returns:
    --------
    float
        Information ratio
    """
    if len(returns) != len(benchmark_returns) or len(returns) == 0:
        return 0.0
    
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()
    
    if tracking_error == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * active_returns.mean() / tracking_error


def calculate_treynor_ratio(returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.0, 
                            periods_per_year: int = 252) -> float:
    """
    Calculate Treynor ratio (excess return / beta).
    
    Parameters:
    -----------
    returns : pd.Series
        Series of strategy returns
    market_returns : pd.Series
        Series of market returns
    risk_free_rate : float, default=0.0
        Risk-free rate (annualized)
    periods_per_year : int, default=252
        Number of trading periods per year
        
    Returns:
    --------
    float
        Treynor ratio
    """
    if len(returns) != len(market_returns) or len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    excess_market = market_returns - (risk_free_rate / periods_per_year)
    
    # Calculate beta
    covariance = np.cov(returns, market_returns)[0, 1]
    market_variance = market_returns.var()
    
    if market_variance == 0:
        return 0.0
    
    beta = covariance / market_variance
    
    if beta == 0:
        return 0.0
    
    annual_excess_return = excess_returns.mean() * periods_per_year
    return annual_excess_return / beta


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, float, float]:
    """
    Calculate maximum drawdown and related metrics.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
        
    Returns:
    --------
    tuple
        (max_drawdown, max_drawdown_duration, recovery_time)
    """
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (peak - cumulative_returns) / peak
    
    max_drawdown = drawdown.max()
    max_dd_idx = drawdown.idxmax()
    
    # Calculate duration of max drawdown
    if max_dd_idx is not None and not pd.isna(max_dd_idx):
        # Find when drawdown started (when peak was last reached)
        peak_before_dd = peak.iloc[:max_dd_idx+1].idxmax()
        duration = max_dd_idx - peak_before_dd if peak_before_dd is not None else 0
        
        # Calculate recovery time (time to reach previous peak)
        recovery_idx = None
        if max_dd_idx < len(cumulative_returns) - 1:
            peak_value = peak.iloc[max_dd_idx]
            recovery_mask = cumulative_returns.iloc[max_dd_idx+1:] >= peak_value
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
                recovery_time = recovery_idx - max_dd_idx if recovery_idx is not None else 0
            else:
                recovery_time = len(cumulative_returns) - max_dd_idx - 1
        else:
            recovery_time = 0
    else:
        duration = 0.0
        recovery_time = 0.0
    
    return max_drawdown, float(duration), float(recovery_time)


def calculate_trade_statistics(trades: pd.DataFrame, profit_col: str = 'profit_loss') -> Dict[str, float]:
    """
    Calculate trade statistics including win rate, average trade duration, etc.
    
    Parameters:
    -----------
    trades : pd.DataFrame
        DataFrame containing trade records
    profit_col : str, default='profit_loss'
        Column name for profit/loss
        
    Returns:
    --------
    dict
        Dictionary containing trade statistics
    """
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_trade_duration': 0.0,
            'win_streak': 0,
            'loss_streak': 0
        }
    
    winning_trades = trades[trades[profit_col] > 0]
    losing_trades = trades[trades[profit_col] < 0]
    
    # Calculate win/loss streaks
    win_loss_sequence = (trades[profit_col] > 0).astype(int)
    streaks = []
    current_streak = 1
    for i in range(1, len(win_loss_sequence)):
        if win_loss_sequence.iloc[i] == win_loss_sequence.iloc[i-1]:
            current_streak += 1
        else:
            streaks.append((win_loss_sequence.iloc[i-1], current_streak))
            current_streak = 1
    streaks.append((win_loss_sequence.iloc[-1], current_streak))
    
    win_streaks = [s[1] for s in streaks if s[0] == 1]
    loss_streaks = [s[1] for s in streaks if s[0] == 0]
    
    # Calculate average trade duration if date column exists
    avg_duration = 0.0
    if 'date' in trades.columns and len(trades) > 1:
        trades_sorted = trades.sort_values('date')
        durations = trades_sorted['date'].diff().dt.total_seconds() / 3600  # Convert to hours
        avg_duration = durations.mean() if not durations.isna().all() else 0.0
    
    return {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) if len(trades) > 0 else 0.0,
        'average_win': winning_trades[profit_col].mean() if len(winning_trades) > 0 else 0.0,
        'average_loss': losing_trades[profit_col].mean() if len(losing_trades) > 0 else 0.0,
        'largest_win': winning_trades[profit_col].max() if len(winning_trades) > 0 else 0.0,
        'largest_loss': losing_trades[profit_col].min() if len(losing_trades) > 0 else 0.0,
        'average_trade_duration': avg_duration,
        'win_streak': max(win_streaks) if win_streaks else 0,
        'loss_streak': max(loss_streaks) if loss_streaks else 0
    }


def calculate_comprehensive_metrics(returns: pd.Series, trades: Optional[pd.DataFrame] = None,
                                    benchmark_returns: Optional[pd.Series] = None,
                                    risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for a trading strategy.
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    trades : pd.DataFrame, optional
        DataFrame containing trade records
    benchmark_returns : pd.Series, optional
        Series of benchmark returns for comparison
    risk_free_rate : float, default=0.0
        Risk-free rate (annualized)
    periods_per_year : int, default=252
        Number of trading periods per year
        
    Returns:
    --------
    dict
        Dictionary containing all performance metrics
    """
    metrics = {}
    
    # Basic return metrics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = ((1 + returns).prod() ** (periods_per_year / len(returns))) - 1
    metrics['volatility'] = returns.std() * np.sqrt(periods_per_year)
    
    # Risk-adjusted metrics
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns, periods_per_year)
    
    # Drawdown metrics
    max_dd, dd_duration, recovery_time = calculate_max_drawdown(returns)
    metrics['max_drawdown'] = max_dd
    metrics['max_drawdown_duration'] = dd_duration
    metrics['recovery_time'] = recovery_time
    
    # Benchmark comparison if provided
    if benchmark_returns is not None:
        metrics['information_ratio'] = calculate_information_ratio(returns, benchmark_returns, periods_per_year)
        metrics['treynor_ratio'] = calculate_treynor_ratio(returns, benchmark_returns, risk_free_rate, periods_per_year)
        metrics['alpha'] = returns.mean() * periods_per_year - benchmark_returns.mean() * periods_per_year
        metrics['beta'] = np.cov(returns, benchmark_returns)[0, 1] / benchmark_returns.var() if benchmark_returns.var() > 0 else 0
    
    # Trade statistics if provided
    if trades is not None:
        trade_stats = calculate_trade_statistics(trades)
        metrics.update(trade_stats)
        
        # Calculate profit factor
        if 'profit_loss' in trades.columns:
            total_profit = trades[trades['profit_loss'] > 0]['profit_loss'].sum()
            total_loss = abs(trades[trades['profit_loss'] < 0]['profit_loss'].sum())
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
    
    return metrics

