"""
Alpha factor selection and ranking module.

Provides tools for selecting the most predictive alpha factors using
statistical significance testing, information coefficient, and other metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, f_regression


def calculate_information_coefficient(alpha: pd.Series, forward_returns: pd.Series, 
                                     method: str = 'pearson') -> Tuple[float, float]:
    """
    Calculate Information Coefficient (IC) between alpha and forward returns.
    
    Parameters:
    -----------
    alpha : pd.Series
        Alpha factor values
    forward_returns : pd.Series
        Forward returns (target variable)
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
    --------
    tuple
        (IC value, p-value)
    """
    # Align series and remove NaN
    aligned = pd.concat([alpha, forward_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0, 1.0
    
    alpha_vals = aligned.iloc[:, 0]
    returns_vals = aligned.iloc[:, 1]
    
    if method == 'pearson':
        corr, p_value = stats.pearsonr(alpha_vals, returns_vals)
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(alpha_vals, returns_vals)
    elif method == 'kendall':
        corr, p_value = stats.kendalltau(alpha_vals, returns_vals)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr, p_value


def calculate_ic_series(alphas: pd.DataFrame, forward_returns: pd.Series, 
                       window: int = 20, method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate rolling Information Coefficient series.
    
    Parameters:
    -----------
    alphas : pd.DataFrame
        DataFrame with alpha factors as columns
    forward_returns : pd.Series
        Forward returns series
    window : int, default=20
        Rolling window size
    method : str, default='pearson'
        Correlation method
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with IC values for each alpha over time
    """
    ic_series = pd.DataFrame(index=alphas.index, columns=alphas.columns)
    
    for col in alphas.columns:
        for i in range(window, len(alphas)):
            window_alpha = alphas[col].iloc[i-window:i]
            window_returns = forward_returns.iloc[i-window:i]
            ic, _ = calculate_information_coefficient(window_alpha, window_returns, method)
            ic_series.loc[alphas.index[i], col] = ic
    
    return ic_series


def calculate_ic_statistics(ic_series: pd.Series) -> Dict[str, float]:
    """
    Calculate IC statistics (mean, std, IR, etc.).
    
    Parameters:
    -----------
    ic_series : pd.Series
        Series of IC values
        
    Returns:
    --------
    dict
        Dictionary with IC statistics
    """
    ic_clean = ic_series.dropna()
    
    if len(ic_clean) == 0:
        return {
            'mean_ic': 0.0,
            'std_ic': 0.0,
            'ir': 0.0,  # Information Ratio = mean(IC) / std(IC)
            'ic_skew': 0.0,
            'ic_kurtosis': 0.0,
            'positive_ic_pct': 0.0
        }
    
    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    ir = mean_ic / std_ic if std_ic > 0 else 0.0
    
    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'ir': ir,
        'ic_skew': ic_clean.skew(),
        'ic_kurtosis': ic_clean.kurtosis(),
        'positive_ic_pct': (ic_clean > 0).mean() * 100
    }


def test_alpha_significance(alpha: pd.Series, forward_returns: pd.Series, 
                           alpha_level: float = 0.05) -> Dict[str, Any]:
    """
    Test statistical significance of alpha factor.
    
    Parameters:
    -----------
    alpha : pd.Series
        Alpha factor values
    forward_returns : pd.Series
        Forward returns
    alpha_level : float, default=0.05
        Significance level
        
    Returns:
    --------
    dict
        Dictionary with test results
    """
    # Align and clean data
    aligned = pd.concat([alpha, forward_returns], axis=1).dropna()
    if len(aligned) < 2:
        return {
            'is_significant': False,
            'p_value': 1.0,
            't_statistic': 0.0,
            'correlation': 0.0
        }
    
    alpha_vals = aligned.iloc[:, 0]
    returns_vals = aligned.iloc[:, 1]
    
    # Correlation test
    corr, p_value = stats.pearsonr(alpha_vals, returns_vals)
    
    # T-test for mean difference between high and low alpha groups
    median_alpha = alpha_vals.median()
    high_group = returns_vals[alpha_vals > median_alpha]
    low_group = returns_vals[alpha_vals <= median_alpha]
    
    if len(high_group) > 0 and len(low_group) > 0:
        t_stat, t_p_value = stats.ttest_ind(high_group, low_group)
    else:
        t_stat, t_p_value = 0.0, 1.0
    
    return {
        'is_significant': p_value < alpha_level,
        'p_value': p_value,
        't_statistic': t_stat,
        't_p_value': t_p_value,
        'correlation': corr,
        'mean_high_alpha_return': high_group.mean() if len(high_group) > 0 else 0.0,
        'mean_low_alpha_return': low_group.mean() if len(low_group) > 0 else 0.0
    }


def analyze_alpha_decay(ic_series: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    """
    Analyze alpha decay over time (how IC changes with lag).
    
    Parameters:
    -----------
    ic_series : pd.Series
        Series of IC values
    max_lag : int, default=10
        Maximum lag to analyze
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with decay analysis
    """
    decay_results = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            ic_lag = ic_series
        else:
            ic_lag = ic_series.shift(-lag)
        
        ic_clean = ic_lag.dropna()
        if len(ic_clean) > 0:
            decay_results.append({
                'lag': lag,
                'mean_ic': ic_clean.mean(),
                'std_ic': ic_clean.std(),
                'ir': ic_clean.mean() / ic_clean.std() if ic_clean.std() > 0 else 0.0
            })
    
    return pd.DataFrame(decay_results)


def rank_alphas(alphas: pd.DataFrame, forward_returns: pd.Series,
               method: str = 'ic_ir') -> pd.DataFrame:
    """
    Rank alpha factors by various metrics.
    
    Parameters:
    -----------
    alphas : pd.DataFrame
        DataFrame with alpha factors as columns
    forward_returns : pd.Series
        Forward returns series
    method : str, default='ic_ir'
        Ranking method: 'ic_ir' (IC Information Ratio), 'mean_ic', 'mutual_info', 'f_score'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with ranked alphas
    """
    rankings = []
    
    for col in alphas.columns:
        alpha = alphas[col]
        
        # Calculate IC
        ic, p_value = calculate_information_coefficient(alpha, forward_returns)
        
        # Calculate IC series and statistics
        ic_series = calculate_ic_series(pd.DataFrame({col: alpha}), forward_returns)
        ic_stats = calculate_ic_statistics(ic_series[col])
        
        # Significance test
        sig_test = test_alpha_significance(alpha, forward_returns)
        
        # Mutual information
        aligned = pd.concat([alpha, forward_returns], axis=1).dropna()
        if len(aligned) > 0:
            mi = mutual_info_regression(
                aligned.iloc[:, 0].values.reshape(-1, 1),
                aligned.iloc[:, 1].values,
                random_state=42
            )[0]
        else:
            mi = 0.0
        
        # F-score
        aligned_clean = aligned.dropna()
        if len(aligned_clean) > 1:
            f_score, _ = f_regression(
                aligned_clean.iloc[:, 0].values.reshape(-1, 1),
                aligned_clean.iloc[:, 1].values
            )
            f_score = f_score[0] if len(f_score) > 0 else 0.0
        else:
            f_score = 0.0
        
        rankings.append({
            'alpha_name': col,
            'mean_ic': ic_stats['mean_ic'],
            'ic_ir': ic_stats['ir'],
            'ic_std': ic_stats['std_ic'],
            'correlation': sig_test['correlation'],
            'p_value': sig_test['p_value'],
            'is_significant': sig_test['is_significant'],
            'mutual_info': mi,
            'f_score': f_score,
            'positive_ic_pct': ic_stats['positive_ic_pct']
        })
    
    ranking_df = pd.DataFrame(rankings)
    
    # Sort by selected method
    if method == 'ic_ir':
        ranking_df = ranking_df.sort_values('ic_ir', ascending=False)
    elif method == 'mean_ic':
        ranking_df = ranking_df.sort_values('mean_ic', ascending=False)
    elif method == 'mutual_info':
        ranking_df = ranking_df.sort_values('mutual_info', ascending=False)
    elif method == 'f_score':
        ranking_df = ranking_df.sort_values('f_score', ascending=False)
    else:
        raise ValueError(f"Unknown ranking method: {method}")
    
    return ranking_df


def select_top_alphas(ranking_df: pd.DataFrame, top_n: int = 20,
                     min_ic: float = 0.0, min_ir: float = 0.0,
                     require_significant: bool = True) -> List[str]:
    """
    Select top alpha factors based on criteria.
    
    Parameters:
    -----------
    ranking_df : pd.DataFrame
        DataFrame with alpha rankings
    top_n : int, default=20
        Number of top alphas to select
    min_ic : float, default=0.0
        Minimum mean IC threshold
    min_ir : float, default=0.0
        Minimum IC Information Ratio threshold
    require_significant : bool, default=True
        Whether to require statistical significance
        
    Returns:
    --------
    list
        List of selected alpha factor names
    """
    filtered = ranking_df.copy()
    
    # Apply filters
    if require_significant:
        filtered = filtered[filtered['is_significant'] == True]
    
    filtered = filtered[filtered['mean_ic'] >= min_ic]
    filtered = filtered[filtered['ic_ir'] >= min_ir]
    
    # Select top N
    selected = filtered.head(top_n)['alpha_name'].tolist()
    
    return selected

