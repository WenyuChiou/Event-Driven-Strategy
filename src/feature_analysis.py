"""
Feature importance analysis module using SHAP values and other interpretability methods.

Provides tools for understanding model decisions and feature contributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from sklearn.inspection import permutation_importance


def calculate_shap_values(model: Any, X: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate SHAP values for model interpretability.
    
    Note: Requires shap library to be installed.
    This function provides a wrapper around SHAP calculations.
    
    Parameters:
    -----------
    model : Any
        Trained model with predict_proba or predict method
    X : pd.DataFrame
        Feature data
    sample_size : int, optional
        Number of samples to use for SHAP calculation (for large datasets)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with SHAP values for each feature
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP library is required. Install it with: pip install shap")
    
    # Sample data if needed
    if sample_size and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    # Create SHAP explainer based on model type
    model_type = type(model).__name__.lower()
    
    if 'tree' in model_type or 'forest' in model_type or 'boost' in model_type or 'lgbm' in model_type or 'xgb' in model_type:
        explainer = shap.TreeExplainer(model)
    else:
        # Use KernelExplainer for non-tree models (slower but more general)
        explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:100])
    
    shap_values = explainer.shap_values(X_sample)
    
    # Handle multi-class output
    if isinstance(shap_values, list):
        # For multi-class, use the class with highest average importance
        shap_values = np.array(shap_values)
        shap_values = np.mean(np.abs(shap_values), axis=0)
    
    return pd.DataFrame(shap_values, columns=X_sample.columns, index=X_sample.index)


def calculate_permutation_importance(model: Any, X: pd.DataFrame, y: pd.Series, 
                                    n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Calculate permutation importance for features.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : pd.DataFrame
        Feature data
    y : pd.Series
        Target data
    n_repeats : int, default=10
        Number of times to permute each feature
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with importance scores and standard deviations
    """
    # Use sklearn's permutation_importance
    result = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats, 
        random_state=random_state,
        scoring='neg_mean_squared_error' if hasattr(model, 'predict') else None
    )
    
    importance_df = pd.DataFrame({
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }, index=X.columns)
    
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    return importance_df


def analyze_feature_interactions(X: pd.DataFrame, y: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """
    Analyze feature interactions using correlation and mutual information.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature data
    y : pd.Series
        Target data
    top_n : int, default=10
        Number of top interactions to return
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature interaction scores
    """
    from sklearn.feature_selection import mutual_info_regression
    
    # Calculate mutual information between features and target
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    # Calculate feature-feature correlations
    corr_matrix = X.corr().abs()
    
    # Find highly correlated feature pairs
    interactions = []
    for i, col1 in enumerate(X.columns):
        for col2 in X.columns[i+1:]:
            corr = corr_matrix.loc[col1, col2]
            if corr > 0.7:  # Threshold for high correlation
                interactions.append({
                    'feature1': col1,
                    'feature2': col2,
                    'correlation': corr
                })
    
    interactions_df = pd.DataFrame(interactions).sort_values('correlation', ascending=False)
    
    return {
        'mutual_information': mi_df.head(top_n),
        'feature_interactions': interactions_df.head(top_n)
    }


def plot_feature_importance_shap(shap_values: pd.DataFrame, top_n: int = 20, 
                                 figsize: Tuple[int, int] = (10, 8), 
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance using SHAP values.
    
    Parameters:
    -----------
    shap_values : pd.DataFrame
        DataFrame with SHAP values
    top_n : int, default=20
        Number of top features to display
    figsize : tuple, default=(10, 8)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Calculate mean absolute SHAP values
    importance = shap_values.abs().mean().sort_values(ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    importance.plot(kind='barh', ax=ax)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance_over_time(shap_values: pd.DataFrame, dates: pd.Series, 
                                     feature_name: str, window: int = 50,
                                     figsize: Tuple[int, int] = (12, 6),
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance over time using rolling window.
    
    Parameters:
    -----------
    shap_values : pd.DataFrame
        DataFrame with SHAP values
    dates : pd.Series
        Date series corresponding to SHAP values
    feature_name : str
        Name of feature to plot
    window : int, default=50
        Rolling window size
    figsize : tuple, default=(12, 6)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if feature_name not in shap_values.columns:
        raise ValueError(f"Feature '{feature_name}' not found in SHAP values")
    
    feature_shap = shap_values[feature_name].abs()
    rolling_importance = feature_shap.rolling(window=window).mean()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, rolling_importance, linewidth=2)
    ax.fill_between(dates, 0, rolling_importance, alpha=0.3)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance Over Time: {feature_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_feature_importance_summary(model: Any, X: pd.DataFrame, y: pd.Series,
                                     use_shap: bool = True, top_n: int = 20) -> Dict[str, Any]:
    """
    Create comprehensive feature importance summary.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : pd.DataFrame
        Feature data
    y : pd.Series
        Target data
    use_shap : bool, default=True
        Whether to use SHAP values (slower but more accurate)
    top_n : int, default=20
        Number of top features to analyze
        
    Returns:
    --------
    dict
        Dictionary containing various importance metrics
    """
    summary = {}
    
    # Built-in feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        summary['built_in_importance'] = feature_importance.head(top_n)
    
    # Permutation importance
    perm_importance = calculate_permutation_importance(model, X, y)
    summary['permutation_importance'] = perm_importance.head(top_n)
    
    # SHAP values (if requested and available)
    if use_shap:
        try:
            shap_values = calculate_shap_values(model, X, sample_size=1000)
            shap_importance = shap_values.abs().mean().sort_values(ascending=False)
            summary['shap_importance'] = pd.DataFrame({
                'feature': shap_importance.index,
                'importance': shap_importance.values
            }).head(top_n)
        except Exception as e:
            summary['shap_error'] = str(e)
    
    # Feature interactions
    interactions = analyze_feature_interactions(X, y, top_n=top_n)
    summary['interactions'] = interactions
    
    return summary

