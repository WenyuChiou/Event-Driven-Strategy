"""
Modular feature engineering pipeline.

Provides a flexible pipeline for feature engineering with caching and versioning.
"""

import pandas as pd
import numpy as np
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime


class FeaturePipeline:
    """
    Modular feature engineering pipeline with caching support.
    
    Allows easy addition/removal of feature engineering steps,
    caching of intermediate results, and feature versioning.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, use_cache: bool = True):
        """
        Initialize feature pipeline.
        
        Parameters:
        -----------
        cache_dir : Path, optional
            Directory for caching intermediate results
        use_cache : bool, default=True
            Whether to use caching
        """
        self.steps: List[Dict[str, Any]] = []
        self.use_cache = use_cache
        
        if cache_dir is None:
            from src.config import Config
            cache_dir = Config.BASE_DIR / "cache" / "features"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def add_step(self, name: str, function: Callable, 
                dependencies: Optional[List[str]] = None,
                cache: bool = True) -> None:
        """
        Add a feature engineering step to the pipeline.
        
        Parameters:
        -----------
        name : str
            Name of the step
        function : callable
            Function to execute for this step
        dependencies : list, optional
            Names of steps this step depends on
        cache : bool, default=True
            Whether to cache results of this step
        """
        self.steps.append({
            'name': name,
            'function': function,
            'dependencies': dependencies or [],
            'cache': cache
        })
    
    def _get_cache_key(self, step_name: str, data_hash: str) -> Path:
        """Generate cache file path."""
        return self.cache_dir / f"{step_name}_{data_hash}.pkl"
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data for cache key."""
        data_str = pd.util.hash_pandas_object(data).sum()
        return hashlib.md5(str(data_str).encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path) -> None:
        """Save data to cache."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _resolve_dependencies(self) -> List[str]:
        """Resolve step execution order based on dependencies."""
        # Simple topological sort
        executed = []
        remaining = {step['name']: step for step in self.steps}
        
        while remaining:
            # Find steps with no unmet dependencies
            ready = []
            for name, step in remaining.items():
                deps_met = all(dep in executed for dep in step['dependencies'])
                if deps_met:
                    ready.append(name)
            
            if not ready:
                # Circular dependency or missing dependency
                raise ValueError("Cannot resolve dependencies. Check for circular dependencies.")
            
            # Execute ready steps (in order added if multiple ready)
            for step in self.steps:
                if step['name'] in ready:
                    executed.append(step['name'])
                    del remaining[step['name']]
                    break
        
        return executed
    
    def fit_transform(self, data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Execute all pipeline steps on data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        verbose : bool, default=True
            Whether to print progress
            
        Returns:
        --------
        pd.DataFrame
            Transformed data
        """
        result = data.copy()
        data_hash = self._calculate_data_hash(data)
        
        # Resolve execution order
        execution_order = self._resolve_dependencies()
        
        # Execute steps in order
        for step_name in execution_order:
            step = next(s for s in self.steps if s['name'] == step_name)
            
            if verbose:
                print(f"Executing step: {step_name}")
            
            # Check cache
            cache_path = None
            cached_result = None
            
            if self.use_cache and step['cache']:
                cache_path = self._get_cache_key(step_name, data_hash)
                cached_result = self._load_from_cache(cache_path)
            
            if cached_result is not None:
                if verbose:
                    print(f"  Using cached result for {step_name}")
                result = cached_result
            else:
                # Execute step
                try:
                    result = step['function'](result)
                    
                    # Save to cache
                    if self.use_cache and step['cache'] and cache_path:
                        self._save_to_cache(result, cache_path)
                        if verbose:
                            print(f"  Cached result for {step_name}")
                except Exception as e:
                    print(f"Error in step {step_name}: {e}")
                    raise
        
        return result
    
    def transform(self, data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Note: For now, same as fit_transform. Can be extended to skip
        fitting steps if needed.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        verbose : bool, default=True
            Whether to print progress
            
        Returns:
        --------
        pd.DataFrame
            Transformed data
        """
        return self.fit_transform(data, verbose=verbose)
    
    def get_step_info(self) -> pd.DataFrame:
        """
        Get information about all pipeline steps.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with step information
        """
        info = []
        for step in self.steps:
            info.append({
                'name': step['name'],
                'dependencies': ', '.join(step['dependencies']) if step['dependencies'] else 'None',
                'cached': step['cache']
            })
        
        return pd.DataFrame(info)
    
    def clear_cache(self, step_name: Optional[str] = None) -> None:
        """
        Clear cache for a specific step or all steps.
        
        Parameters:
        -----------
        step_name : str, optional
            Name of step to clear cache for. If None, clears all caches.
        """
        if step_name:
            # Clear cache for specific step
            pattern = f"{step_name}_*.pkl"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
        else:
            # Clear all caches
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()


def create_default_pipeline() -> FeaturePipeline:
    """
    Create a default feature engineering pipeline.
    
    Returns:
    --------
    FeaturePipeline
        Configured pipeline with common steps
    """
    from src.feature_engineering import calculate_features
    from package.FE import FeatureEngineering
    
    pipeline = FeaturePipeline()
    
    # Step 1: Calculate technical features
    pipeline.add_step(
        name='technical_features',
        function=lambda df: calculate_features(df)[0],
        dependencies=[],
        cache=True
    )
    
    # Step 2: Feature selection (requires technical features)
    def feature_selection_step(df):
        fe = FeatureEngineering()
        X_final, _, _ = fe.fit(df, target_column='Label')
        return X_final
    
    pipeline.add_step(
        name='feature_selection',
        function=feature_selection_step,
        dependencies=['technical_features'],
        cache=True
    )
    
    return pipeline

