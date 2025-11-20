"""
Configuration module for the trading event detection system.

This module provides centralized configuration management including paths,
parameters, and environment-specific settings.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """
    Centralized configuration class for the trading system.
    
    Provides access to all paths, parameters, and settings used throughout
    the application. All paths are dynamically constructed relative to the
    project root directory.
    """
    
    # Base directory - project root
    BASE_DIR = Path(__file__).parent.parent.absolute()
    
    # Data directories
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model directories
    MODELS_DIR = BASE_DIR / "models"
    
    # Results directories
    RESULTS_DIR = BASE_DIR / "results"
    BACKTEST_RESULTS_DIR = RESULTS_DIR / "backtests"
    VISUALIZATION_DIR = RESULTS_DIR / "visualization"
    SUMMARIES_DIR = RESULTS_DIR / "summaries"
    MA_STRATEGY_DIR = RESULTS_DIR / "ma_strategy"
    
    # Figure directories
    FIG_DIR = BASE_DIR / "Fig"
    
    # Examples directory
    EXAMPLES_DIR = BASE_DIR / "examples"
    
    # Package directory
    PACKAGE_DIR = BASE_DIR / "package"
    
    # Event detection parameters
    EVENT_DETECTION_PARAMS = {
        'profit_loss_window': 3,
        'atr_window': 14,
        'long_profit_threshold': 10.0,
        'short_loss_threshold': -10.0,
        'volume_multiplier': 2.0,
        'use_atr_filter': True
    }
    
    # Feature calculation parameters
    FEATURE_CALCULATION_PARAMS = {
        'slope_window': 3,
        'ema_window': 9,
        'avg_vol_window': 9,
        'long_ema_window': 13
    }
    
    # Feature engineering parameters
    FEATURE_ENGINEERING_PARAMS = {
        'variance_threshold': 0.005,
        'lasso_eps': 1e-4,
        'corr_threshold': 0.9,
        'remove_column_name': ['date', 'Profit_Loss_Points', 'Event', 'Label']
    }
    
    # Trading strategy parameters
    TRADING_STRATEGY_PARAMS = {
        'long_threshold': 0.0026,
        'short_threshold': 0.0026,
        'holding_period': 3,
        'exclude_times': {
            "08:45", "08:46", "08:47", "08:48", "08:49",  # First 5 minutes after morning open
            "13:41", "13:42", "13:43", "13:44", "13:45",  # First 5 minutes after afternoon open
            "15:00", "15:01", "15:02", "15:03", "15:04",  # Last 5 minutes before day session close
            "03:55", "03:56", "03:57", "03:58", "03:59"   # Last 5 minutes before night session close
        }
    }
    
    # Backtesting parameters
    BACKTESTING_PARAMS = {
        'profit_loss_window': 3,
        'max_profit_loss': 50
    }
    
    # Model default parameters
    MODEL_DEFAULTS = {
        'randomforest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        },
        'gradientboosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        'lightgbm': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    }
    
    # Optimization search spaces
    OPTIMIZATION_SPACE = {
        'randomforest': {
            'n_estimators': (50, 300),
            'max_depth': (3, 15),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10)
        },
        'gradientboosting': {
            'n_estimators': (50, 300),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'subsample': (0.5, 1.0)
        },
        'xgboost': {
            'n_estimators': (50, 300),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0)
        },
        'lightgbm': {
            'n_estimators': (50, 300),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'num_leaves': (20, 150),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0)
        }
    }
    
    @classmethod
    def ensure_directories(cls) -> None:
        """
        Ensure all required directories exist.
        
        Creates all necessary directories if they don't already exist.
        """
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.BACKTEST_RESULTS_DIR,
            cls.VISUALIZATION_DIR,
            cls.SUMMARIES_DIR,
            cls.MA_STRATEGY_DIR,
            cls.FIG_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_paths(cls) -> Dict[str, Path]:
        """
        Get all paths as a dictionary.
        
        Returns:
        --------
        dict
            Dictionary containing all path configurations
        """
        return {
            'base': cls.BASE_DIR,
            'data': cls.DATA_DIR,
            'raw_data': cls.RAW_DATA_DIR,
            'processed_data': cls.PROCESSED_DATA_DIR,
            'models': cls.MODELS_DIR,
            'results': cls.RESULTS_DIR,
            'backtest_results': cls.BACKTEST_RESULTS_DIR,
            'visualization': cls.VISUALIZATION_DIR,
            'summaries': cls.SUMMARIES_DIR,
            'ma_strategy': cls.MA_STRATEGY_DIR,
            'fig': cls.FIG_DIR,
            'examples': cls.EXAMPLES_DIR,
            'package': cls.PACKAGE_DIR
        }
    
    @classmethod
    def get_default_data_paths(cls) -> Dict[str, Path]:
        """
        Get default data file paths.
        
        Returns:
        --------
        dict
            Dictionary containing default training and validation data paths
        """
        return {
            'training': cls.RAW_DATA_DIR / "TX00_training.xlsx",
            'validation': cls.RAW_DATA_DIR / "TX00_validation.xlsx"
        }
    
    @classmethod
    def get_default_model_paths(cls, model_name: str = 'lightgbm') -> Dict[str, Path]:
        """
        Get default model file paths.
        
        Parameters:
        -----------
        model_name : str, default='lightgbm'
            Name of the model
            
        Returns:
        --------
        dict
            Dictionary containing model, feature, and scaler paths
        """
        return {
            'model': cls.MODELS_DIR / f"{model_name}.joblib",
            'features': cls.MODELS_DIR / "features.xlsx",
            'scaler': cls.MODELS_DIR / "scaler.joblib",
            'params': cls.MODELS_DIR / "params.json"
        }


# Create a singleton instance for easy access
config = Config()

# Ensure directories exist on import
config.ensure_directories()

