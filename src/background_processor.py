# src/background_processor.py
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

class BackgroundModelTrainer:
    """Class to handle background model training"""
    
    def __init__(self):
        """Initialize the background trainer"""
        self.is_running = False
        self.progress = 0.0
        self.status_message = "Not started"
        self.results = None
        self.thread = None
        self.stop_requested = False
    
    def start_training(self, df: pd.DataFrame, model_type: str) -> bool:
        """
        Start model training in the background
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to process
        model_type : str
            Type of model to train
            
        Returns:
        --------
        bool
            Whether training was successfully started
        """
        # If already running, return failure
        if self.is_running:
            return False
        
        # Reset state
        self.is_running = True
        self.progress = 0.0
        self.status_message = "Initializing training..."
        self.results = None
        self.stop_requested = False
        
        # Create and start a background thread
        self.thread = threading.Thread(
            target=self._run_training,
            args=(df.copy(), model_type),
            daemon=True  # Set as daemon thread so it will exit when the main thread exits
        )
        self.thread.start()
        
        return True
    
    def stop_training(self) -> bool:
        """
        Request to stop the training process
        
        Returns:
        --------
        bool
            Whether stop request was successful
        """
        if not self.is_running:
            return False
        
        self.stop_requested = True
        self.status_message = "Stopping training..."
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current training status
        
        Returns:
        --------
        dict
            Dictionary containing current status information
        """
        return {
            "is_running": self.is_running,
            "progress": self.progress,
            "status_message": self.status_message,
            "has_results": self.results is not None
        }
    
    def _run_training(self, df: pd.DataFrame, model_type: str) -> None:
        """
        Actual training process running in the background
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to process
        model_type : str
            Type of model to train
        """
        try:
            # Stage 1: Data preprocessing
            self.status_message = "Preprocessing data..."
            for i in range(10):
                if self.stop_requested:
                    raise InterruptedError("Training aborted by user")
                    
                time.sleep(0.3)  # Simulate processing time
                self.progress = 0.05 + i * 0.01
            
            # Stage 2: Feature engineering
            self.status_message = "Performing feature engineering..."
            for i in range(20):
                if self.stop_requested:
                    raise InterruptedError("Training aborted by user")
                    
                time.sleep(0.2)  # Simulate processing time
                self.progress = 0.15 + i * 0.01
            
            # Stage 3: Parameter optimization
            self.status_message = "Optimizing hyperparameters..."
            for i in range(30):
                if self.stop_requested:
                    raise InterruptedError("Training aborted by user")
                    
                time.sleep(0.3)  # Simulate processing time
                self.progress = 0.35 + i * 0.01
            
            # Stage 4: Model training
            self.status_message = f"Training {model_type} model..."
            for i in range(20):
                if self.stop_requested:
                    raise InterruptedError("Training aborted by user")
                    
                time.sleep(0.3)  # Simulate processing time
                self.progress = 0.65 + i * 0.01
            
            # Stage 5: Model evaluation
            self.status_message = "Evaluating model performance..."
            for i in range(10):
                if self.stop_requested:
                    raise InterruptedError("Training aborted by user")
                    
                time.sleep(0.2)  # Simulate processing time
                self.progress = 0.85 + i * 0.01
            
            # Complete and generate results
            self.status_message = "Generating final report..."
            time.sleep(1)
            self.progress = 0.95
            
            # Simulate evaluation results
            self.results = {
                "model_type": model_type,
                "accuracy": np.random.uniform(0.7, 0.95),
                "precision": np.random.uniform(0.6, 0.9),
                "recall": np.random.uniform(0.6, 0.9),
                "f1_score": np.random.uniform(0.6, 0.9),
                "training_time": f"{np.random.randint(10, 120)} seconds",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Final stage
            self.status_message = "Training completed!"
            self.progress = 1.0
            
        except InterruptedError as e:
            # User abort
            self.status_message = f"Training aborted by user: {str(e)}"
            self.results = None
            
        except Exception as e:
            # Error occurred
            self.status_message = f"Error during training: {str(e)}"
            self.results = None
            
        finally:
            # Ensure training state is correct
            self.is_running = False
            self.stop_requested = False


class ModelCache:
    """Model cache management class"""
    
    def __init__(self, max_cache_size: int = 10, ttl_hours: int = 24):
        """
        Initialize model cache
        
        Parameters:
        -----------
        max_cache_size : int
            Maximum number of models to cache
        ttl_hours : int
            Cache time-to-live in hours
        """
        self.cache = {}  # Cache dictionary
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_hours * 3600
        self.access_times = {}  # Record last access time for each model
    
    def get_model(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get model from cache
        
        Parameters:
        -----------
        key : str
            Cache key for the model
            
        Returns:
        --------
        Optional[Dict[str, Any]]
            Cached model data, or None if not found
        """
        # Check if model exists in cache
        if key not in self.cache:
            return None
        
        # Check if cache has expired
        current_time = time.time()
        if current_time - self.access_times[key]["created"] > self.ttl_seconds:
            # Cache expired, delete and return None
            del self.cache[key]
            del self.access_times[key]
            return None
        
        # Update access time
        self.access_times[key]["last_accessed"] = current_time
        
        # Return cached model
        return self.cache[key]
    
    def cache_model(self, key: str, model_data: Dict[str, Any]) -> None:
        """
        Add model to cache
        
        Parameters:
        -----------
        key : str
            Cache key for the model
        model_data : Dict[str, Any]
            Model data to cache
        """
        current_time = time.time()
        
        # Check if maximum cache size is reached
        if len(self.cache) >= self.max_cache_size and key not in self.cache:
            # Find and delete oldest entry
            oldest_key = min(self.access_times, key=lambda k: self.access_times[k]["last_accessed"])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        # Add new model to cache
        self.cache[key] = model_data
        self.access_times[key] = {
            "created": current_time,
            "last_accessed": current_time
        }
    
    def clear_cache(self) -> int:
        """
        Clear all cached models
        
        Returns:
        --------
        int
            Number of cache entries cleared
        """
        count = len(self.cache)
        self.cache.clear()
        self.access_times.clear()
        return count
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries
        
        Returns:
        --------
        int
            Number of cache entries cleared
        """
        current_time = time.time()
        expired_keys = [
            key for key in self.cache 
            if current_time - self.access_times[key]["created"] > self.ttl_seconds
        ]
        
        # Delete expired cache entries
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
        
        return len(expired_keys)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing cache information
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_cache_size,
            "ttl_hours": self.ttl_seconds / 3600,
            "models": [
                {
                    "key": key,
                    "age_hours": (time.time() - self.access_times[key]["created"]) / 3600,
                    "last_accessed": datetime.fromtimestamp(
                        self.access_times[key]["last_accessed"]
                    ).strftime("%Y-%m-%d %H:%M:%S")
                }
                for key in self.cache
            ]
        }


# Advanced usage example: Model cache decorator
def cached_model_training(ttl_hours: int = 24):
    """
    Model training cache decorator
    
    Parameters:
    -----------
    ttl_hours : int
        Cache time-to-live in hours
        
    Returns:
    --------
    Function decorator
    """
    # Create model cache
    model_cache = ModelCache(ttl_hours=ttl_hours)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key
            key_parts = [str(arg) for arg in args]
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = func.__name__ + "_" + "_".join(key_parts)
            
            # Try to get from cache
            cached_result = model_cache.get_model(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Not in cache, execute original function
            result = func(*args, **kwargs)
            
            # Cache result
            model_cache.cache_model(cache_key, result)
            
            return result
        return wrapper
    return decorator