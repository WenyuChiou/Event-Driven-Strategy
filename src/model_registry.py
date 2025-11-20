"""
Model registry and versioning module.

Provides model versioning, metadata tracking, and model comparison capabilities.
"""

import os
import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib


class ModelRegistry:
    """
    Model registry for tracking and managing model versions.
    
    Stores model metadata including training date, parameters, performance metrics,
    and enables easy model comparison and rollback.
    """
    
    def __init__(self, registry_dir: Optional[Path] = None):
        """
        Initialize model registry.
        
        Parameters:
        -----------
        registry_dir : Path, optional
            Directory for storing registry data. If None, uses default models directory.
        """
        if registry_dir is None:
            from src.config import Config
            registry_dir = Config.MODELS_DIR / "registry"
        
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash of model file for integrity checking."""
        with open(model_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def register_model(self, model_path: str, model_name: str, 
                      model_type: str, parameters: Dict[str, Any],
                      performance_metrics: Dict[str, Any],
                      feature_list: Optional[List[str]] = None,
                      training_data_info: Optional[Dict[str, Any]] = None,
                      notes: Optional[str] = None) -> str:
        """
        Register a new model version.
        
        Parameters:
        -----------
        model_path : str
            Path to the model file
        model_name : str
            Name of the model
        model_type : str
            Type of model (e.g., 'lightgbm', 'xgboost')
        parameters : dict
            Model hyperparameters
        performance_metrics : dict
            Model performance metrics
        feature_list : list, optional
            List of features used
        training_data_info : dict, optional
            Information about training data
        notes : str, optional
            Additional notes
        
        Returns:
        --------
        str
            Version ID of the registered model
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_name}_{timestamp}"
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_path_obj)
        
        # Create registry entry
        entry = {
            'version_id': version_id,
            'model_name': model_name,
            'model_type': model_type,
            'model_path': str(model_path_obj.absolute()),
            'model_hash': model_hash,
            'registration_date': datetime.now().isoformat(),
            'parameters': parameters,
            'performance_metrics': performance_metrics,
            'feature_list': feature_list,
            'training_data_info': training_data_info,
            'notes': notes
        }
        
        # Add to registry
        if model_name not in self.registry:
            self.registry[model_name] = []
        
        self.registry[model_name].append(entry)
        self._save_registry()
        
        return version_id
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        list
            List of model version entries
        """
        return self.registry.get(model_name, [])
    
    def get_latest_version(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict or None
            Latest model version entry
        """
        versions = self.get_model_versions(model_name)
        if not versions:
            return None
        
        # Sort by registration date (most recent first)
        versions_sorted = sorted(
            versions, 
            key=lambda x: x['registration_date'], 
            reverse=True
        )
        
        return versions_sorted[0]
    
    def get_version(self, model_name: str, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific model version.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        version_id : str
            Version ID
            
        Returns:
        --------
        dict or None
            Model version entry
        """
        versions = self.get_model_versions(model_name)
        for version in versions:
            if version['version_id'] == version_id:
                return version
        return None
    
    def compare_models(self, model_name: str, version_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple model versions.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        version_ids : list, optional
            Specific version IDs to compare. If None, compares all versions.
            
        Returns:
        --------
        pd.DataFrame
            Comparison table
        """
        versions = self.get_model_versions(model_name)
        
        if version_ids:
            versions = [v for v in versions if v['version_id'] in version_ids]
        
        if not versions:
            return pd.DataFrame()
        
        # Extract comparison data
        comparison_data = []
        for version in versions:
            row = {
                'version_id': version['version_id'],
                'registration_date': version['registration_date'],
                'model_type': version['model_type']
            }
            
            # Add parameters
            for key, value in version['parameters'].items():
                row[f'param_{key}'] = value
            
            # Add performance metrics
            for key, value in version['performance_metrics'].items():
                row[f'metric_{key}'] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def verify_model_integrity(self, model_name: str, version_id: str) -> bool:
        """
        Verify model file integrity using hash.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        version_id : str
            Version ID
            
        Returns:
        --------
        bool
            True if integrity check passes
        """
        version = self.get_version(model_name, version_id)
        if version is None:
            return False
        
        model_path = Path(version['model_path'])
        if not model_path.exists():
            return False
        
        current_hash = self._calculate_model_hash(model_path)
        return current_hash == version['model_hash']
    
    def list_all_models(self) -> List[str]:
        """
        List all registered model names.
        
        Returns:
        --------
        list
            List of model names
        """
        return list(self.registry.keys())
    
    def export_registry(self, export_path: str) -> None:
        """
        Export registry to file.
        
        Parameters:
        -----------
        export_path : str
            Path to export file
        """
        with open(export_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)


def register_trained_model(model_path: str, model_name: str, model_type: str,
                          parameters: Dict[str, Any], metrics: Dict[str, Any],
                          feature_list: Optional[List[str]] = None,
                          registry: Optional[ModelRegistry] = None) -> str:
    """
    Convenience function to register a trained model.
    
    Parameters:
    -----------
    model_path : str
        Path to model file
    model_name : str
        Model name
    model_type : str
        Model type
    parameters : dict
        Model parameters
    metrics : dict
        Performance metrics
    feature_list : list, optional
        Feature list
    registry : ModelRegistry, optional
        Registry instance. If None, creates a new one.
        
    Returns:
    --------
    str
        Version ID
    """
    if registry is None:
        registry = ModelRegistry()
    
    return registry.register_model(
        model_path=model_path,
        model_name=model_name,
        model_type=model_type,
        parameters=parameters,
        performance_metrics=metrics,
        feature_list=feature_list
    )

