"""
Model utilities for loading and preprocessing models and data
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def load_model_and_features(model_path, feature_path, scaler_path):
    """
    Load trained model, feature list and scaler
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model file
    feature_path : str
        Path to the feature list file
    scaler_path : str
        Path to the scaler file
    
    Returns:
    --------
    tuple
        (model, feature_list, scaler)
    """
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature list file not found: {feature_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    # Load model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    # Load feature list
    try:
        if feature_path.endswith('.xlsx'):
            feature_df = pd.read_excel(feature_path)
            # Assuming the feature names are in the first column
            feature_list = feature_df.iloc[:, 0].tolist()
        elif feature_path.endswith('.csv'):
            feature_df = pd.read_csv(feature_path)
            feature_list = feature_df.iloc[:, 0].tolist()
        else:
            # Try to load as a text file
            with open(feature_path, 'r') as f:
                feature_list = [line.strip() for line in f.readlines()]
                
        print(f"Loaded {len(feature_list)} features from {feature_path}")
    except Exception as e:
        raise Exception(f"Error loading feature list: {str(e)}")
    
    # Load scaler
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    except Exception as e:
        print(f"Warning: Error loading scaler: {str(e)}. Creating a new scaler.")
        scaler = StandardScaler()
    
    return model, feature_list, scaler


def preprocess_backtest_data(data, feature_list, scaler):
    """
    Preprocess data for backtesting
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to preprocess
    feature_list : list
        List of features to use
    scaler : sklearn.preprocessing.StandardScaler
        Scaler to use for preprocessing
    
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target vector
    """
    # Make a copy of the data to avoid modifying the original
    df_copy = data.copy()
    
    # Check for missing features
    missing_features = [feat for feat in feature_list if feat not in df_copy.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features are missing from the data.")
        print(f"First few missing features: {missing_features[:5]}")
        
        # Try to handle missing features
        for feat in missing_features:
            # For now, just set missing features to 0
            df_copy[feat] = 0
    
    # Extract features
    X = df_copy[feature_list].values
    
    # Scale features if scaler is provided
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            print(f"Warning: Error scaling features: {str(e)}. Using unscaled features.")
    
    # Check if target column exists
    if 'target' in df_copy.columns:
        y = df_copy['target'].values
    else:
        # If no target column is found, create a dummy target
        print("Warning: No target column found in data. Creating a dummy target.")
        y = np.zeros(len(df_copy))
    
    return X, y