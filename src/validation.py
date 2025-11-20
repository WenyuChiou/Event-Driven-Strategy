"""
Input validation module for the trading event detection system.

Provides functions to validate file paths, parameters, and data formats.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Any


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_file_path(file_path: str, must_exist: bool = True, 
                      allowed_extensions: Optional[List[str]] = None) -> Path:
    """
    Validate file path.
    
    Parameters:
    -----------
    file_path : str
        File path to validate
    must_exist : bool, default=True
        Whether the file must exist
    allowed_extensions : list, optional
        List of allowed file extensions (e.g., ['.xlsx', '.csv'])
        
    Returns:
    --------
    Path
        Validated Path object
        
    Raises:
    -------
    ValidationError
        If validation fails
    """
    if not file_path:
        raise ValidationError("File path cannot be empty")
    
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")
    
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(
                f"File extension '{path.suffix}' not allowed. "
                f"Allowed extensions: {allowed_extensions}"
            )
    
    return path


def validate_directory_path(dir_path: str, must_exist: bool = False, 
                           create_if_missing: bool = True) -> Path:
    """
    Validate directory path.
    
    Parameters:
    -----------
    dir_path : str
        Directory path to validate
    must_exist : bool, default=False
        Whether the directory must exist
    create_if_missing : bool, default=True
        Whether to create directory if it doesn't exist
        
    Returns:
    --------
    Path
        Validated Path object
        
    Raises:
    -------
    ValidationError
        If validation fails
    """
    if not dir_path:
        raise ValidationError("Directory path cannot be empty")
    
    path = Path(dir_path)
    
    if not path.exists():
        if must_exist:
            raise ValidationError(f"Directory does not exist: {dir_path}")
        elif create_if_missing:
            path.mkdir(parents=True, exist_ok=True)
    
    if path.exists() and not path.is_dir():
        raise ValidationError(f"Path exists but is not a directory: {dir_path}")
    
    return path


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None,
                       min_rows: int = 1, check_nulls: bool = True) -> pd.DataFrame:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list, optional
        List of required column names
    min_rows : int, default=1
        Minimum number of rows required
    check_nulls : bool, default=True
        Whether to check for null values in required columns
        
    Returns:
    --------
    pd.DataFrame
        Validated DataFrame
        
    Raises:
    -------
    ValidationError
        If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValidationError(
            f"DataFrame must have at least {min_rows} rows, got {len(df)}"
        )
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValidationError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )
        
        if check_nulls:
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                raise ValidationError(
                    f"Found null values in required columns:\n{null_counts[null_counts > 0]}"
                )
    
    return df


def validate_model_type(model_type: str, allowed_types: Optional[List[str]] = None) -> str:
    """
    Validate model type.
    
    Parameters:
    -----------
    model_type : str
        Model type to validate
    allowed_types : list, optional
        List of allowed model types
        
    Returns:
    --------
    str
        Validated model type
        
    Raises:
    -------
    ValidationError
        If validation fails
    """
    if not model_type:
        raise ValidationError("Model type cannot be empty")
    
    model_type = model_type.lower()
    
    if allowed_types:
        allowed_lower = [t.lower() for t in allowed_types]
        if model_type not in allowed_lower:
            raise ValidationError(
                f"Model type '{model_type}' not allowed. "
                f"Allowed types: {allowed_types}"
            )
    
    return model_type


def validate_numeric_parameter(value: Any, param_name: str, 
                              min_value: Optional[float] = None,
                              max_value: Optional[float] = None,
                              allow_none: bool = False) -> Optional[float]:
    """
    Validate numeric parameter.
    
    Parameters:
    -----------
    value : Any
        Value to validate
    param_name : str
        Name of the parameter (for error messages)
    min_value : float, optional
        Minimum allowed value
    max_value : float, optional
        Maximum allowed value
    allow_none : bool, default=False
        Whether None is allowed
        
    Returns:
    --------
    float or None
        Validated numeric value
        
    Raises:
    -------
    ValidationError
        If validation fails
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(f"Parameter '{param_name}' cannot be None")
    
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Parameter '{param_name}' must be numeric, got {type(value).__name__}"
        )
    
    if min_value is not None and num_value < min_value:
        raise ValidationError(
            f"Parameter '{param_name}' must be >= {min_value}, got {num_value}"
        )
    
    if max_value is not None and num_value > max_value:
        raise ValidationError(
            f"Parameter '{param_name}' must be <= {max_value}, got {num_value}"
        )
    
    return num_value


def validate_probability_threshold(value: float, param_name: str = "threshold") -> float:
    """
    Validate probability threshold (must be between 0 and 1).
    
    Parameters:
    -----------
    value : float
        Threshold value to validate
    param_name : str, default="threshold"
        Name of the parameter
        
    Returns:
    --------
    float
        Validated threshold
        
    Raises:
    -------
    ValidationError
        If validation fails
    """
    return validate_numeric_parameter(
        value, param_name, min_value=0.0, max_value=1.0
    )


def validate_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLCV (Open, High, Low, Close, Volume) data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
        
    Returns:
    --------
    pd.DataFrame
        Validated DataFrame
        
    Raises:
    -------
    ValidationError
        If validation fails
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Check required columns
    df = validate_dataframe(df, required_columns=required_columns, check_nulls=False)
    
    # Check price consistency: high >= close >= low and high >= open >= low
    price_errors = []
    if (df['high'] < df['close']).any():
        price_errors.append("high < close")
    if (df['close'] < df['low']).any():
        price_errors.append("close < low")
    if (df['high'] < df['open']).any():
        price_errors.append("high < open")
    if (df['open'] < df['low']).any():
        price_errors.append("open < low")
    
    if price_errors:
        raise ValidationError(
            f"Price consistency errors found: {', '.join(price_errors)}. "
            "Please check your data."
        )
    
    # Check for negative volumes
    if (df['volume'] < 0).any():
        raise ValidationError("Volume cannot be negative")
    
    # Check for zero or negative prices
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if (df[col] <= 0).any():
            raise ValidationError(f"Price column '{col}' contains zero or negative values")
    
    return df


def validate_date_column(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Validate date column in DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    date_column : str, default='date'
        Name of the date column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with validated date column
        
    Raises:
    -------
    ValidationError
        If validation fails
    """
    if date_column not in df.columns:
        raise ValidationError(f"Date column '{date_column}' not found in DataFrame")
    
    # Try to convert to datetime
    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except Exception as e:
        raise ValidationError(
            f"Failed to convert '{date_column}' to datetime: {str(e)}"
        )
    
    # Check for null dates
    if df[date_column].isnull().any():
        raise ValidationError(f"Date column '{date_column}' contains null values")
    
    return df

