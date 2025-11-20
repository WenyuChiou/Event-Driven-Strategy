"""
Logging configuration module for the trading event detection system.

Provides unified logging setup with file and console handlers,
different log levels, and formatted output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "trading_system",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Set up and configure a logger with file and console handlers.
    
    Parameters:
    -----------
    name : str, default="trading_system"
        Name of the logger
    log_file : str, optional
        Path to log file. If None, uses default log file name.
    log_level : int, default=logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_dir : Path, optional
        Directory for log files. If None, uses project root/logs.
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file or log_dir:
        if log_dir is None:
            # Use project root/logs directory
            from src.config import Config
            log_dir = Config.BASE_DIR / "logs"
            log_dir.mkdir(exist_ok=True)
        
        if log_file is None:
            log_file = f"{name}.log"
        
        log_path = Path(log_dir) / log_file
        
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "trading_system") -> logging.Logger:
    """
    Get or create a logger instance.
    
    Parameters:
    -----------
    name : str, default="trading_system"
        Name of the logger
        
    Returns:
    --------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


# Default logger instance
default_logger = get_logger("trading_system")

