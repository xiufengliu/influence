"""
Logging utilities for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import sys
from pathlib import Path

import config


def setup_logger(name, level=None, log_file=None):
    """
    Set up a logger with the specified name and level.
    
    Parameters
    ----------
    name : str
        Logger name.
    level : str or int, default=None
        Logging level. If None, uses the level from config.
    log_file : str, default=None
        Path to log file. If None, logs to console only.
        
    Returns
    -------
    logging.Logger
        Configured logger.
    """
    # Get level from config if not specified
    if level is None:
        level = config.LOGGING_PARAMS['level']
    
    # Convert string level to numeric
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(config.LOGGING_PARAMS['format'])
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file is not None:
        # Create directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
