"""
Logging Configuration Module for Hybrid Information Retrieval System

This module provides centralized logging configuration with support for:
- Environment-based log levels (DEBUG, INFO, WARNING)
- File handler with rotation to prevent log file growth
- Console handler for stdout output
- Both handlers using consistent format
- Automatic logs directory creation
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


# Global logger storage for consistency
_loggers = {}


def get_logger(name: str, config: Optional['Config'] = None) -> logging.Logger:
    """
    Get a configured logger instance by module name.
    
    Creates and configures a logger for the given module name if not already created.
    Reuses existing logger instances on subsequent calls with the same name.
    
    Args:
        name: Module name for the logger (typically __name__)
        config: Optional Config instance for environment and log directory path.
               If not provided, uses default paths and DEBUG level.
               Expected to have attributes: environment, logs_dir
               where environment is one of: "dev", "staging", "prod"
    
    Returns:
        Configured logger instance with file and console handlers
        
    Behavior:
        - File handler: Writes to logs/{name}.log with rotation (5 files, 10MB each)
        - Console handler: Outputs to stdout
        - Log format: %(asctime)s - %(name)s - %(levelname)s - %(message)s
        - Log level determined by environment:
            - DEBUG for "dev" environment
            - INFO for "staging" environment
            - WARNING for "prod" environment
        - Logs directory created automatically if it doesn't exist
    """
    
    # Return cached logger if already configured
    if name in _loggers:
        return _loggers[name]
    
    # Determine log level based on environment
    if config is not None:
        environment = getattr(config, 'environment', 'dev')
        logs_dir = getattr(config, 'logs_dir', './logs')
    else:
        environment = 'dev'
        logs_dir = './logs'
    
    # Map environment to log level
    log_level_map = {
        'dev': logging.DEBUG,
        'staging': logging.INFO,
        'prod': logging.WARNING,
    }
    log_level = log_level_map.get(environment, logging.DEBUG)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Define log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Create logs directory if it doesn't exist
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    
    # File handler with rotation
    # RotatingFileHandler: maxBytes=10MB, backupCount=5 (keeps 5 rotated files)
    log_file = logs_path / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5  # Keep 5 rotated files
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler for stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Cache the logger
    _loggers[name] = logger
    
    return logger


def configure_logging(config: Optional['Config'] = None) -> None:
    """
    Configure logging for the entire application.
    
    This function sets up the root logger and can be called at application startup
    to ensure all logging is properly configured before any modules use loggers.
    
    Args:
        config: Optional Config instance for environment and log directory path
        
    Returns:
        None
    """
    # Get root logger
    root_logger = get_logger('hybrid_ir', config)
    
    # Log startup message
    environment = getattr(config, 'environment', 'dev') if config else 'dev'
    logs_dir = getattr(config, 'logs_dir', './logs') if config else './logs'
    
    root_logger.info(f"Logging configured for {environment} environment")
    root_logger.info(f"Log files directory: {logs_dir}")


if __name__ == "__main__":
    # Test logging configuration
    
    # Test 1: Basic logger without config
    print("Test 1: Creating logger without config (defaults to DEBUG)")
    logger1 = get_logger("test_module_1")
    logger1.debug("Debug message")
    logger1.info("Info message")
    logger1.warning("Warning message")
    print()
    
    # Test 2: Logger with mocked config
    print("Test 2: Creating logger with dev environment config")
    
    class MockConfig:
        environment = "dev"
        logs_dir = "./logs"
    
    logger2 = get_logger("test_module_2", MockConfig())
    logger2.debug("Debug message from dev environment")
    logger2.info("Info message from dev environment")
    print()
    
    # Test 3: Logger with staging environment
    print("Test 3: Creating logger with staging environment config")
    
    class StagingConfig:
        environment = "staging"
        logs_dir = "./logs"
    
    logger3 = get_logger("test_module_3", StagingConfig())
    logger3.debug("Debug message (should not appear in staging)")
    logger3.info("Info message (should appear in staging)")
    logger3.warning("Warning message (should appear in staging)")
    print()
    
    # Test 4: Logger with prod environment
    print("Test 4: Creating logger with prod environment config")
    
    class ProdConfig:
        environment = "prod"
        logs_dir = "./logs"
    
    logger4 = get_logger("test_module_4", ProdConfig())
    logger4.debug("Debug message (should not appear in prod)")
    logger4.info("Info message (should not appear in prod)")
    logger4.warning("Warning message (should appear in prod)")
    print()
    
    # Test 5: Logger caching (same logger returned)
    print("Test 5: Verifying logger caching")
    logger1_again = get_logger("test_module_1")
    print(f"Logger reused: {logger1 is logger1_again}")
    print()
    
    print("Logging tests completed. Check ./logs/ directory for log files.")
