"""
Unit tests for the logging module.

Tests verify:
- Logger creation and caching
- Environment-based log level configuration
- File and console handler setup
- Log directory creation
- Proper integration with Config class
"""

import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the logging module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logger import get_logger, configure_logging, _loggers


class TestGetLogger:
    """Tests for the get_logger function."""
    
    def setup_method(self):
        """Clear cached loggers before each test."""
        # Close all handlers and clear cached loggers
        for logger in _loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        _loggers.clear()
    
    def teardown_method(self):
        """Clean up loggers after each test."""
        # Close all handlers and clear cached loggers
        for logger in _loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        _loggers.clear()
    
    def test_get_logger_creates_logger(self):
        """Test that get_logger creates a logger instance."""
        logger = get_logger("test_module")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_get_logger_caching(self):
        """Test that get_logger returns cached instances."""
        logger1 = get_logger("cached_module")
        logger2 = get_logger("cached_module")
        assert logger1 is logger2, "Same logger should be returned on subsequent calls"
    
    def test_get_logger_dev_environment(self):
        """Test that dev environment sets DEBUG log level."""
        class DevConfig:
            environment = "dev"
            logs_dir = tempfile.gettempdir()
        
        logger = get_logger("dev_module", DevConfig())
        assert logger.level == logging.DEBUG
    
    def test_get_logger_staging_environment(self):
        """Test that staging environment sets INFO log level."""
        class StagingConfig:
            environment = "staging"
            logs_dir = tempfile.gettempdir()
        
        logger = get_logger("staging_module", StagingConfig())
        assert logger.level == logging.INFO
    
    def test_get_logger_prod_environment(self):
        """Test that prod environment sets WARNING log level."""
        class ProdConfig:
            environment = "prod"
            logs_dir = tempfile.gettempdir()
        
        logger = get_logger("prod_module", ProdConfig())
        assert logger.level == logging.WARNING
    
    def test_get_logger_default_environment(self):
        """Test that unknown environment defaults to DEBUG level."""
        class UnknownConfig:
            environment = "unknown"
            logs_dir = tempfile.gettempdir()
        
        logger = get_logger("unknown_module", UnknownConfig())
        assert logger.level == logging.DEBUG
    
    def test_get_logger_creates_logs_directory(self):
        """Test that get_logger creates the logs directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_path = Path(tmpdir) / "new_logs"
            assert not logs_path.exists(), "Logs directory should not exist yet"
            
            class Config:
                environment = "dev"
                logs_dir = str(logs_path)
            
            logger = get_logger("test_module", Config())
            
            # Check that logs directory was created
            assert logs_path.exists(), "Logs directory should be created"
            
            # Clean up handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    
    def test_get_logger_creates_log_file(self):
        """Test that get_logger creates a log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class Config:
                environment = "dev"
                logs_dir = str(tmpdir)
            
            logger = get_logger("file_test_module_unique", Config())
            log_file = Path(tmpdir) / "file_test_module_unique.log"
            
            # Check file exists after logger creation
            assert log_file.exists(), "Log file should be created"
            
            # Clean up handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    
    def test_get_logger_has_file_handler(self):
        """Test that logger has a file handler."""
        logger = get_logger("handler_test_module")
        
        # Find file handler
        file_handlers = [h for h in logger.handlers 
                        if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handlers) > 0, "Logger should have a RotatingFileHandler"
    
    def test_get_logger_has_console_handler(self):
        """Test that logger has a console handler."""
        logger = get_logger("console_test_module")
        
        # Find stream handler
        stream_handlers = [h for h in logger.handlers 
                          if isinstance(h, logging.StreamHandler) 
                          and not isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(stream_handlers) > 0, "Logger should have a console StreamHandler"
    
    def test_get_logger_format(self):
        """Test that logger handlers use correct format."""
        logger = get_logger("format_test_module")
        
        expected_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        for handler in logger.handlers:
            if handler.formatter:
                assert handler.formatter._fmt == expected_format, \
                    f"Handler format should be '{expected_format}'"
    
    def test_get_logger_without_config_defaults(self):
        """Test that get_logger works without config (uses defaults)."""
        with tempfile.TemporaryDirectory():
            logger = get_logger("no_config_module")
            assert logger is not None
            assert logger.level == logging.DEBUG  # Default to dev environment
    
    def test_logger_messages_written_to_file(self):
        """Test that log messages are actually written to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class Config:
                environment = "dev"
                logs_dir = str(tmpdir)
            
            logger = get_logger("write_test_module_unique", Config())
            test_message = "This is a test message"
            logger.info(test_message)
            
            # Flush handlers to ensure file is written
            for handler in logger.handlers:
                handler.flush()
            
            log_file = Path(tmpdir) / "write_test_module_unique.log"
            with open(log_file, 'r') as f:
                content = f.read()
            
            assert test_message in content, "Log message should be written to file"
            
            # Clean up handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    
    def test_debug_level_logs_debug_messages_in_dev(self):
        """Test that DEBUG level logs debug messages in dev environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class Config:
                environment = "dev"
                logs_dir = str(tmpdir)
            
            logger = get_logger("debug_test_module_unique", Config())
            logger.debug("Debug message")
            
            # Flush handlers
            for handler in logger.handlers:
                handler.flush()
            
            log_file = Path(tmpdir) / "debug_test_module_unique.log"
            with open(log_file, 'r') as f:
                content = f.read()
            
            assert "Debug message" in content, "DEBUG messages should be logged in dev"
            
            # Clean up handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    
    def test_debug_level_suppressed_in_staging(self):
        """Test that DEBUG level is suppressed in staging environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class Config:
                environment = "staging"
                logs_dir = str(tmpdir)
            
            logger = get_logger("debug_suppress_module_unique", Config())
            logger.debug("Debug message (should not appear)")
            logger.info("Info message")
            
            # Flush handlers
            for handler in logger.handlers:
                handler.flush()
            
            log_file = Path(tmpdir) / "debug_suppress_module_unique.log"
            with open(log_file, 'r') as f:
                content = f.read()
            
            assert "Debug message" not in content, "DEBUG messages should not be logged in staging"
            assert "Info message" in content, "INFO messages should be logged in staging"
            
            # Clean up handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    
    def test_rotating_file_handler_configuration(self):
        """Test that RotatingFileHandler is configured correctly."""
        logger = get_logger("rotating_test_module")
        
        # Find rotating file handler
        rotating_handlers = [h for h in logger.handlers 
                            if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(rotating_handlers) == 1, "Should have exactly one RotatingFileHandler"
        
        handler = rotating_handlers[0]
        assert handler.maxBytes == 10 * 1024 * 1024, "maxBytes should be 10MB"
        assert handler.backupCount == 5, "backupCount should be 5"


class TestConfigureLogging:
    """Tests for the configure_logging function."""
    
    def setup_method(self):
        """Clear cached loggers before each test."""
        # Close all handlers and clear cached loggers
        for logger in _loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        _loggers.clear()
    
    def teardown_method(self):
        """Clean up loggers after each test."""
        # Close all handlers and clear cached loggers
        for logger in _loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        _loggers.clear()
    
    def test_configure_logging_initializes_root_logger(self):
        """Test that configure_logging initializes the root logger."""
        class Config:
            environment = "dev"
            logs_dir = tempfile.gettempdir()
        
        configure_logging(Config())
        
        # Check that root logger was created
        assert "hybrid_ir" in _loggers
    
    def test_configure_logging_without_config(self):
        """Test that configure_logging works without config."""
        configure_logging()
        
        # Should not raise an exception
        assert "hybrid_ir" in _loggers


class TestIntegrationWithConfig:
    """Integration tests with the Config class."""
    
    def setup_method(self):
        """Clear cached loggers before each test."""
        # Close all handlers and clear cached loggers
        for logger in _loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        _loggers.clear()
    
    def teardown_method(self):
        """Clean up loggers after each test."""
        # Close all handlers and clear cached loggers
        for logger in _loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        _loggers.clear()
    
    def test_get_logger_with_real_config(self):
        """Test get_logger integration with actual Config class."""
        # Import Config from config module
        from config import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                environment="dev",
                logs_dir=str(tmpdir)
            )
            
            logger = get_logger("config_integration_test_unique", config)
            
            assert logger is not None
            assert logger.level == logging.DEBUG
            
            # Verify logs directory exists
            assert Path(tmpdir).exists()
            
            # Clean up handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
