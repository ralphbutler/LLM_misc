import os
import logging
import time
from pathlib import Path
from typing import Optional, Union


class Logger:
    """Custom logger that writes to both console and file."""
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[Union[str, Path]] = None,
        level: int = logging.INFO,
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory to store log files (defaults to ./logs)
            level: Logging level
        """
        self.name = name
        self.level = level
        
        # Set up the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Define a standard formatter
        log_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add console handler with standard formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)

        # Add file handler if log_dir is provided
        if log_dir is not None:
            # Create log directory if it doesn't exist
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Use a fixed log file name
            log_file_name = "aider_mcp_server.log"
            log_file_path = log_dir / log_file_name

            # Set up file handler to append
            file_handler = logging.FileHandler(log_file_path, mode='a')
            # Use the same formatter as the console handler
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)

            self.log_file_path = log_file_path
            self.logger.info(f"Logging to: {log_file_path}")

    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log a critical message."""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log an exception message with traceback."""
        self.logger.exception(message, **kwargs)


def get_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files (defaults to ./logs)
        level: Logging level

    Returns:
        Configured Logger instance
    """
    if log_dir is None:
        # Default log directory is ./logs
        log_dir = Path("./logs")
    
    return Logger(
        name=name,
        log_dir=log_dir,
        level=level,
    )
