"""Logging utilities with rich formatting."""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


_console = Console()
_initialized = False


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging with rich formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    global _initialized
    
    if _initialized:
        return
    
    handlers = [
        RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
    ]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    if not _initialized:
        setup_logging()
    
    return logging.getLogger(name)


def get_console() -> Console:
    """Get the rich console instance for direct output.
    
    Returns:
        Rich Console instance
    """
    return _console
