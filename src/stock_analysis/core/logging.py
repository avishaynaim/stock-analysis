"""
Logging configuration and utilities.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


class LogFormatter(logging.Formatter):
    """Custom log formatter."""

    SIMPLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
    DETAILED_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        if fmt is None:
            fmt = self.SIMPLE_FORMAT
        super().__init__(fmt, datefmt)


class LoggingManager:
    """Configure and manage logging."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._loggers: dict[str, logging.Logger] = {}

    def setup(self) -> None:
        """Configure logging based on configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handlers

        # Remove existing handlers
        root_logger.handlers = []

        # Console handler
        console_config = self.config.get("console", {})
        if console_config.get("enabled", True):
            self._setup_console_handler(root_logger, console_config)

        # File handler
        file_config = self.config.get("file", {})
        if file_config.get("enabled", False):
            self._setup_file_handler(root_logger, file_config)

        # Set component-specific levels
        for component, level in self.config.get("components", {}).items():
            logger = logging.getLogger(f"stock_analysis.{component}")
            logger.setLevel(getattr(logging, level.upper()))

    def _setup_console_handler(
        self, logger: logging.Logger, config: dict
    ) -> None:
        """Setup console logging handler."""
        level = getattr(logging, config.get("level", "INFO").upper())

        if config.get("colors", True) and sys.stdout.isatty():
            # Use rich handler for colorful output
            handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
            )
        else:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(LogFormatter(datefmt="%H:%M:%S"))

        handler.setLevel(level)
        logger.addHandler(handler)

    def _setup_file_handler(
        self, logger: logging.Logger, config: dict
    ) -> None:
        """Setup file logging handler."""
        log_path = Path(config.get("path", "stock_analysis.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            log_path,
            maxBytes=config.get("max_size_mb", 100) * 1024 * 1024,
            backupCount=config.get("backup_count", 5),
        )
        handler.setLevel(getattr(logging, config.get("level", "DEBUG").upper()))
        handler.setFormatter(
            LogFormatter(
                fmt=LogFormatter.DETAILED_FORMAT,
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        logger.addHandler(handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger."""
        full_name = f"stock_analysis.{name}"

        if full_name not in self._loggers:
            self._loggers[full_name] = logging.getLogger(full_name)

        return self._loggers[full_name]


# Global logging manager
_logging_manager: Optional[LoggingManager] = None


def setup_logging(config: Optional[dict] = None) -> None:
    """Setup logging from configuration."""
    global _logging_manager
    _logging_manager = LoggingManager(config)
    _logging_manager.setup()


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a component."""
    global _logging_manager
    if _logging_manager:
        return _logging_manager.get_logger(name)
    return logging.getLogger(f"stock_analysis.{name}")
