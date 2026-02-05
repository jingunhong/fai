"""Structured logging configuration for fai."""

import logging
import sys
from typing import IO, Literal

# Log level type alias
LogLevel = Literal["debug", "info", "warning", "error", "critical"]

# Mapping from string log levels to logging constants
LOG_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Simple format for console output (less verbose)
SIMPLE_FORMAT = "%(levelname)s: %(message)s"

# Package logger name
PACKAGE_NAME = "fai"

# Track whether logging has been configured
_logging_configured = False


def setup_logging(
    level: LogLevel = "warning",
    format_string: str | None = None,
    stream: IO[str] | None = None,
) -> None:
    """Configure logging for the fai package.

    This function sets up the root logger for the fai package with the specified
    log level and format. It should be called early in application startup,
    typically from the CLI entry point.

    Args:
        level: Log level to set. One of: debug, info, warning, error, critical.
            Defaults to "warning".
        format_string: Custom format string for log messages. If None, uses
            SIMPLE_FORMAT for console output.
        stream: Stream to write logs to. Defaults to sys.stderr.

    Raises:
        ValueError: If level is not a valid log level string.

    Example:
        >>> setup_logging(level="debug")
        >>> setup_logging(level="info", format_string="%(message)s")
    """
    global _logging_configured

    if level not in LOG_LEVEL_MAP:
        valid_levels = ", ".join(LOG_LEVEL_MAP.keys())
        raise ValueError(f"Invalid log level: {level}. Must be one of: {valid_levels}")

    # Get the numeric log level
    numeric_level = LOG_LEVEL_MAP[level]

    # Use provided format or default to simple format
    fmt = format_string if format_string is not None else SIMPLE_FORMAT

    # Use provided stream or default to stderr
    output_stream = stream if stream is not None else sys.stderr

    # Get the package logger
    logger = logging.getLogger(PACKAGE_NAME)

    # Remove existing handlers to avoid duplicate logs on reconfiguration
    logger.handlers.clear()

    # Set the log level
    logger.setLevel(numeric_level)

    # Create and configure handler
    handler = logging.StreamHandler(output_stream)
    handler.setLevel(numeric_level)

    # Create formatter and add to handler
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    This function returns a logger that is a child of the fai package logger.
    The logger name should typically be __name__ of the calling module.

    Args:
        name: Name for the logger, typically __name__ of the module.

    Returns:
        A logger instance configured as a child of the fai package logger.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    # If the name already starts with the package name, use it as-is
    if name.startswith(f"{PACKAGE_NAME}."):
        return logging.getLogger(name)
    # Otherwise, prefix with package name for proper hierarchy
    return logging.getLogger(f"{PACKAGE_NAME}.{name}")


def is_logging_configured() -> bool:
    """Check whether logging has been configured.

    Returns:
        True if setup_logging has been called, False otherwise.
    """
    return _logging_configured


def reset_logging() -> None:
    """Reset logging configuration to unconfigured state.

    This is primarily useful for testing to ensure clean state between tests.
    """
    global _logging_configured
    logger = logging.getLogger(PACKAGE_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    _logging_configured = False
