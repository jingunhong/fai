"""Tests for the logging module."""

import io
import logging
from collections.abc import Generator

import pytest

from fai.logging import (
    DEFAULT_FORMAT,
    LOG_LEVEL_MAP,
    PACKAGE_NAME,
    SIMPLE_FORMAT,
    get_logger,
    is_logging_configured,
    reset_logging,
    setup_logging,
)


@pytest.fixture(autouse=True)  # type: ignore[misc]
def reset_logging_state() -> Generator[None, None, None]:
    """Reset logging state before and after each test."""
    reset_logging()
    yield
    reset_logging()


def test_setup_logging_default_level() -> None:
    """Verify default log level is warning."""
    stream = io.StringIO()
    setup_logging(stream=stream)

    logger = get_logger("test")
    logger.warning("warning message")
    logger.info("info message")

    output = stream.getvalue()
    assert "warning message" in output
    assert "info message" not in output


def test_setup_logging_debug_level() -> None:
    """Verify debug level logs debug messages."""
    stream = io.StringIO()
    setup_logging(level="debug", stream=stream)

    logger = get_logger("test")
    logger.debug("debug message")

    output = stream.getvalue()
    assert "debug message" in output


def test_setup_logging_info_level() -> None:
    """Verify info level logs info and above."""
    stream = io.StringIO()
    setup_logging(level="info", stream=stream)

    logger = get_logger("test")
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")

    output = stream.getvalue()
    assert "debug message" not in output
    assert "info message" in output
    assert "warning message" in output


def test_setup_logging_error_level() -> None:
    """Verify error level only logs error and critical."""
    stream = io.StringIO()
    setup_logging(level="error", stream=stream)

    logger = get_logger("test")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    output = stream.getvalue()
    assert "warning message" not in output
    assert "error message" in output
    assert "critical message" in output


def test_setup_logging_critical_level() -> None:
    """Verify critical level only logs critical messages."""
    stream = io.StringIO()
    setup_logging(level="critical", stream=stream)

    logger = get_logger("test")
    logger.error("error message")
    logger.critical("critical message")

    output = stream.getvalue()
    assert "error message" not in output
    assert "critical message" in output


def test_setup_logging_invalid_level() -> None:
    """Verify invalid log level raises ValueError."""
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging(level="invalid")  # type: ignore[arg-type]


def test_setup_logging_custom_format() -> None:
    """Verify custom format string is used."""
    stream = io.StringIO()
    custom_format = "CUSTOM: %(message)s"
    setup_logging(level="info", format_string=custom_format, stream=stream)

    logger = get_logger("test")
    logger.info("test message")

    output = stream.getvalue()
    assert "CUSTOM: test message" in output


def test_setup_logging_reconfiguration() -> None:
    """Verify logging can be reconfigured without duplicate handlers."""
    stream1 = io.StringIO()
    stream2 = io.StringIO()

    # First configuration
    setup_logging(level="info", stream=stream1)
    logger = get_logger("test")
    logger.info("message 1")

    # Reconfigure
    setup_logging(level="debug", stream=stream2)
    logger.debug("message 2")

    # Check that message 2 only appears in stream2
    assert "message 1" in stream1.getvalue()
    assert "message 2" not in stream1.getvalue()
    assert "message 2" in stream2.getvalue()


def test_get_logger_returns_child_logger() -> None:
    """Verify get_logger returns a child of package logger."""
    logger = get_logger("mymodule")
    assert logger.name == f"{PACKAGE_NAME}.mymodule"


def test_get_logger_with_package_prefix() -> None:
    """Verify get_logger handles names already prefixed with package name."""
    logger = get_logger(f"{PACKAGE_NAME}.mymodule")
    assert logger.name == f"{PACKAGE_NAME}.mymodule"


def test_get_logger_inherits_level() -> None:
    """Verify child logger inherits level from package logger."""
    stream = io.StringIO()
    setup_logging(level="info", stream=stream)

    logger = get_logger("child")
    logger.debug("debug message")
    logger.info("info message")

    output = stream.getvalue()
    assert "debug message" not in output
    assert "info message" in output


def test_is_logging_configured_false_initially() -> None:
    """Verify is_logging_configured returns False before setup."""
    assert is_logging_configured() is False


def test_is_logging_configured_true_after_setup() -> None:
    """Verify is_logging_configured returns True after setup."""
    setup_logging()
    assert is_logging_configured() is True


def test_reset_logging() -> None:
    """Verify reset_logging clears configuration."""
    setup_logging()
    assert is_logging_configured() is True

    reset_logging()
    assert is_logging_configured() is False


def test_reset_logging_clears_handlers() -> None:
    """Verify reset_logging removes all handlers."""
    setup_logging()
    package_logger = logging.getLogger(PACKAGE_NAME)
    assert len(package_logger.handlers) > 0

    reset_logging()
    assert len(package_logger.handlers) == 0


def test_log_level_map_contains_all_levels() -> None:
    """Verify LOG_LEVEL_MAP contains all expected levels."""
    expected_levels = {"debug", "info", "warning", "error", "critical"}
    assert set(LOG_LEVEL_MAP.keys()) == expected_levels


def test_log_level_map_values() -> None:
    """Verify LOG_LEVEL_MAP maps to correct logging constants."""
    assert LOG_LEVEL_MAP["debug"] == logging.DEBUG
    assert LOG_LEVEL_MAP["info"] == logging.INFO
    assert LOG_LEVEL_MAP["warning"] == logging.WARNING
    assert LOG_LEVEL_MAP["error"] == logging.ERROR
    assert LOG_LEVEL_MAP["critical"] == logging.CRITICAL


def test_default_format_constant() -> None:
    """Verify DEFAULT_FORMAT constant is defined."""
    assert "%(asctime)s" in DEFAULT_FORMAT
    assert "%(name)s" in DEFAULT_FORMAT
    assert "%(levelname)s" in DEFAULT_FORMAT
    assert "%(message)s" in DEFAULT_FORMAT


def test_simple_format_constant() -> None:
    """Verify SIMPLE_FORMAT constant is defined."""
    assert "%(levelname)s" in SIMPLE_FORMAT
    assert "%(message)s" in SIMPLE_FORMAT


def test_package_name_constant() -> None:
    """Verify PACKAGE_NAME constant is 'fai'."""
    assert PACKAGE_NAME == "fai"


def test_logger_no_propagation() -> None:
    """Verify logger does not propagate to root logger."""
    setup_logging()
    package_logger = logging.getLogger(PACKAGE_NAME)
    assert package_logger.propagate is False


def test_logging_with_format_arguments() -> None:
    """Verify logging works with format arguments."""
    stream = io.StringIO()
    setup_logging(level="info", stream=stream)

    logger = get_logger("test")
    logger.info("User %s logged in from %s", "alice", "192.168.1.1")

    output = stream.getvalue()
    assert "User alice logged in from 192.168.1.1" in output


def test_logging_with_exception() -> None:
    """Verify logging captures exception info."""
    stream = io.StringIO()
    setup_logging(level="error", stream=stream)

    logger = get_logger("test")
    try:
        raise ValueError("test error")
    except ValueError:
        logger.exception("An error occurred")

    output = stream.getvalue()
    assert "An error occurred" in output
    assert "ValueError: test error" in output


def test_multiple_loggers_share_configuration() -> None:
    """Verify multiple loggers share the same configuration."""
    stream = io.StringIO()
    setup_logging(level="info", stream=stream)

    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    logger1.info("from module1")
    logger2.info("from module2")

    output = stream.getvalue()
    assert "from module1" in output
    assert "from module2" in output


def test_setup_logging_uses_simple_format_by_default() -> None:
    """Verify setup_logging uses SIMPLE_FORMAT when no format specified."""
    stream = io.StringIO()
    setup_logging(level="info", stream=stream)

    logger = get_logger("test")
    logger.info("test message")

    output = stream.getvalue()
    # Simple format should be "LEVEL: message"
    assert "INFO: test message" in output
    # Should not have timestamp (which DEFAULT_FORMAT has)
    assert "%(asctime)s" not in output
