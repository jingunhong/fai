"""Tests for the retry module."""

from unittest.mock import MagicMock, patch

import pytest
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    APITimeoutError as AnthropicAPITimeoutError,
)
from anthropic import (
    InternalServerError as AnthropicInternalServerError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from fai.retry import (
    DEFAULT_BASE_DELAY,
    DEFAULT_EXPONENTIAL_BASE,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    calculate_delay,
    retry_with_backoff,
)


def test_retry_with_backoff_success_on_first_try() -> None:
    """Verify decorated function succeeds on first try without retrying."""
    mock_func = MagicMock(return_value="success")
    decorated = retry_with_backoff()(mock_func)

    result = decorated()

    assert result == "success"
    assert mock_func.call_count == 1


def test_retry_with_backoff_success_after_retries() -> None:
    """Verify decorated function succeeds after transient failures."""
    mock_func = MagicMock(
        side_effect=[
            RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff()(mock_func)

    with patch("fai.retry.time.sleep"):
        result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_with_backoff_exhausts_retries() -> None:
    """Verify decorated function raises after exhausting retries."""
    error = RateLimitError(
        message="Rate limited",
        response=MagicMock(status_code=429),
        body=None,
    )
    mock_func = MagicMock(side_effect=error)
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff(max_retries=2)(mock_func)

    with patch("fai.retry.time.sleep"), pytest.raises(RateLimitError):
        decorated()

    # Initial attempt + 2 retries = 3 total
    assert mock_func.call_count == 3


def test_retry_with_backoff_does_not_retry_on_non_retryable_error() -> None:
    """Verify non-retryable errors are raised immediately."""
    mock_func = MagicMock(side_effect=ValueError("Invalid input"))
    decorated = retry_with_backoff()(mock_func)

    with pytest.raises(ValueError, match="Invalid input"):
        decorated()

    assert mock_func.call_count == 1


def test_retry_with_backoff_retries_on_api_connection_error() -> None:
    """Verify retry on APIConnectionError."""
    mock_func = MagicMock(
        side_effect=[
            APIConnectionError(request=MagicMock()),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff()(mock_func)

    with patch("fai.retry.time.sleep"):
        result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_with_backoff_retries_on_api_timeout_error() -> None:
    """Verify retry on APITimeoutError."""
    mock_func = MagicMock(
        side_effect=[
            APITimeoutError(request=MagicMock()),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff()(mock_func)

    with patch("fai.retry.time.sleep"):
        result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_with_backoff_retries_on_internal_server_error() -> None:
    """Verify retry on InternalServerError."""
    mock_func = MagicMock(
        side_effect=[
            InternalServerError(
                message="Server error",
                response=MagicMock(status_code=500),
                body=None,
            ),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff()(mock_func)

    with patch("fai.retry.time.sleep"):
        result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_with_backoff_calls_sleep_with_delay() -> None:
    """Verify sleep is called with appropriate delay between retries."""
    mock_func = MagicMock(
        side_effect=[
            RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff(base_delay=1.0, jitter=False)(mock_func)

    with patch("fai.retry.time.sleep") as mock_sleep:
        decorated()

    mock_sleep.assert_called_once_with(1.0)


def test_retry_with_backoff_exponential_delay() -> None:
    """Verify delay increases exponentially between retries."""
    mock_func = MagicMock(
        side_effect=[
            RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff(
        base_delay=1.0, exponential_base=2.0, jitter=False, max_retries=3
    )(mock_func)

    with patch("fai.retry.time.sleep") as mock_sleep:
        decorated()

    # Delays: 1.0, 2.0, 4.0
    assert mock_sleep.call_count == 3
    assert mock_sleep.call_args_list[0][0][0] == 1.0
    assert mock_sleep.call_args_list[1][0][0] == 2.0
    assert mock_sleep.call_args_list[2][0][0] == 4.0


def test_retry_with_backoff_respects_max_delay() -> None:
    """Verify delay is capped at max_delay."""
    mock_func = MagicMock(
        side_effect=[
            RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff(
        base_delay=10.0, max_delay=5.0, exponential_base=2.0, jitter=False
    )(mock_func)

    with patch("fai.retry.time.sleep") as mock_sleep:
        decorated()

    # Both delays should be capped at 5.0
    assert mock_sleep.call_args_list[0][0][0] == 5.0
    assert mock_sleep.call_args_list[1][0][0] == 5.0


def test_retry_with_backoff_preserves_function_metadata() -> None:
    """Verify decorated function preserves original function metadata."""

    @retry_with_backoff()
    def my_function() -> str:
        """My docstring."""
        return "result"

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."


def test_retry_with_backoff_passes_arguments() -> None:
    """Verify decorated function passes arguments correctly."""

    @retry_with_backoff()
    def add(a: int, b: int) -> int:
        return a + b

    result = add(2, 3)
    assert result == 5


def test_retry_with_backoff_passes_keyword_arguments() -> None:
    """Verify decorated function passes keyword arguments correctly."""

    @retry_with_backoff()
    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    result = greet("Alice", greeting="Hi")
    assert result == "Hi, Alice!"


def test_retry_with_backoff_custom_max_retries() -> None:
    """Verify custom max_retries is respected."""
    mock_func = MagicMock(
        side_effect=RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff(max_retries=5)(mock_func)

    with patch("fai.retry.time.sleep"), pytest.raises(RateLimitError):
        decorated()

    # Initial attempt + 5 retries = 6 total
    assert mock_func.call_count == 6


def test_retry_with_backoff_zero_max_retries() -> None:
    """Verify max_retries=0 means no retries."""
    mock_func = MagicMock(
        side_effect=RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff(max_retries=0)(mock_func)

    with pytest.raises(RateLimitError):
        decorated()

    assert mock_func.call_count == 1


def test_calculate_delay_basic() -> None:
    """Verify basic delay calculation."""
    delay = calculate_delay(
        attempt=0, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=False
    )
    assert delay == 1.0


def test_calculate_delay_exponential() -> None:
    """Verify exponential increase in delay."""
    delay_0 = calculate_delay(
        attempt=0, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=False
    )
    delay_1 = calculate_delay(
        attempt=1, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=False
    )
    delay_2 = calculate_delay(
        attempt=2, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=False
    )

    assert delay_0 == 1.0
    assert delay_1 == 2.0
    assert delay_2 == 4.0


def test_calculate_delay_capped_at_max() -> None:
    """Verify delay is capped at max_delay."""
    delay = calculate_delay(
        attempt=10, base_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False
    )
    assert delay == 5.0


def test_calculate_delay_with_jitter() -> None:
    """Verify jitter adds randomness to delay."""
    delays = set()
    for _ in range(10):
        delay = calculate_delay(
            attempt=0, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=True
        )
        delays.add(delay)

    # With jitter, we should get different delays
    assert len(delays) > 1
    # Jitter should keep delays between 0.5 and 2.0 (base * 0.5 to base * 1.5)
    for delay in delays:
        assert 0.5 <= delay <= 2.0


def test_calculate_delay_with_different_base() -> None:
    """Verify different exponential base values."""
    delay = calculate_delay(
        attempt=2, base_delay=1.0, max_delay=60.0, exponential_base=3.0, jitter=False
    )
    assert delay == 9.0  # 1.0 * 3^2


def test_default_constants() -> None:
    """Verify default constants have expected values."""
    assert DEFAULT_MAX_RETRIES == 3
    assert DEFAULT_BASE_DELAY == 1.0
    assert DEFAULT_MAX_DELAY == 60.0
    assert DEFAULT_EXPONENTIAL_BASE == 2.0


def test_retry_with_backoff_logs_on_retry() -> None:
    """Verify retry attempts are logged."""
    mock_func = MagicMock(
        side_effect=[
            RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff()(mock_func)

    with (
        patch("fai.retry.time.sleep"),
        patch("fai.retry.logger") as mock_logger,
    ):
        decorated()

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0]
    assert "Attempt 1/4 failed" in call_args[0] % call_args[1:]


def test_retry_with_backoff_logs_on_exhaustion() -> None:
    """Verify final failure is logged as warning."""
    mock_func = MagicMock(
        side_effect=RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff(max_retries=1)(mock_func)

    with (
        patch("fai.retry.time.sleep"),
        patch("fai.retry.logger") as mock_logger,
        pytest.raises(RateLimitError),
    ):
        decorated()

    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args[0]
    assert "All 2 retry attempts failed" in call_args[0] % call_args[1:]


# Tests for Anthropic exception handling


def test_retry_with_backoff_retries_on_anthropic_rate_limit_error() -> None:
    """Verify retry on Anthropic RateLimitError."""
    mock_func = MagicMock(
        side_effect=[
            AnthropicRateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff()(mock_func)

    with patch("fai.retry.time.sleep"):
        result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_with_backoff_retries_on_anthropic_api_connection_error() -> None:
    """Verify retry on Anthropic APIConnectionError."""
    mock_func = MagicMock(
        side_effect=[
            AnthropicAPIConnectionError(request=MagicMock()),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff()(mock_func)

    with patch("fai.retry.time.sleep"):
        result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_with_backoff_retries_on_anthropic_api_timeout_error() -> None:
    """Verify retry on Anthropic APITimeoutError."""
    mock_func = MagicMock(
        side_effect=[
            AnthropicAPITimeoutError(request=MagicMock()),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff()(mock_func)

    with patch("fai.retry.time.sleep"):
        result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_with_backoff_retries_on_anthropic_internal_server_error() -> None:
    """Verify retry on Anthropic InternalServerError."""
    mock_func = MagicMock(
        side_effect=[
            AnthropicInternalServerError(
                message="Server error",
                response=MagicMock(status_code=500),
                body=None,
            ),
            "success",
        ]
    )
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff()(mock_func)

    with patch("fai.retry.time.sleep"):
        result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_with_backoff_exhausts_retries_on_anthropic_error() -> None:
    """Verify function raises after exhausting retries with Anthropic error."""
    error = AnthropicRateLimitError(
        message="Rate limited",
        response=MagicMock(status_code=429),
        body=None,
    )
    mock_func = MagicMock(side_effect=error)
    mock_func.__name__ = "mock_func"
    decorated = retry_with_backoff(max_retries=2)(mock_func)

    with patch("fai.retry.time.sleep"), pytest.raises(AnthropicRateLimitError):
        decorated()

    # Initial attempt + 2 retries = 3 total
    assert mock_func.call_count == 3
