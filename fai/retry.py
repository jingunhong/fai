"""Retry logic with exponential backoff for API calls."""

import logging
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds
DEFAULT_EXPONENTIAL_BASE = 2.0

# Exceptions that should trigger a retry (transient errors)
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    jitter: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff calculation.
        jitter: Whether to add random jitter to delay.

    Returns:
        Decorated function that retries on transient failures.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.warning(
                            "All %d retry attempts failed for %s",
                            max_retries + 1,
                            func.__name__,
                        )
                        raise

                    delay = calculate_delay(
                        attempt, base_delay, max_delay, exponential_base, jitter
                    )
                    logger.info(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.2fs...",
                        attempt + 1,
                        max_retries + 1,
                        func.__name__,
                        type(e).__name__,
                        delay,
                    )
                    time.sleep(delay)

            # This should never be reached, but satisfies type checker
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
) -> float:
    """Calculate delay for a retry attempt with exponential backoff.

    Args:
        attempt: The current attempt number (0-indexed).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        exponential_base: Base for exponential calculation.
        jitter: Whether to add random jitter.

    Returns:
        Delay in seconds.
    """
    delay = base_delay * (exponential_base**attempt)
    delay = min(delay, max_delay)

    if jitter:
        # Add random jitter between 0% and 100% of the delay
        delay = delay * (0.5 + random.random())  # noqa: S311

    return delay
