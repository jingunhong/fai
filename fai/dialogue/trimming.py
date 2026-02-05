"""Conversation history trimming for long sessions.

Limits conversation history by estimating token count and removing
oldest messages when the total exceeds a configurable maximum.
"""

from fai.logging import get_logger

logger = get_logger(__name__)

# Approximate characters per token (conservative estimate that works
# across both OpenAI and Anthropic models).
CHARS_PER_TOKEN = 4

# Default maximum tokens for conversation history.
DEFAULT_MAX_HISTORY_TOKENS = 4096


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple character-based heuristic (4 chars ≈ 1 token) that works
    reasonably well across both OpenAI and Anthropic models.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count (minimum 1 for non-empty text).
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_history_tokens(history: list[dict[str, str]]) -> int:
    """Estimate the total token count for a conversation history.

    Args:
        history: Conversation history with "role" and "content" keys.

    Returns:
        Estimated total token count across all messages.
    """
    total = 0
    for msg in history:
        total += estimate_tokens(msg["content"])
    return total


def trim_history(
    history: list[dict[str, str]],
    max_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
) -> list[dict[str, str]]:
    """Trim conversation history to fit within a token budget.

    Removes the oldest message pairs (user + assistant) from the front of
    the history until the total estimated tokens is within the limit.
    Messages are always removed in pairs to maintain conversation coherence.

    Args:
        history: Conversation history with "role" and "content" keys.
            Expected to contain alternating user/assistant message pairs.
        max_tokens: Maximum allowed tokens for the history.

    Returns:
        Trimmed history that fits within the token budget.

    Raises:
        ValueError: If max_tokens is not positive.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    if not history:
        return []

    total_tokens = estimate_history_tokens(history)

    if total_tokens <= max_tokens:
        return history

    # Remove messages from the front in pairs (user + assistant)
    trimmed = list(history)
    while estimate_history_tokens(trimmed) > max_tokens and len(trimmed) >= 2:
        removed = trimmed[:2]
        trimmed = trimmed[2:]
        logger.debug(
            "Trimmed oldest pair: %s / %s",
            removed[0]["content"][:30],
            removed[1]["content"][:30],
        )

    # If still over budget with a single message remaining, keep it
    # (we never return an empty list if there was content)
    if not trimmed and history:
        trimmed = [history[-1]]
        logger.debug("History reduced to single most recent message")

    logger.info(
        "Trimmed history from %d to %d messages (%d → %d est. tokens)",
        len(history),
        len(trimmed),
        total_tokens,
        estimate_history_tokens(trimmed),
    )

    return trimmed
