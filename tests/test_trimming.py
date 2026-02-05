"""Tests for the conversation history trimming module."""

import pytest

from fai.dialogue.trimming import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_HISTORY_TOKENS,
    estimate_history_tokens,
    estimate_tokens,
    trim_history,
)

# --- estimate_tokens tests ---


def test_estimate_tokens_empty_string() -> None:
    """Verify estimate_tokens returns 0 for empty string."""
    assert estimate_tokens("") == 0


def test_estimate_tokens_short_text() -> None:
    """Verify estimate_tokens returns minimum 1 for non-empty short text."""
    assert estimate_tokens("hi") == 1


def test_estimate_tokens_known_length() -> None:
    """Verify estimate_tokens uses CHARS_PER_TOKEN ratio."""
    # 20 chars / 4 chars_per_token = 5 tokens
    text = "a" * 20
    assert estimate_tokens(text) == 20 // CHARS_PER_TOKEN


def test_estimate_tokens_longer_text() -> None:
    """Verify estimate_tokens scales with text length."""
    short = estimate_tokens("Hello")
    long = estimate_tokens("Hello " * 100)
    assert long > short


def test_estimate_tokens_minimum_one_for_nonempty() -> None:
    """Verify estimate_tokens returns at least 1 for any non-empty text."""
    # 1 char / 4 = 0, but should be clamped to 1
    assert estimate_tokens("a") == 1
    assert estimate_tokens("ab") == 1
    assert estimate_tokens("abc") == 1


# --- estimate_history_tokens tests ---


def test_estimate_history_tokens_empty_history() -> None:
    """Verify estimate_history_tokens returns 0 for empty history."""
    assert estimate_history_tokens([]) == 0


def test_estimate_history_tokens_single_message() -> None:
    """Verify estimate_history_tokens counts a single message."""
    history = [{"role": "user", "content": "Hello world!"}]
    expected = estimate_tokens("Hello world!")
    assert estimate_history_tokens(history) == expected


def test_estimate_history_tokens_multiple_messages() -> None:
    """Verify estimate_history_tokens sums all messages."""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
    ]
    expected = estimate_tokens("Hello") + estimate_tokens(
        "Hi there! How can I help you?"
    )
    assert estimate_history_tokens(history) == expected


def test_estimate_history_tokens_long_conversation() -> None:
    """Verify estimate_history_tokens handles many turns."""
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
        for i in range(20)
    ]
    total = sum(estimate_tokens(msg["content"]) for msg in history)
    assert estimate_history_tokens(history) == total


# --- trim_history tests ---


def test_trim_history_empty_history() -> None:
    """Verify trim_history returns empty list for empty history."""
    assert trim_history([], max_tokens=100) == []


def test_trim_history_within_budget_returns_unchanged() -> None:
    """Verify trim_history returns history unchanged when within budget."""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    result = trim_history(history, max_tokens=1000)
    assert result == history


def test_trim_history_removes_oldest_pairs() -> None:
    """Verify trim_history removes oldest message pairs first."""
    # Create history that exceeds a small token limit
    history = [
        {"role": "user", "content": "a" * 100},  # ~25 tokens
        {"role": "assistant", "content": "b" * 100},  # ~25 tokens
        {"role": "user", "content": "c" * 100},  # ~25 tokens
        {"role": "assistant", "content": "d" * 100},  # ~25 tokens
    ]
    # Total ~100 tokens, trim to 60 tokens -> should remove first pair
    result = trim_history(history, max_tokens=60)
    assert len(result) == 2
    assert result[0]["content"] == "c" * 100
    assert result[1]["content"] == "d" * 100


def test_trim_history_removes_multiple_pairs() -> None:
    """Verify trim_history can remove multiple oldest pairs."""
    history = [
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
        {"role": "user", "content": "c" * 100},
        {"role": "assistant", "content": "d" * 100},
        {"role": "user", "content": "e" * 100},
        {"role": "assistant", "content": "f" * 100},
    ]
    # Each message ~25 tokens, total ~150 tokens
    # With max_tokens=55, only last pair (~50 tokens) should remain
    result = trim_history(history, max_tokens=55)
    assert len(result) == 2
    assert result[0]["content"] == "e" * 100
    assert result[1]["content"] == "f" * 100


def test_trim_history_preserves_most_recent() -> None:
    """Verify trim_history always keeps the most recent messages."""
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
    ]
    # Set limit so only the last pair fits
    last_pair_tokens = estimate_tokens("new question") + estimate_tokens("new answer")
    result = trim_history(history, max_tokens=last_pair_tokens)
    assert len(result) == 2
    assert result[0]["content"] == "new question"
    assert result[1]["content"] == "new answer"


def test_trim_history_keeps_last_message_when_single_exceeds_budget() -> None:
    """Verify trim_history keeps at least one message even if it exceeds budget."""
    history = [
        {"role": "user", "content": "a" * 1000},  # ~250 tokens
    ]
    # Even with max_tokens=1, we should keep the last message
    result = trim_history(history, max_tokens=1)
    assert len(result) == 1
    assert result[0]["content"] == "a" * 1000


def test_trim_history_invalid_max_tokens_raises() -> None:
    """Verify trim_history raises ValueError for non-positive max_tokens."""
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        trim_history([], max_tokens=0)

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        trim_history([], max_tokens=-1)


def test_trim_history_uses_default_max_tokens() -> None:
    """Verify trim_history uses DEFAULT_MAX_HISTORY_TOKENS when not specified."""
    # Create history well within default limit
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    result = trim_history(history)
    assert result == history


def test_trim_history_default_trims_large_history() -> None:
    """Verify trim_history trims when history exceeds default limit."""
    # Create history that exceeds DEFAULT_MAX_HISTORY_TOKENS
    # Each message ~250 tokens, so 40 messages ~10000 tokens > 4096 default
    history = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": "x" * 1000,
        }
        for i in range(40)
    ]
    result = trim_history(history)
    assert len(result) < len(history)
    assert estimate_history_tokens(result) <= DEFAULT_MAX_HISTORY_TOKENS


def test_trim_history_does_not_mutate_input() -> None:
    """Verify trim_history does not mutate the original history list."""
    history = [
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
        {"role": "user", "content": "c" * 100},
        {"role": "assistant", "content": "d" * 100},
    ]
    original_len = len(history)
    trim_history(history, max_tokens=30)
    assert len(history) == original_len


def test_trim_history_result_within_budget() -> None:
    """Verify trimmed history token count is within max_tokens."""
    history = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message number {i} with some content " * 10,
        }
        for i in range(20)
    ]
    max_tokens = 200
    result = trim_history(history, max_tokens=max_tokens)
    assert estimate_history_tokens(result) <= max_tokens


def test_trim_history_pair_removal_with_odd_messages() -> None:
    """Verify trim_history handles odd number of messages (single remaining)."""
    # 3 messages: when we remove first pair, 1 remains
    history = [
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
        {"role": "user", "content": "c" * 100},
    ]
    # Limit that only allows ~1 message
    result = trim_history(history, max_tokens=30)
    assert len(result) == 1
    assert result[0]["content"] == "c" * 100


def test_constants_are_reasonable() -> None:
    """Verify module constants have reasonable values."""
    assert CHARS_PER_TOKEN == 4
    assert DEFAULT_MAX_HISTORY_TOKENS == 4096
    assert DEFAULT_MAX_HISTORY_TOKENS > 0
