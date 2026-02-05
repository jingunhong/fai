"""Tests for the dialogue component."""

from unittest.mock import MagicMock, patch

import pytest

from fai.dialogue import (
    DEFAULT_MODELS,
    MODEL_IDS,
    SYSTEM_PROMPT,
    generate_response,
)
from fai.types import DialogueResponse


@pytest.fixture  # type: ignore[misc]
def mock_openai_response() -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! How can I help you today?"
    return mock_response


@pytest.fixture  # type: ignore[misc]
def mock_openai_client(mock_openai_response: MagicMock) -> MagicMock:
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response
    return mock_client


@pytest.fixture  # type: ignore[misc]
def mock_anthropic_response() -> MagicMock:
    """Create a mock Anthropic message response."""
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.type = "text"
    mock_content.text = "Hello from Claude! How can I assist you?"
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture  # type: ignore[misc]
def mock_anthropic_client(mock_anthropic_response: MagicMock) -> MagicMock:
    """Create a mock Anthropic client."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_anthropic_response
    return mock_client


def test_generate_response_returns_dialogue_response(
    mock_openai_client: MagicMock,
) -> None:
    """Verify generate_response returns a DialogueResponse."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client):
        result = generate_response("Hello", [])

    assert isinstance(result, DialogueResponse)
    assert result.text == "Hello! How can I help you today?"


def test_generate_response_with_empty_history(mock_openai_client: MagicMock) -> None:
    """Verify generate_response works with empty history."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client):
        result = generate_response("What's the weather?", [])

    assert isinstance(result, DialogueResponse)
    mock_openai_client.chat.completions.create.assert_called_once()

    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What's the weather?"


def test_generate_response_with_history(mock_openai_client: MagicMock) -> None:
    """Verify generate_response includes history in the API call."""
    history = [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello! Nice to meet you."},
    ]

    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client):
        generate_response("How are you?", history)

    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]

    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Hi there"}
    assert messages[2] == {"role": "assistant", "content": "Hello! Nice to meet you."}
    assert messages[3] == {"role": "user", "content": "How are you?"}


def test_generate_response_uses_gpt4o_model_by_default(
    mock_openai_client: MagicMock,
) -> None:
    """Verify generate_response uses the gpt-4o model by default."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client):
        generate_response("Test message", [])

    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-4o"


def test_generate_response_empty_user_text_raises() -> None:
    """Verify generate_response raises ValueError for empty user text."""
    with pytest.raises(ValueError, match="user_text cannot be empty"):
        generate_response("", [])


def test_generate_response_whitespace_only_user_text_raises() -> None:
    """Verify generate_response raises ValueError for whitespace-only user text."""
    with pytest.raises(ValueError, match="user_text cannot be empty"):
        generate_response("   ", [])


def test_generate_response_handles_none_content(
    mock_openai_client: MagicMock,
) -> None:
    """Verify generate_response handles None content from API."""
    mock_openai_client.chat.completions.create.return_value.choices[
        0
    ].message.content = None

    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client):
        result = generate_response("Hello", [])

    assert result.text == ""


def test_generate_response_with_long_history(mock_openai_client: MagicMock) -> None:
    """Verify generate_response handles multiple turns of history."""
    history = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Third question"},
        {"role": "assistant", "content": "Third answer"},
    ]

    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client):
        generate_response("Fourth question", history)

    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]

    # system + 6 history + 1 current = 8 messages
    assert len(messages) == 8


def test_system_prompt_is_defined() -> None:
    """Verify SYSTEM_PROMPT exists and has reasonable content."""
    assert SYSTEM_PROMPT is not None
    assert len(SYSTEM_PROMPT) > 50
    assert (
        "friendly" in SYSTEM_PROMPT.lower() or "conversation" in SYSTEM_PROMPT.lower()
    )


# Tests for Claude backend


def test_generate_response_with_claude_backend(
    mock_anthropic_client: MagicMock,
) -> None:
    """Verify generate_response works with Claude backend."""
    with patch("fai.dialogue.generate.Anthropic", return_value=mock_anthropic_client):
        result = generate_response("Hello", [], backend="claude")

    assert isinstance(result, DialogueResponse)
    assert result.text == "Hello from Claude! How can I assist you?"


def test_generate_response_claude_with_empty_history(
    mock_anthropic_client: MagicMock,
) -> None:
    """Verify Claude backend works with empty history."""
    with patch("fai.dialogue.generate.Anthropic", return_value=mock_anthropic_client):
        result = generate_response("What's the weather?", [], backend="claude")

    assert isinstance(result, DialogueResponse)
    mock_anthropic_client.messages.create.assert_called_once()

    call_args = mock_anthropic_client.messages.create.call_args
    messages = call_args.kwargs["messages"]

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What's the weather?"
    assert call_args.kwargs["system"] == SYSTEM_PROMPT


def test_generate_response_claude_with_history(
    mock_anthropic_client: MagicMock,
) -> None:
    """Verify Claude backend includes history in the API call."""
    history = [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello! Nice to meet you."},
    ]

    with patch("fai.dialogue.generate.Anthropic", return_value=mock_anthropic_client):
        generate_response("How are you?", history, backend="claude")

    call_args = mock_anthropic_client.messages.create.call_args
    messages = call_args.kwargs["messages"]

    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "Hi there"}
    assert messages[1] == {"role": "assistant", "content": "Hello! Nice to meet you."}
    assert messages[2] == {"role": "user", "content": "How are you?"}


def test_generate_response_claude_uses_claude_sonnet_model_by_default(
    mock_anthropic_client: MagicMock,
) -> None:
    """Verify Claude backend uses the claude-sonnet-4 model by default."""
    with patch("fai.dialogue.generate.Anthropic", return_value=mock_anthropic_client):
        generate_response("Test message", [], backend="claude")

    call_args = mock_anthropic_client.messages.create.call_args
    assert call_args.kwargs["model"] == "claude-sonnet-4-20250514"


def test_generate_response_claude_sets_max_tokens(
    mock_anthropic_client: MagicMock,
) -> None:
    """Verify Claude backend sets max_tokens."""
    with patch("fai.dialogue.generate.Anthropic", return_value=mock_anthropic_client):
        generate_response("Test message", [], backend="claude")

    call_args = mock_anthropic_client.messages.create.call_args
    assert call_args.kwargs["max_tokens"] == 1024


def test_generate_response_claude_empty_user_text_raises() -> None:
    """Verify Claude backend raises ValueError for empty user text."""
    with pytest.raises(ValueError, match="user_text cannot be empty"):
        generate_response("", [], backend="claude")


def test_generate_response_claude_whitespace_only_raises() -> None:
    """Verify Claude backend raises ValueError for whitespace-only user text."""
    with pytest.raises(ValueError, match="user_text cannot be empty"):
        generate_response("   ", [], backend="claude")


def test_generate_response_claude_handles_non_text_content(
    mock_anthropic_client: MagicMock,
) -> None:
    """Verify Claude backend handles non-text content blocks."""
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.type = "tool_use"  # non-text content type
    mock_response.content = [mock_content]
    mock_anthropic_client.messages.create.return_value = mock_response

    with patch("fai.dialogue.generate.Anthropic", return_value=mock_anthropic_client):
        result = generate_response("Hello", [], backend="claude")

    assert result.text == ""


def test_generate_response_claude_with_long_history(
    mock_anthropic_client: MagicMock,
) -> None:
    """Verify Claude backend handles multiple turns of history."""
    history = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Third question"},
        {"role": "assistant", "content": "Third answer"},
    ]

    with patch("fai.dialogue.generate.Anthropic", return_value=mock_anthropic_client):
        generate_response("Fourth question", history, backend="claude")

    call_args = mock_anthropic_client.messages.create.call_args
    messages = call_args.kwargs["messages"]

    # 6 history + 1 current = 7 messages
    assert len(messages) == 7


def test_generate_response_invalid_backend_raises() -> None:
    """Verify generate_response raises ValueError for invalid backend."""
    with pytest.raises(ValueError, match="Invalid backend"):
        generate_response("Hello", [], backend="invalid")  # type: ignore[arg-type]


def test_generate_response_default_backend_is_openai(
    mock_openai_client: MagicMock,
) -> None:
    """Verify generate_response defaults to OpenAI backend."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client):
        result = generate_response("Hello", [])

    assert isinstance(result, DialogueResponse)
    mock_openai_client.chat.completions.create.assert_called_once()


# Tests for model selection


def test_model_ids_contains_all_models() -> None:
    """Verify MODEL_IDS contains all expected model mappings."""
    assert "gpt-4o" in MODEL_IDS
    assert "gpt-4o-mini" in MODEL_IDS
    assert "claude-sonnet" in MODEL_IDS
    assert "claude-haiku" in MODEL_IDS
    assert MODEL_IDS["gpt-4o"] == "gpt-4o"
    assert MODEL_IDS["gpt-4o-mini"] == "gpt-4o-mini"
    assert MODEL_IDS["claude-sonnet"] == "claude-sonnet-4-20250514"
    assert MODEL_IDS["claude-haiku"] == "claude-haiku-3-5-20241022"


def test_default_models_for_backends() -> None:
    """Verify DEFAULT_MODELS has correct defaults for each backend."""
    assert DEFAULT_MODELS["openai"] == "gpt-4o"
    assert DEFAULT_MODELS["claude"] == "claude-sonnet"


def test_generate_response_with_gpt4o_mini_model(
    mock_openai_client: MagicMock,
) -> None:
    """Verify generate_response uses gpt-4o-mini when specified."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client):
        generate_response("Test message", [], model="gpt-4o-mini")

    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-4o-mini"


def test_generate_response_with_explicit_gpt4o_model(
    mock_openai_client: MagicMock,
) -> None:
    """Verify generate_response uses gpt-4o when explicitly specified."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client):
        generate_response("Test message", [], model="gpt-4o")

    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-4o"


def test_generate_response_with_claude_haiku_model(
    mock_anthropic_client: MagicMock,
) -> None:
    """Verify generate_response uses claude-haiku when specified."""
    with patch("fai.dialogue.generate.Anthropic", return_value=mock_anthropic_client):
        generate_response("Test message", [], backend="claude", model="claude-haiku")

    call_args = mock_anthropic_client.messages.create.call_args
    assert call_args.kwargs["model"] == "claude-haiku-3-5-20241022"


def test_generate_response_with_explicit_claude_sonnet_model(
    mock_anthropic_client: MagicMock,
) -> None:
    """Verify generate_response uses claude-sonnet when explicitly specified."""
    with patch("fai.dialogue.generate.Anthropic", return_value=mock_anthropic_client):
        generate_response("Test message", [], backend="claude", model="claude-sonnet")

    call_args = mock_anthropic_client.messages.create.call_args
    assert call_args.kwargs["model"] == "claude-sonnet-4-20250514"


def test_generate_response_model_none_uses_default_for_openai(
    mock_openai_client: MagicMock,
) -> None:
    """Verify generate_response uses default model when model is None for OpenAI."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client):
        generate_response("Test message", [], backend="openai", model=None)

    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-4o"


def test_generate_response_model_none_uses_default_for_claude(
    mock_anthropic_client: MagicMock,
) -> None:
    """Verify generate_response uses default model when model is None for Claude."""
    with patch("fai.dialogue.generate.Anthropic", return_value=mock_anthropic_client):
        generate_response("Test message", [], backend="claude", model=None)

    call_args = mock_anthropic_client.messages.create.call_args
    assert call_args.kwargs["model"] == "claude-sonnet-4-20250514"
