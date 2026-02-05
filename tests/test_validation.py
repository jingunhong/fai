"""Tests for the API key validation module."""

from unittest.mock import patch

from fai.validation import (
    ValidationResult,
    _build_error_message,
    _get_key_description,
    validate_api_keys,
)


def test_validation_result_dataclass() -> None:
    """Verify ValidationResult dataclass fields."""
    result = ValidationResult(
        is_valid=True,
        missing_keys=[],
        error_message="",
    )
    assert result.is_valid is True
    assert result.missing_keys == []
    assert result.error_message == ""


def test_validation_result_with_missing_keys() -> None:
    """Verify ValidationResult with missing keys."""
    result = ValidationResult(
        is_valid=False,
        missing_keys=["OPENAI_API_KEY"],
        error_message="Some error",
    )
    assert result.is_valid is False
    assert result.missing_keys == ["OPENAI_API_KEY"]
    assert result.error_message == "Some error"


def test_validate_api_keys_all_present() -> None:
    """Verify validation passes when all required keys are present."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test-key",
        },
        clear=True,
    ):
        result = validate_api_keys(dialogue_backend="openai", tts_backend="openai")

    assert result.is_valid is True
    assert result.missing_keys == []
    assert result.error_message == ""


def test_validate_api_keys_missing_openai() -> None:
    """Verify validation fails when OPENAI_API_KEY is missing."""
    with patch.dict("os.environ", {}, clear=True):
        result = validate_api_keys(dialogue_backend="openai", tts_backend="openai")

    assert result.is_valid is False
    assert "OPENAI_API_KEY" in result.missing_keys
    assert "OPENAI_API_KEY" in result.error_message


def test_validate_api_keys_empty_openai() -> None:
    """Verify validation fails when OPENAI_API_KEY is empty."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True):
        result = validate_api_keys(dialogue_backend="openai", tts_backend="openai")

    assert result.is_valid is False
    assert "OPENAI_API_KEY" in result.missing_keys


def test_validate_api_keys_whitespace_openai() -> None:
    """Verify validation fails when OPENAI_API_KEY is whitespace only."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "   "}, clear=True):
        result = validate_api_keys(dialogue_backend="openai", tts_backend="openai")

    assert result.is_valid is False
    assert "OPENAI_API_KEY" in result.missing_keys


def test_validate_api_keys_claude_backend_all_present() -> None:
    """Verify validation passes for Claude backend with all keys present."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test-key",
            "ANTHROPIC_API_KEY": "sk-ant-test-key",
        },
        clear=True,
    ):
        result = validate_api_keys(dialogue_backend="claude", tts_backend="openai")

    assert result.is_valid is True
    assert result.missing_keys == []


def test_validate_api_keys_claude_backend_missing_anthropic() -> None:
    """Verify validation fails for Claude backend when ANTHROPIC_API_KEY is missing."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test-key",
        },
        clear=True,
    ):
        result = validate_api_keys(dialogue_backend="claude", tts_backend="openai")

    assert result.is_valid is False
    assert "ANTHROPIC_API_KEY" in result.missing_keys
    assert "OPENAI_API_KEY" not in result.missing_keys


def test_validate_api_keys_claude_backend_empty_anthropic() -> None:
    """Verify validation fails for Claude backend when ANTHROPIC_API_KEY is empty."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test-key",
            "ANTHROPIC_API_KEY": "",
        },
        clear=True,
    ):
        result = validate_api_keys(dialogue_backend="claude", tts_backend="openai")

    assert result.is_valid is False
    assert "ANTHROPIC_API_KEY" in result.missing_keys


def test_validate_api_keys_elevenlabs_tts_all_present() -> None:
    """Verify validation passes for ElevenLabs TTS with all keys present."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test-key",
            "ELEVENLABS_API_KEY": "el-test-key",
        },
        clear=True,
    ):
        result = validate_api_keys(dialogue_backend="openai", tts_backend="elevenlabs")

    assert result.is_valid is True
    assert result.missing_keys == []


def test_validate_api_keys_elevenlabs_tts_missing_elevenlabs() -> None:
    """Verify validation fails for ElevenLabs TTS when ELEVENLABS_API_KEY is missing."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test-key",
        },
        clear=True,
    ):
        result = validate_api_keys(dialogue_backend="openai", tts_backend="elevenlabs")

    assert result.is_valid is False
    assert "ELEVENLABS_API_KEY" in result.missing_keys
    assert "OPENAI_API_KEY" not in result.missing_keys


def test_validate_api_keys_elevenlabs_tts_empty_elevenlabs() -> None:
    """Verify validation fails for ElevenLabs TTS when ELEVENLABS_API_KEY is empty."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test-key",
            "ELEVENLABS_API_KEY": "",
        },
        clear=True,
    ):
        result = validate_api_keys(dialogue_backend="openai", tts_backend="elevenlabs")

    assert result.is_valid is False
    assert "ELEVENLABS_API_KEY" in result.missing_keys


def test_validate_api_keys_multiple_missing() -> None:
    """Verify validation fails with multiple missing keys."""
    with patch.dict("os.environ", {}, clear=True):
        result = validate_api_keys(dialogue_backend="claude", tts_backend="elevenlabs")

    assert result.is_valid is False
    assert "OPENAI_API_KEY" in result.missing_keys
    assert "ANTHROPIC_API_KEY" in result.missing_keys
    assert "ELEVENLABS_API_KEY" in result.missing_keys
    assert len(result.missing_keys) == 3


def test_validate_api_keys_all_backends_all_keys_present() -> None:
    """Verify validation passes with all backends and all keys."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test-key",
            "ANTHROPIC_API_KEY": "sk-ant-test-key",
            "ELEVENLABS_API_KEY": "el-test-key",
        },
        clear=True,
    ):
        result = validate_api_keys(dialogue_backend="claude", tts_backend="elevenlabs")

    assert result.is_valid is True
    assert result.missing_keys == []


def test_validate_api_keys_default_backends() -> None:
    """Verify validation with default backends only requires OPENAI_API_KEY."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test-key",
        },
        clear=True,
    ):
        result = validate_api_keys()

    assert result.is_valid is True
    assert result.missing_keys == []


def test_build_error_message_single_key() -> None:
    """Verify error message format for single missing key."""
    message = _build_error_message(["OPENAI_API_KEY"])

    assert "Missing required API key(s):" in message
    assert "OPENAI_API_KEY" in message
    assert "Required for speech transcription" in message
    assert ".env" in message


def test_build_error_message_multiple_keys() -> None:
    """Verify error message format for multiple missing keys."""
    message = _build_error_message(
        ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY"]
    )

    assert "Missing required API key(s):" in message
    assert "OPENAI_API_KEY" in message
    assert "ANTHROPIC_API_KEY" in message
    assert "ELEVENLABS_API_KEY" in message


def test_get_key_description_openai() -> None:
    """Verify description for OPENAI_API_KEY."""
    desc = _get_key_description("OPENAI_API_KEY")
    assert "Whisper" in desc or "transcription" in desc


def test_get_key_description_anthropic() -> None:
    """Verify description for ANTHROPIC_API_KEY."""
    desc = _get_key_description("ANTHROPIC_API_KEY")
    assert "Claude" in desc


def test_get_key_description_elevenlabs() -> None:
    """Verify description for ELEVENLABS_API_KEY."""
    desc = _get_key_description("ELEVENLABS_API_KEY")
    assert "ElevenLabs" in desc


def test_get_key_description_unknown() -> None:
    """Verify description for unknown key."""
    desc = _get_key_description("UNKNOWN_API_KEY")
    assert "Required" in desc
