"""Tests for the CLI module."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fai import cli
from fai.validation import ValidationResult


@pytest.fixture  # type: ignore[misc]
def valid_api_keys() -> ValidationResult:
    """Return a valid API keys validation result."""
    return ValidationResult(
        is_valid=True,
        missing_keys=[],
        error_message="",
    )


def test_main_exists() -> None:
    """Verify main function is importable."""
    assert callable(cli.main)


def test_cli_list_backends_no_available(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --list-backends shows message when none available."""
    with (
        patch("sys.argv", ["fai", "--list-backends"]),
        patch("fai.cli.get_available_backends", return_value=[]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "No lip-sync backends available" in captured.out


def test_cli_list_backends_with_available(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --list-backends shows available backends."""
    with (
        patch("sys.argv", ["fai", "--list-backends"]),
        patch("fai.cli.get_available_backends", return_value=["wav2lip", "sadtalker"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "wav2lip" in captured.out
    assert "sadtalker" in captured.out


def test_cli_missing_face_image(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify error when face image doesn't exist."""
    with (
        patch("sys.argv", ["fai", "/nonexistent/face.jpg"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Face image not found" in captured.err


def test_cli_passes_backend_to_run_conversation(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --backend is passed to run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--backend", "none"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once_with(
        face_path,
        text_mode=False,
        backend="none",
        dialogue_backend="openai",
        tts_backend="openai",
        voice=None,
        record=False,
        output_dir=None,
        model=None,
    )


def test_cli_passes_text_mode_to_run_conversation(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --text flag is passed to run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--text"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once_with(
        face_path,
        text_mode=True,
        backend="auto",
        dialogue_backend="openai",
        tts_backend="openai",
        voice=None,
        record=False,
        output_dir=None,
        model=None,
    )


def test_cli_default_backend_is_auto(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify default backend is 'auto'."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("backend") == "auto"


def test_cli_passes_dialogue_backend_to_run_conversation(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --dialogue is passed to run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--dialogue", "claude"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once_with(
        face_path,
        text_mode=False,
        backend="auto",
        dialogue_backend="claude",
        tts_backend="openai",
        voice=None,
        record=False,
        output_dir=None,
        model=None,
    )


def test_cli_default_dialogue_backend_is_openai(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify default dialogue backend is 'openai'."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("dialogue_backend") == "openai"


def test_cli_passes_tts_backend_to_run_conversation(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --tts is passed to run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--tts", "elevenlabs"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once_with(
        face_path,
        text_mode=False,
        backend="auto",
        dialogue_backend="openai",
        tts_backend="elevenlabs",
        voice=None,
        record=False,
        output_dir=None,
        model=None,
    )


def test_cli_default_tts_backend_is_openai(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify default TTS backend is 'openai'."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("tts_backend") == "openai"


def test_cli_passes_record_flag_to_run_conversation(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --record flag is passed to run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--record"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("record") is True


def test_cli_passes_output_dir_to_run_conversation(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --output-dir is passed to run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()
    output_dir = tmp_path / "my_recordings"

    with (
        patch("sys.argv", ["fai", str(face_path), "--output-dir", str(output_dir)]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("output_dir") == output_dir


def test_cli_default_record_is_false(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify default record is False."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("record") is False


def test_cli_default_output_dir_is_none(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify default output_dir is None."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("output_dir") is None


def test_cli_passes_voice_to_run_conversation(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --voice is passed to run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--voice", "echo"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once_with(
        face_path,
        text_mode=False,
        backend="auto",
        dialogue_backend="openai",
        tts_backend="openai",
        voice="echo",
        record=False,
        output_dir=None,
        model=None,
    )


def test_cli_default_voice_is_none(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify default voice is None."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("voice") is None


def test_cli_list_voices_openai(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --list-voices shows OpenAI voices."""
    with (
        patch("sys.argv", ["fai", "--list-voices"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Available voices for openai TTS" in captured.out
    assert "alloy" in captured.out
    assert "echo" in captured.out


def test_cli_list_voices_elevenlabs(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --list-voices with --tts elevenlabs shows ElevenLabs voices."""
    with (
        patch("sys.argv", ["fai", "--list-voices", "--tts", "elevenlabs"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Available voices for elevenlabs TTS" in captured.out
    assert "rachel" in captured.out
    assert "adam" in captured.out


def test_cli_voice_with_tts_backend(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --voice works with --tts backend."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch(
            "sys.argv",
            ["fai", str(face_path), "--tts", "elevenlabs", "--voice", "josh"],
        ),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("tts_backend") == "elevenlabs"
    assert kwargs.get("voice") == "josh"


def test_cli_log_level_default_is_warning(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify default log level is 'warning'."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation"),
        patch("fai.cli.setup_logging") as mock_setup,
    ):
        cli.main()

    mock_setup.assert_called_once_with(level="warning")


def test_cli_log_level_debug(tmp_path: Path, valid_api_keys: ValidationResult) -> None:
    """Verify --log-level debug is passed to setup_logging."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--log-level", "debug"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation"),
        patch("fai.cli.setup_logging") as mock_setup,
    ):
        cli.main()

    mock_setup.assert_called_once_with(level="debug")


def test_cli_log_level_info(tmp_path: Path, valid_api_keys: ValidationResult) -> None:
    """Verify --log-level info is passed to setup_logging."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--log-level", "info"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation"),
        patch("fai.cli.setup_logging") as mock_setup,
    ):
        cli.main()

    mock_setup.assert_called_once_with(level="info")


def test_cli_log_level_error(tmp_path: Path, valid_api_keys: ValidationResult) -> None:
    """Verify --log-level error is passed to setup_logging."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--log-level", "error"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation"),
        patch("fai.cli.setup_logging") as mock_setup,
    ):
        cli.main()

    mock_setup.assert_called_once_with(level="error")


def test_cli_log_level_critical(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --log-level critical is passed to setup_logging."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--log-level", "critical"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation"),
        patch("fai.cli.setup_logging") as mock_setup,
    ):
        cli.main()

    mock_setup.assert_called_once_with(level="critical")


def test_cli_log_level_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify invalid --log-level value is rejected by argparse."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--log-level", "invalid"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_cli_log_level_called_before_other_operations(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify setup_logging is called before run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    call_order: list[str] = []

    def mock_setup_logging(level: str) -> None:
        call_order.append("setup_logging")

    def mock_run_conversation(*args: Any, **kwargs: Any) -> None:
        call_order.append("run_conversation")

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.setup_logging", side_effect=mock_setup_logging),
        patch("fai.cli.run_conversation", side_effect=mock_run_conversation),
    ):
        cli.main()

    assert call_order == ["setup_logging", "run_conversation"]


def test_cli_log_level_with_stream_mode(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --log-level works with --stream mode."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--stream", "--log-level", "debug"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation_stream"),
        patch("fai.cli.setup_logging") as mock_setup,
    ):
        cli.main()

    mock_setup.assert_called_once_with(level="debug")


def test_cli_api_key_validation_called(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify validate_api_keys is called with correct arguments."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch(
            "sys.argv",
            ["fai", str(face_path), "--dialogue", "claude", "--tts", "elevenlabs"],
        ),
        patch(
            "fai.cli.validate_api_keys", return_value=valid_api_keys
        ) as mock_validate,
        patch("fai.cli.run_conversation"),
    ):
        cli.main()

    mock_validate.assert_called_once_with(
        dialogue_backend="claude",
        tts_backend="elevenlabs",
    )


def test_cli_api_key_validation_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI exits with error when API key validation fails."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    invalid_result = ValidationResult(
        is_valid=False,
        missing_keys=["OPENAI_API_KEY"],
        error_message="Missing required API key(s):\n  • OPENAI_API_KEY",
    )

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.validate_api_keys", return_value=invalid_result),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Missing required API key(s)" in captured.err
    assert "OPENAI_API_KEY" in captured.err


def test_cli_api_key_validation_default_backends(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify validate_api_keys is called with default backends."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch(
            "fai.cli.validate_api_keys", return_value=valid_api_keys
        ) as mock_validate,
        patch("fai.cli.run_conversation"),
    ):
        cli.main()

    mock_validate.assert_called_once_with(
        dialogue_backend="openai",
        tts_backend="openai",
    )


def test_cli_api_key_validation_stream_mode(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify validate_api_keys is called in stream mode."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--stream", "--dialogue", "claude"]),
        patch(
            "fai.cli.validate_api_keys", return_value=valid_api_keys
        ) as mock_validate,
        patch("fai.cli.run_conversation_stream"),
    ):
        cli.main()

    mock_validate.assert_called_once_with(
        dialogue_backend="claude",
        tts_backend="openai",
    )


# Tests for --model flag


def test_cli_passes_model_to_run_conversation(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --model is passed to run_conversation."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--model", "gpt-4o-mini"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once_with(
        face_path,
        text_mode=False,
        backend="auto",
        dialogue_backend="openai",
        tts_backend="openai",
        voice=None,
        record=False,
        output_dir=None,
        model="gpt-4o-mini",
    )


def test_cli_default_model_is_none(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify default model is None."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path)]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("model") is None


def test_cli_passes_model_to_run_conversation_stream(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify --model is passed to run_conversation_stream."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch(
            "sys.argv",
            [
                "fai",
                str(face_path),
                "--stream",
                "--dialogue",
                "claude",
                "--model",
                "claude-haiku",
            ],
        ),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation_stream") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once_with(
        face_path,
        text_mode=False,
        dialogue_backend="claude",
        tts_backend="openai",
        voice=None,
        model="claude-haiku",
    )


def test_cli_model_gpt4o_with_openai_backend(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify gpt-4o model works with openai backend."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--model", "gpt-4o"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("model") == "gpt-4o"


def test_cli_model_claude_sonnet_with_claude_backend(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify claude-sonnet model works with claude backend."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch(
            "sys.argv",
            ["fai", str(face_path), "--dialogue", "claude", "--model", "claude-sonnet"],
        ),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("model") == "claude-sonnet"
    assert kwargs.get("dialogue_backend") == "claude"


def test_cli_model_claude_with_openai_backend_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], valid_api_keys: ValidationResult
) -> None:
    """Verify Claude model with OpenAI backend exits with error."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--model", "claude-sonnet"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "not compatible with OpenAI backend" in captured.err


def test_cli_model_openai_with_claude_backend_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], valid_api_keys: ValidationResult
) -> None:
    """Verify OpenAI model with Claude backend exits with error."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch(
            "sys.argv",
            ["fai", str(face_path), "--dialogue", "claude", "--model", "gpt-4o"],
        ),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "not compatible with Claude backend" in captured.err


def test_cli_model_invalid_choice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify invalid model choice is rejected by argparse."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--model", "invalid-model"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_cli_model_gpt4o_mini_with_openai_backend(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify gpt-4o-mini model works with openai backend."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch("sys.argv", ["fai", str(face_path), "--model", "gpt-4o-mini"]),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("model") == "gpt-4o-mini"


def test_cli_model_claude_haiku_with_claude_backend(
    tmp_path: Path, valid_api_keys: ValidationResult
) -> None:
    """Verify claude-haiku model works with claude backend."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    with (
        patch(
            "sys.argv",
            ["fai", str(face_path), "--dialogue", "claude", "--model", "claude-haiku"],
        ),
        patch("fai.cli.validate_api_keys", return_value=valid_api_keys),
        patch("fai.cli.run_conversation") as mock_run,
    ):
        cli.main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("model") == "claude-haiku"
    assert kwargs.get("dialogue_backend") == "claude"
