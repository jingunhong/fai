"""Tests for the orchestrator component."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fai.orchestrator import run_conversation
from fai.types import AudioData, DialogueResponse, TranscriptResult, VideoFrame


@pytest.fixture  # type: ignore[misc]
def mock_face_path(tmp_path: Path) -> Path:
    """Create a temporary face image file."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()
    return face_path


@pytest.fixture  # type: ignore[misc]
def mock_audio() -> AudioData:
    """Create mock audio data."""
    return AudioData(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)


@pytest.fixture  # type: ignore[misc]
def mock_frame() -> VideoFrame:
    """Create a mock video frame."""
    return VideoFrame(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        timestamp_ms=0,
    )


def test_run_conversation_missing_image_raises() -> None:
    """Verify run_conversation raises FileNotFoundError for missing image."""
    fake_path = Path("/nonexistent/face.jpg")

    with pytest.raises(FileNotFoundError, match="Face image not found"):
        run_conversation(fake_path)


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_text_mode_single_turn(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify text mode conversation flow for a single turn."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi there!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True)

    # Verify each component was called once
    assert mock_generate.call_count == 1
    assert mock_synthesize.call_count == 1
    assert mock_animate.call_count == 1
    assert mock_display.call_count == 1

    # Verify generate_response was called with correct user text
    first_call = mock_generate.call_args_list[0]
    assert first_call[0][0] == "Hello"  # user_text

    # Verify synthesize was called with response text, backend, and voice
    mock_synthesize.assert_called_with("Hi there!", backend="openai", voice=None)


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_maintains_history(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify conversation history is maintained across turns."""
    mock_input.side_effect = ["First message", "Second message", KeyboardInterrupt]
    mock_generate.side_effect = [
        DialogueResponse(text="First response"),
        DialogueResponse(text="Second response"),
    ]
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    # Track history at time of each call
    captured_histories: list[list[dict[str, str]]] = []

    def capture_history(
        user_text: str,
        history: list[dict[str, str]],
        backend: str = "openai",
        model: str | None = None,
    ) -> DialogueResponse:
        captured_histories.append(list(history))  # Copy the history
        if len(captured_histories) == 1:
            return DialogueResponse(text="First response")
        return DialogueResponse(text="Second response")

    mock_generate.side_effect = capture_history

    run_conversation(mock_face_path, text_mode=True)

    # First call should have empty history
    assert captured_histories[0] == []

    # Second call should have first turn in history
    assert len(captured_histories[1]) == 2
    assert captured_histories[1][0] == {"role": "user", "content": "First message"}
    assert captured_histories[1][1] == {
        "role": "assistant",
        "content": "First response",
    }


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_skips_empty_input(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify empty input is skipped."""
    mock_input.side_effect = ["", "   ", "Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True)

    # Should only process "Hello", not empty strings
    assert mock_generate.call_count == 1
    first_call = mock_generate.call_args_list[0]
    assert first_call[0][0] == "Hello"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("fai.orchestrator.loop.transcribe")
@patch("fai.orchestrator.loop.record_audio")
def test_run_conversation_voice_mode(
    mock_record: MagicMock,
    mock_transcribe: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify voice mode uses record_audio and transcribe."""
    mock_record.return_value = mock_audio
    mock_transcribe.side_effect = [
        TranscriptResult(text="Hello from voice"),
        KeyboardInterrupt,
    ]
    mock_generate.return_value = DialogueResponse(text="Voice response")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=False)

    # Verify voice pipeline was used
    # (record is called twice: once for turn, once before interrupt)
    assert mock_record.call_count == 2
    mock_record.assert_called_with(5.0)  # DEFAULT_RECORD_DURATION
    assert mock_transcribe.call_count == 2

    # Verify LLM was called with transcribed text
    assert mock_generate.call_count == 1
    first_call = mock_generate.call_args_list[0]
    assert first_call[0][0] == "Hello from voice"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_handles_keyboard_interrupt(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
) -> None:
    """Verify KeyboardInterrupt exits gracefully."""
    mock_input.side_effect = KeyboardInterrupt

    # Should not raise, should exit gracefully
    run_conversation(mock_face_path, text_mode=True)

    mock_generate.assert_not_called()


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_handles_eof(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify EOFError from input is handled."""
    mock_input.side_effect = [EOFError, "Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True)

    # EOF returns empty string, which is skipped, then "Hello" is processed
    assert mock_generate.call_count == 1
    first_call = mock_generate.call_args_list[0]
    assert first_call[0][0] == "Hello"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_passes_backend_to_animate(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify backend parameter is passed to animate function."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True, backend="wav2lip")

    # Verify animate was called with backend parameter
    mock_animate.assert_called_once()
    call_kwargs = mock_animate.call_args[1]
    assert call_kwargs.get("backend") == "wav2lip"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_default_backend_is_auto(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify default backend is 'auto'."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True)

    # Verify animate was called with backend="auto" (default)
    mock_animate.assert_called_once()
    call_kwargs = mock_animate.call_args[1]
    assert call_kwargs.get("backend") == "auto"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_passes_dialogue_backend_to_generate_response(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify dialogue_backend parameter is passed to generate_response."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True, dialogue_backend="claude")

    # Verify generate_response was called with backend parameter
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs.get("backend") == "claude"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_default_dialogue_backend_is_openai(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify default dialogue_backend is 'openai'."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True)

    # Verify generate_response was called with backend="openai" (default)
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs.get("backend") == "openai"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_passes_tts_backend_to_synthesize(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify tts_backend parameter is passed to synthesize."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True, tts_backend="elevenlabs")

    # Verify synthesize was called with backend parameter
    mock_synthesize.assert_called_once()
    call_kwargs = mock_synthesize.call_args[1]
    assert call_kwargs.get("backend") == "elevenlabs"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_default_tts_backend_is_openai(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify default tts_backend is 'openai'."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True)

    # Verify synthesize was called with backend="openai" (default)
    mock_synthesize.assert_called_once()
    call_kwargs = mock_synthesize.call_args[1]
    assert call_kwargs.get("backend") == "openai"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_passes_voice_to_synthesize(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify voice parameter is passed to synthesize."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True, voice="echo")

    # Verify synthesize was called with voice parameter
    mock_synthesize.assert_called_once()
    call_kwargs = mock_synthesize.call_args[1]
    assert call_kwargs.get("voice") == "echo"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_default_voice_is_none(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify default voice is None."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True)

    # Verify synthesize was called with voice=None (default)
    mock_synthesize.assert_called_once()
    call_kwargs = mock_synthesize.call_args[1]
    assert call_kwargs.get("voice") is None


# Tests for model parameter


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_passes_model_to_generate_response(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify model parameter is passed to generate_response."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True, model="gpt-4o-mini")

    # Verify generate_response was called with model parameter
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs.get("model") == "gpt-4o-mini"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_default_model_is_none(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify default model is None."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(mock_face_path, text_mode=True)

    # Verify generate_response was called with model=None (default)
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs.get("model") is None


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_passes_claude_model_to_generate_response(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    mock_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify Claude model parameter is passed to generate_response."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = mock_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation(
        mock_face_path,
        text_mode=True,
        dialogue_backend="claude",
        model="claude-haiku",
    )

    # Verify generate_response was called with correct parameters
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs.get("backend") == "claude"
    assert call_kwargs.get("model") == "claude-haiku"
