"""Tests for the voice component."""

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fai.types import AudioData
from fai.voice import play_audio, stop_audio, synthesize


def _create_mock_wav_bytes(
    sample_rate: int = 24000,
    duration_seconds: float = 0.1,
    sample_width: int = 2,
) -> bytes:
    """Create mock WAV file bytes for testing."""
    n_samples = int(sample_rate * duration_seconds)

    if sample_width == 2:
        # 16-bit signed samples
        samples = np.sin(2 * np.pi * 440 * np.arange(n_samples) / sample_rate) * 16000
        raw_data = samples.astype(np.int16).tobytes()
    else:
        # 8-bit unsigned samples
        samples = (
            np.sin(2 * np.pi * 440 * np.arange(n_samples) / sample_rate) * 64 + 128
        )
        raw_data = samples.astype(np.uint8).tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_data)

    return buffer.getvalue()


@pytest.fixture  # type: ignore[misc]
def mock_wav_response() -> MagicMock:
    """Create a mock OpenAI TTS response with WAV data."""
    mock_response = MagicMock()
    mock_response.content = _create_mock_wav_bytes()
    return mock_response


@pytest.fixture  # type: ignore[misc]
def mock_openai_client(mock_wav_response: MagicMock) -> MagicMock:
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = mock_wav_response
    return mock_client


def test_synthesize_returns_audio_data(mock_openai_client: MagicMock) -> None:
    """Verify synthesize returns an AudioData object."""
    with patch("fai.voice.synthesize.OpenAI", return_value=mock_openai_client):
        result = synthesize("Hello world")

    assert isinstance(result, AudioData)
    assert isinstance(result.samples, np.ndarray)
    assert result.samples.dtype == np.float32
    assert result.sample_rate == 24000


def test_synthesize_calls_openai_api(mock_openai_client: MagicMock) -> None:
    """Verify synthesize calls the OpenAI TTS API with correct parameters."""
    with patch("fai.voice.synthesize.OpenAI", return_value=mock_openai_client):
        synthesize("Test message")

    mock_openai_client.audio.speech.create.assert_called_once_with(
        model="tts-1",
        voice="alloy",
        input="Test message",
        response_format="wav",
    )


def test_synthesize_empty_text_raises() -> None:
    """Verify synthesize raises ValueError for empty text."""
    with pytest.raises(ValueError, match="text cannot be empty"):
        synthesize("")


def test_synthesize_whitespace_only_text_raises() -> None:
    """Verify synthesize raises ValueError for whitespace-only text."""
    with pytest.raises(ValueError, match="text cannot be empty"):
        synthesize("   ")


def test_synthesize_samples_are_normalized(mock_openai_client: MagicMock) -> None:
    """Verify synthesized samples are normalized to [-1, 1] range."""
    with patch("fai.voice.synthesize.OpenAI", return_value=mock_openai_client):
        result = synthesize("Test")

    assert result.samples.min() >= -1.0
    assert result.samples.max() <= 1.0


def test_synthesize_handles_8bit_audio() -> None:
    """Verify synthesize handles 8-bit audio correctly."""
    mock_response = MagicMock()
    mock_response.content = _create_mock_wav_bytes(sample_width=1)

    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = mock_response

    with patch("fai.voice.synthesize.OpenAI", return_value=mock_client):
        result = synthesize("Test")

    assert isinstance(result, AudioData)
    assert result.samples.dtype == np.float32
    assert result.samples.min() >= -1.0
    assert result.samples.max() <= 1.0


def test_synthesize_with_long_text(mock_openai_client: MagicMock) -> None:
    """Verify synthesize works with longer text."""
    long_text = "This is a longer piece of text that would generate more audio."

    with patch("fai.voice.synthesize.OpenAI", return_value=mock_openai_client):
        result = synthesize(long_text)

    assert isinstance(result, AudioData)
    mock_openai_client.audio.speech.create.assert_called_once()
    call_args = mock_openai_client.audio.speech.create.call_args
    assert call_args.kwargs["input"] == long_text


def test_synthesize_preserves_sample_rate(mock_openai_client: MagicMock) -> None:
    """Verify synthesize preserves the sample rate from the WAV file."""
    # Create WAV with specific sample rate
    mock_response = MagicMock()
    mock_response.content = _create_mock_wav_bytes(sample_rate=16000)
    mock_openai_client.audio.speech.create.return_value = mock_response

    with patch("fai.voice.synthesize.OpenAI", return_value=mock_openai_client):
        result = synthesize("Test")

    assert result.sample_rate == 16000


# =============================================================================
# Tests for play_audio
# =============================================================================


@pytest.fixture  # type: ignore[misc]
def sample_audio() -> AudioData:
    """Create sample audio data for testing."""
    samples = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    return AudioData(samples=samples, sample_rate=16000)


def test_play_audio_calls_sounddevice_play(sample_audio: AudioData) -> None:
    """Verify play_audio calls sd.play with correct parameters."""
    with patch("fai.voice.playback.sd") as mock_sd:
        play_audio(sample_audio, blocking=False)

    mock_sd.play.assert_called_once()
    call_args = mock_sd.play.call_args
    np.testing.assert_array_equal(call_args[0][0], sample_audio.samples)
    assert call_args[1]["samplerate"] == sample_audio.sample_rate


def test_play_audio_blocking_waits(sample_audio: AudioData) -> None:
    """Verify play_audio with blocking=True calls sd.wait."""
    with patch("fai.voice.playback.sd") as mock_sd:
        play_audio(sample_audio, blocking=True)

    mock_sd.play.assert_called_once()
    mock_sd.wait.assert_called_once()


def test_play_audio_non_blocking_does_not_wait(sample_audio: AudioData) -> None:
    """Verify play_audio with blocking=False does not call sd.wait."""
    with patch("fai.voice.playback.sd") as mock_sd:
        play_audio(sample_audio, blocking=False)

    mock_sd.play.assert_called_once()
    mock_sd.wait.assert_not_called()


def test_play_audio_empty_samples_raises() -> None:
    """Verify play_audio raises ValueError for empty samples."""
    empty_audio = AudioData(samples=np.array([], dtype=np.float32), sample_rate=16000)

    with pytest.raises(ValueError, match="audio samples cannot be empty"):
        play_audio(empty_audio)


def test_play_audio_default_blocking_is_true(sample_audio: AudioData) -> None:
    """Verify play_audio defaults to blocking=True."""
    with patch("fai.voice.playback.sd") as mock_sd:
        play_audio(sample_audio)

    mock_sd.wait.assert_called_once()


def test_play_audio_with_different_sample_rates() -> None:
    """Verify play_audio respects the sample rate from AudioData."""
    samples = np.zeros(1000, dtype=np.float32)

    for sample_rate in [8000, 16000, 24000, 44100, 48000]:
        audio = AudioData(samples=samples, sample_rate=sample_rate)

        with patch("fai.voice.playback.sd") as mock_sd:
            play_audio(audio, blocking=False)

        assert mock_sd.play.call_args[1]["samplerate"] == sample_rate


# =============================================================================
# Tests for stop_audio
# =============================================================================


def test_stop_audio_calls_sounddevice_stop() -> None:
    """Verify stop_audio calls sd.stop."""
    with patch("fai.voice.playback.sd") as mock_sd:
        stop_audio()

    mock_sd.stop.assert_called_once()
