"""Tests for the perception component."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fai.perception import record_audio, transcribe
from fai.types import AudioData, TranscriptResult


@pytest.fixture  # type: ignore[misc]
def sample_audio() -> AudioData:
    """Create sample audio data for testing."""
    # 1 second of silence at 16kHz
    samples = np.zeros(16000, dtype=np.float32)
    return AudioData(samples=samples, sample_rate=16000)


@pytest.fixture  # type: ignore[misc]
def mock_whisper_response() -> MagicMock:
    """Create a mock Whisper API response."""
    mock_response = MagicMock()
    mock_response.text = "Hello, how are you?"
    return mock_response


@pytest.fixture  # type: ignore[misc]
def mock_openai_client(mock_whisper_response: MagicMock) -> MagicMock:
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_whisper_response
    return mock_client


def test_transcribe_returns_transcript_result(
    mock_openai_client: MagicMock, sample_audio: AudioData
) -> None:
    """Verify transcribe returns a TranscriptResult."""
    with patch("fai.perception.transcribe.OpenAI", return_value=mock_openai_client):
        result = transcribe(sample_audio)

    assert isinstance(result, TranscriptResult)
    assert result.text == "Hello, how are you?"


def test_transcribe_calls_whisper_api(
    mock_openai_client: MagicMock, sample_audio: AudioData
) -> None:
    """Verify transcribe calls the Whisper API."""
    with patch("fai.perception.transcribe.OpenAI", return_value=mock_openai_client):
        transcribe(sample_audio)

    mock_openai_client.audio.transcriptions.create.assert_called_once()
    call_args = mock_openai_client.audio.transcriptions.create.call_args
    assert call_args.kwargs["model"] == "whisper-1"


def test_transcribe_sends_wav_file(
    mock_openai_client: MagicMock, sample_audio: AudioData
) -> None:
    """Verify transcribe sends audio as a WAV file."""
    with patch("fai.perception.transcribe.OpenAI", return_value=mock_openai_client):
        transcribe(sample_audio)

    call_args = mock_openai_client.audio.transcriptions.create.call_args
    file_arg = call_args.kwargs["file"]
    assert hasattr(file_arg, "name")
    assert file_arg.name == "audio.wav"


def test_transcribe_empty_audio_raises() -> None:
    """Verify transcribe raises ValueError for empty audio."""
    empty_audio = AudioData(samples=np.array([], dtype=np.float32), sample_rate=16000)

    with pytest.raises(ValueError, match="audio samples cannot be empty"):
        transcribe(empty_audio)


def test_record_audio_returns_audio_data() -> None:
    """Verify record_audio returns an AudioData object."""
    mock_samples = np.zeros((16000, 1), dtype=np.float32)

    with (
        patch("fai.perception.record.sd.rec", return_value=mock_samples) as mock_rec,
        patch("fai.perception.record.sd.wait"),
    ):
        result = record_audio(1.0)

    assert isinstance(result, AudioData)
    assert result.sample_rate == 16000
    mock_rec.assert_called_once()


def test_record_audio_uses_correct_parameters() -> None:
    """Verify record_audio calls sounddevice with correct parameters."""
    mock_samples = np.zeros((32000, 1), dtype=np.float32)

    with (
        patch("fai.perception.record.sd.rec", return_value=mock_samples) as mock_rec,
        patch("fai.perception.record.sd.wait"),
    ):
        record_audio(2.0)

    mock_rec.assert_called_once_with(
        32000,  # 2 seconds * 16kHz
        samplerate=16000,
        channels=1,
        dtype=np.float32,
    )


def test_record_audio_waits_for_completion() -> None:
    """Verify record_audio waits for recording to complete."""
    mock_samples = np.zeros((16000, 1), dtype=np.float32)

    with (
        patch("fai.perception.record.sd.rec", return_value=mock_samples),
        patch("fai.perception.record.sd.wait") as mock_wait,
    ):
        record_audio(1.0)

    mock_wait.assert_called_once()


def test_record_audio_flattens_samples() -> None:
    """Verify record_audio returns flattened 1D array."""
    # sounddevice returns (n_samples, n_channels) shape
    mock_samples = np.zeros((16000, 1), dtype=np.float32)

    with (
        patch("fai.perception.record.sd.rec", return_value=mock_samples),
        patch("fai.perception.record.sd.wait"),
    ):
        result = record_audio(1.0)

    assert result.samples.ndim == 1
    assert len(result.samples) == 16000


def test_record_audio_zero_duration_raises() -> None:
    """Verify record_audio raises ValueError for zero duration."""
    with pytest.raises(ValueError, match="duration_seconds must be positive"):
        record_audio(0.0)


def test_record_audio_negative_duration_raises() -> None:
    """Verify record_audio raises ValueError for negative duration."""
    with pytest.raises(ValueError, match="duration_seconds must be positive"):
        record_audio(-1.0)


def test_transcribe_with_different_sample_rate(
    mock_openai_client: MagicMock,
) -> None:
    """Verify transcribe works with different sample rates."""
    # Audio at 44.1kHz sample rate
    samples = np.zeros(44100, dtype=np.float32)
    audio = AudioData(samples=samples, sample_rate=44100)

    with patch("fai.perception.transcribe.OpenAI", return_value=mock_openai_client):
        result = transcribe(audio)

    assert isinstance(result, TranscriptResult)


def test_record_audio_with_fractional_duration() -> None:
    """Verify record_audio handles fractional durations."""
    # 0.5 seconds = 8000 samples at 16kHz
    mock_samples = np.zeros((8000, 1), dtype=np.float32)

    with (
        patch("fai.perception.record.sd.rec", return_value=mock_samples) as mock_rec,
        patch("fai.perception.record.sd.wait"),
    ):
        record_audio(0.5)

    mock_rec.assert_called_once()
    call_args = mock_rec.call_args
    assert call_args[0][0] == 8000  # Expected sample count


# Tests for timeout parameter


def test_transcribe_passes_timeout_to_openai_client(
    mock_openai_client: MagicMock, sample_audio: AudioData
) -> None:
    """Verify transcribe passes timeout to OpenAI client constructor."""
    with patch(
        "fai.perception.transcribe.OpenAI", return_value=mock_openai_client
    ) as mock_cls:
        transcribe(sample_audio, timeout=30.0)

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs.get("timeout") == 30.0


def test_transcribe_no_timeout_omits_from_client(
    mock_openai_client: MagicMock, sample_audio: AudioData
) -> None:
    """Verify transcribe omits timeout when None."""
    with patch(
        "fai.perception.transcribe.OpenAI", return_value=mock_openai_client
    ) as mock_cls:
        transcribe(sample_audio, timeout=None)

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert "timeout" not in call_kwargs


def test_transcribe_default_timeout_is_none(
    mock_openai_client: MagicMock, sample_audio: AudioData
) -> None:
    """Verify transcribe default timeout is None (omitted from client)."""
    with patch(
        "fai.perception.transcribe.OpenAI", return_value=mock_openai_client
    ) as mock_cls:
        transcribe(sample_audio)

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert "timeout" not in call_kwargs
