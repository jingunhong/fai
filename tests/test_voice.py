"""Tests for the voice component."""

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fai.types import AudioData
from fai.voice import (
    ElevenLabsVoice,
    OpenAIVoice,
    TTSBackend,
    get_available_voices,
    play_audio,
    stop_audio,
    synthesize,
)
from tests.helpers import create_mock_pcm_bytes


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


@pytest.fixture  # type: ignore[misc]
def mock_elevenlabs_client() -> MagicMock:
    """Create a mock ElevenLabs client."""
    mock_client = MagicMock()
    # Return a generator-like iterator of PCM audio chunks
    mock_client.text_to_speech.convert.return_value = iter(
        [create_mock_pcm_bytes(22050, 0.05), create_mock_pcm_bytes(22050, 0.05)]
    )
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


def test_synthesize_default_backend_is_openai(mock_openai_client: MagicMock) -> None:
    """Verify synthesize defaults to OpenAI backend."""
    with patch("fai.voice.synthesize.OpenAI", return_value=mock_openai_client):
        synthesize("Test")

    mock_openai_client.audio.speech.create.assert_called_once()


def test_synthesize_explicit_openai_backend(mock_openai_client: MagicMock) -> None:
    """Verify synthesize with explicit openai backend uses OpenAI."""
    with patch("fai.voice.synthesize.OpenAI", return_value=mock_openai_client):
        result = synthesize("Test", backend="openai")

    assert isinstance(result, AudioData)
    mock_openai_client.audio.speech.create.assert_called_once()


def test_synthesize_invalid_backend_raises() -> None:
    """Verify synthesize raises ValueError for invalid backend."""
    with pytest.raises(ValueError, match="Invalid backend"):
        synthesize("Test", backend="invalid")  # type: ignore[arg-type]


# =============================================================================
# Tests for ElevenLabs TTS backend
# =============================================================================


def test_synthesize_elevenlabs_returns_audio_data(
    mock_elevenlabs_client: MagicMock,
) -> None:
    """Verify synthesize with ElevenLabs backend returns AudioData."""
    with patch("fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_client):
        result = synthesize("Hello world", backend="elevenlabs")

    assert isinstance(result, AudioData)
    assert isinstance(result.samples, np.ndarray)
    assert result.samples.dtype == np.float32
    assert result.sample_rate == 22050


def test_synthesize_elevenlabs_calls_api(mock_elevenlabs_client: MagicMock) -> None:
    """Verify synthesize with ElevenLabs backend calls the API correctly."""
    with patch("fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_client):
        synthesize("Test message", backend="elevenlabs")

    mock_elevenlabs_client.text_to_speech.convert.assert_called_once_with(
        voice_id="21m00Tcm4TlvDq8ikWAM",
        text="Test message",
        model_id="eleven_monolingual_v1",
        output_format="pcm_22050",
    )


def test_synthesize_elevenlabs_samples_normalized(
    mock_elevenlabs_client: MagicMock,
) -> None:
    """Verify ElevenLabs samples are normalized to [-1, 1] range."""
    with patch("fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_client):
        result = synthesize("Test", backend="elevenlabs")

    assert result.samples.min() >= -1.0
    assert result.samples.max() <= 1.0


def test_synthesize_elevenlabs_concatenates_chunks(
    mock_elevenlabs_client: MagicMock,
) -> None:
    """Verify ElevenLabs response chunks are concatenated correctly."""
    with patch("fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_client):
        result = synthesize("Test", backend="elevenlabs")

    # Verify that we got samples from both chunks (count depends on float precision)
    # Each chunk is ~0.05s at 22050 Hz (~1102 samples), so 2 chunks should be ~2204
    assert len(result.samples) > 2000  # At least 2 chunks worth
    assert len(result.samples) < 2500  # But not too many


def test_synthesize_elevenlabs_empty_text_raises() -> None:
    """Verify ElevenLabs backend raises ValueError for empty text."""
    with pytest.raises(ValueError, match="text cannot be empty"):
        synthesize("", backend="elevenlabs")


def test_synthesize_elevenlabs_whitespace_only_raises() -> None:
    """Verify ElevenLabs backend raises ValueError for whitespace-only text."""
    with pytest.raises(ValueError, match="text cannot be empty"):
        synthesize("   ", backend="elevenlabs")


def test_synthesize_elevenlabs_with_long_text(
    mock_elevenlabs_client: MagicMock,
) -> None:
    """Verify ElevenLabs backend works with longer text."""
    long_text = "This is a longer piece of text for ElevenLabs synthesis."

    with patch("fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_client):
        result = synthesize(long_text, backend="elevenlabs")

    assert isinstance(result, AudioData)
    call_args = mock_elevenlabs_client.text_to_speech.convert.call_args
    assert call_args.kwargs["text"] == long_text


def test_tts_backend_type_alias() -> None:
    """Verify TTSBackend type alias is exported correctly."""
    # This verifies the import works
    backend: TTSBackend = "openai"
    assert backend == "openai"
    backend = "elevenlabs"
    assert backend == "elevenlabs"


# =============================================================================
# Tests for voice selection
# =============================================================================


def test_get_available_voices_openai() -> None:
    """Verify get_available_voices returns OpenAI voices."""
    voices = get_available_voices("openai")
    assert "alloy" in voices
    assert "echo" in voices
    assert "nova" in voices
    assert "shimmer" in voices
    assert len(voices) == 9  # OpenAI has 9 voices


def test_get_available_voices_elevenlabs() -> None:
    """Verify get_available_voices returns ElevenLabs voices."""
    voices = get_available_voices("elevenlabs")
    assert "rachel" in voices
    assert "adam" in voices
    assert "josh" in voices
    assert len(voices) == 8  # ElevenLabs has 8 voices


def test_get_available_voices_invalid_backend_raises() -> None:
    """Verify get_available_voices raises ValueError for invalid backend."""
    with pytest.raises(ValueError, match="Invalid backend"):
        get_available_voices("invalid")  # type: ignore[arg-type]


def test_synthesize_openai_with_voice(mock_openai_client: MagicMock) -> None:
    """Verify synthesize passes voice to OpenAI API."""
    with patch("fai.voice.synthesize.OpenAI", return_value=mock_openai_client):
        synthesize("Test message", backend="openai", voice="echo")

    mock_openai_client.audio.speech.create.assert_called_once_with(
        model="tts-1",
        voice="echo",
        input="Test message",
        response_format="wav",
    )


def test_synthesize_openai_with_all_voices(mock_openai_client: MagicMock) -> None:
    """Verify all OpenAI voices are accepted."""
    openai_voices = [
        "alloy",
        "ash",
        "coral",
        "echo",
        "fable",
        "onyx",
        "nova",
        "sage",
        "shimmer",
    ]

    for voice in openai_voices:
        mock_openai_client.audio.speech.create.reset_mock()
        with patch("fai.voice.synthesize.OpenAI", return_value=mock_openai_client):
            synthesize("Test", backend="openai", voice=voice)

        assert mock_openai_client.audio.speech.create.call_args[1]["voice"] == voice


def test_synthesize_openai_invalid_voice_raises() -> None:
    """Verify OpenAI backend raises ValueError for invalid voice."""
    with pytest.raises(ValueError, match="Invalid OpenAI voice"):
        synthesize("Test", backend="openai", voice="invalid_voice")


def test_synthesize_openai_default_voice_is_alloy(
    mock_openai_client: MagicMock,
) -> None:
    """Verify OpenAI default voice is 'alloy'."""
    with patch("fai.voice.synthesize.OpenAI", return_value=mock_openai_client):
        synthesize("Test", backend="openai")

    assert mock_openai_client.audio.speech.create.call_args[1]["voice"] == "alloy"


def test_synthesize_elevenlabs_with_voice(mock_elevenlabs_client: MagicMock) -> None:
    """Verify synthesize passes voice to ElevenLabs API."""
    with patch("fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_client):
        synthesize("Test message", backend="elevenlabs", voice="adam")

    # Adam's voice ID
    expected_voice_id = "pNInz6obpgDQGcFmaJgB"
    mock_elevenlabs_client.text_to_speech.convert.assert_called_once_with(
        voice_id=expected_voice_id,
        text="Test message",
        model_id="eleven_monolingual_v1",
        output_format="pcm_22050",
    )


def test_synthesize_elevenlabs_with_all_voices(
    mock_elevenlabs_client: MagicMock,
) -> None:
    """Verify all ElevenLabs voices are accepted."""
    elevenlabs_voices = [
        "rachel",
        "adam",
        "antoni",
        "bella",
        "domi",
        "elli",
        "josh",
        "arnold",
    ]

    for voice in elevenlabs_voices:
        # Reset mock for fresh iterator each time
        mock_elevenlabs_client.text_to_speech.convert.reset_mock()
        mock_elevenlabs_client.text_to_speech.convert.return_value = iter(
            [create_mock_pcm_bytes(22050, 0.05)]
        )
        with patch(
            "fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_client
        ):
            synthesize("Test", backend="elevenlabs", voice=voice)

        mock_elevenlabs_client.text_to_speech.convert.assert_called_once()


def test_synthesize_elevenlabs_invalid_voice_raises() -> None:
    """Verify ElevenLabs backend raises ValueError for invalid voice."""
    with pytest.raises(ValueError, match="Invalid ElevenLabs voice"):
        synthesize("Test", backend="elevenlabs", voice="invalid_voice")


def test_synthesize_elevenlabs_default_voice_is_rachel(
    mock_elevenlabs_client: MagicMock,
) -> None:
    """Verify ElevenLabs default voice is 'rachel'."""
    with patch("fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_client):
        synthesize("Test", backend="elevenlabs")

    # Rachel's voice ID
    expected_voice_id = "21m00Tcm4TlvDq8ikWAM"
    assert (
        mock_elevenlabs_client.text_to_speech.convert.call_args[1]["voice_id"]
        == expected_voice_id
    )


def test_openai_voice_type_alias() -> None:
    """Verify OpenAIVoice type alias is exported correctly."""
    voice: OpenAIVoice = "alloy"
    assert voice == "alloy"
    voice = "shimmer"
    assert voice == "shimmer"


def test_elevenlabs_voice_type_alias() -> None:
    """Verify ElevenLabsVoice type alias is exported correctly."""
    voice: ElevenLabsVoice = "rachel"
    assert voice == "rachel"
    voice = "josh"
    assert voice == "josh"


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


# =============================================================================
# Tests for timeout parameter
# =============================================================================


def test_synthesize_passes_timeout_to_openai_client(
    mock_openai_client: MagicMock,
) -> None:
    """Verify synthesize passes timeout to OpenAI client constructor."""
    with patch(
        "fai.voice.synthesize.OpenAI", return_value=mock_openai_client
    ) as mock_cls:
        synthesize("Hello", timeout=30.0)

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs.get("timeout") == 30.0


def test_synthesize_passes_timeout_to_elevenlabs_client(
    mock_elevenlabs_client: MagicMock,
) -> None:
    """Verify synthesize passes timeout to ElevenLabs client constructor."""
    with patch(
        "fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_client
    ) as mock_cls:
        synthesize("Hello", backend="elevenlabs", timeout=45.0)

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs.get("timeout") == 45.0


def test_synthesize_no_timeout_omits_from_openai_client(
    mock_openai_client: MagicMock,
) -> None:
    """Verify synthesize omits timeout when None for OpenAI."""
    with patch(
        "fai.voice.synthesize.OpenAI", return_value=mock_openai_client
    ) as mock_cls:
        synthesize("Hello", timeout=None)

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert "timeout" not in call_kwargs


def test_synthesize_no_timeout_omits_from_elevenlabs_client(
    mock_elevenlabs_client: MagicMock,
) -> None:
    """Verify synthesize omits timeout when None for ElevenLabs."""
    with patch(
        "fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_client
    ) as mock_cls:
        synthesize("Hello", backend="elevenlabs", timeout=None)

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert "timeout" not in call_kwargs
