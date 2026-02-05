"""Tests for streaming mode components."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fai.dialogue import generate_response_stream
from fai.motion import animate_stream
from fai.orchestrator import run_conversation_stream
from fai.types import AudioChunk, AudioData, TextChunk, VideoFrame
from fai.voice import play_audio_stream, synthesize_stream
from tests.helpers import create_mock_pcm_bytes

# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture  # type: ignore[misc]
def sample_audio() -> AudioData:
    """Create sample audio data for testing (1 second at 16kHz)."""
    samples = np.zeros(16000, dtype=np.float32)
    return AudioData(samples=samples, sample_rate=16000)


@pytest.fixture  # type: ignore[misc]
def sample_audio_chunk() -> AudioChunk:
    """Create sample audio chunk for testing."""
    samples = np.zeros(4000, dtype=np.float32)  # 0.25 seconds at 16kHz
    return AudioChunk(samples=samples, sample_rate=16000, is_final=False)


@pytest.fixture  # type: ignore[misc]
def sample_image() -> np.ndarray:
    """Create a sample BGR image for testing."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture  # type: ignore[misc]
def mock_face_path(tmp_path: Path) -> Path:
    """Create a temporary face image file path."""
    face_path = tmp_path / "face.jpg"
    face_path.touch()
    return face_path


@pytest.fixture  # type: ignore[misc]
def mock_frame() -> VideoFrame:
    """Create a mock video frame."""
    return VideoFrame(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        timestamp_ms=0,
    )


# =============================================================================
# TextChunk and AudioChunk type tests
# =============================================================================


def test_text_chunk_creation() -> None:
    """Verify TextChunk can be created with default is_final."""
    chunk = TextChunk(text="Hello")
    assert chunk.text == "Hello"
    assert chunk.is_final is False


def test_text_chunk_final() -> None:
    """Verify TextChunk is_final flag works."""
    chunk = TextChunk(text="", is_final=True)
    assert chunk.is_final is True


def test_audio_chunk_creation(sample_audio_chunk: AudioChunk) -> None:
    """Verify AudioChunk can be created with all fields."""
    assert len(sample_audio_chunk.samples) == 4000
    assert sample_audio_chunk.sample_rate == 16000
    assert sample_audio_chunk.is_final is False


def test_audio_chunk_final() -> None:
    """Verify AudioChunk is_final flag works."""
    chunk = AudioChunk(
        samples=np.array([], dtype=np.float32),
        sample_rate=16000,
        is_final=True,
    )
    assert chunk.is_final is True


# =============================================================================
# Streaming dialogue generation tests
# =============================================================================


@pytest.fixture  # type: ignore[misc]
def mock_openai_stream() -> Iterator[MagicMock]:
    """Create a mock OpenAI streaming response."""
    mock_chunks = []
    for text in ["Hello", " there", "!"]:
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = text
        mock_chunks.append(mock_chunk)
    # Final chunk with no content
    final_chunk = MagicMock()
    final_chunk.choices = [MagicMock()]
    final_chunk.choices[0].delta.content = None
    mock_chunks.append(final_chunk)
    return iter(mock_chunks)


@pytest.fixture  # type: ignore[misc]
def mock_openai_client_stream(mock_openai_stream: MagicMock) -> MagicMock:
    """Create a mock OpenAI client for streaming."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_stream
    return mock_client


def test_generate_response_stream_yields_text_chunks(
    mock_openai_client_stream: MagicMock,
) -> None:
    """Verify generate_response_stream yields TextChunk objects."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client_stream):
        chunks = list(generate_response_stream("Hello", []))

    # Should have text chunks plus a final marker
    assert len(chunks) > 0
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_generate_response_stream_collects_full_response(
    mock_openai_client_stream: MagicMock,
) -> None:
    """Verify streaming chunks can be collected into full response."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client_stream):
        chunks = list(generate_response_stream("Hello", []))

    # Collect non-final text
    text = "".join(c.text for c in chunks if not c.is_final)
    assert text == "Hello there!"


def test_generate_response_stream_ends_with_final_chunk(
    mock_openai_client_stream: MagicMock,
) -> None:
    """Verify streaming ends with is_final=True chunk."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client_stream):
        chunks = list(generate_response_stream("Hello", []))

    assert chunks[-1].is_final is True


def test_generate_response_stream_empty_user_text_raises() -> None:
    """Verify empty user text raises ValueError."""
    with pytest.raises(ValueError, match="user_text cannot be empty"):
        list(generate_response_stream("", []))


def test_generate_response_stream_whitespace_only_raises() -> None:
    """Verify whitespace-only user text raises ValueError."""
    with pytest.raises(ValueError, match="user_text cannot be empty"):
        list(generate_response_stream("   ", []))


def test_generate_response_stream_invalid_backend_raises() -> None:
    """Verify invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Invalid backend"):
        list(generate_response_stream("Hello", [], backend="invalid"))  # type: ignore[arg-type]


def test_generate_response_stream_uses_streaming_api(
    mock_openai_client_stream: MagicMock,
) -> None:
    """Verify streaming API is used with stream=True."""
    with patch("fai.dialogue.generate.OpenAI", return_value=mock_openai_client_stream):
        list(generate_response_stream("Hello", []))

    # Verify stream=True was passed
    call_kwargs = mock_openai_client_stream.chat.completions.create.call_args[1]
    assert call_kwargs.get("stream") is True


def test_generate_response_stream_with_claude_backend() -> None:
    """Verify Claude backend streaming works."""
    mock_stream = MagicMock()
    mock_stream.text_stream = iter(["Hello", " from", " Claude"])

    mock_client = MagicMock()
    mock_client.messages.stream.return_value.__enter__ = MagicMock(
        return_value=mock_stream
    )
    mock_client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)

    with patch("fai.dialogue.generate.Anthropic", return_value=mock_client):
        chunks = list(generate_response_stream("Hello", [], backend="claude"))

    # Should have text chunks plus a final marker
    assert len(chunks) > 0
    text = "".join(c.text for c in chunks if not c.is_final)
    assert text == "Hello from Claude"


# =============================================================================
# Streaming TTS synthesis tests
# =============================================================================


@pytest.fixture  # type: ignore[misc]
def mock_elevenlabs_stream() -> MagicMock:
    """Create a mock ElevenLabs streaming client."""
    mock_client = MagicMock()
    mock_client.text_to_speech.convert.return_value = iter(
        [create_mock_pcm_bytes(22050, 0.05), create_mock_pcm_bytes(22050, 0.05)]
    )
    return mock_client


def test_synthesize_stream_yields_audio_chunks(
    mock_elevenlabs_stream: MagicMock,
) -> None:
    """Verify synthesize_stream yields AudioChunk objects."""
    with patch("fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_stream):
        chunks = list(synthesize_stream("Hello", backend="elevenlabs"))

    # Should have audio chunks
    assert len(chunks) > 0
    assert all(isinstance(c, AudioChunk) for c in chunks)


def test_synthesize_stream_ends_with_final_chunk(
    mock_elevenlabs_stream: MagicMock,
) -> None:
    """Verify synthesize_stream ends with is_final=True."""
    with patch("fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_stream):
        chunks = list(synthesize_stream("Hello", backend="elevenlabs"))

    assert chunks[-1].is_final is True


def test_synthesize_stream_empty_text_raises() -> None:
    """Verify empty text raises ValueError."""
    with pytest.raises(ValueError, match="text cannot be empty"):
        list(synthesize_stream(""))


def test_synthesize_stream_whitespace_only_raises() -> None:
    """Verify whitespace-only text raises ValueError."""
    with pytest.raises(ValueError, match="text cannot be empty"):
        list(synthesize_stream("   "))


def test_synthesize_stream_invalid_backend_raises() -> None:
    """Verify invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Invalid backend"):
        list(synthesize_stream("Hello", backend="invalid"))  # type: ignore[arg-type]


def test_synthesize_stream_openai_yields_single_chunk() -> None:
    """Verify OpenAI backend yields single chunk (no true streaming)."""
    import io
    import wave

    # Create mock WAV response
    def _create_mock_wav_bytes() -> bytes:
        n_samples = 2400
        samples = np.zeros(n_samples, dtype=np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(samples.tobytes())
        return buffer.getvalue()

    mock_response = MagicMock()
    mock_response.content = _create_mock_wav_bytes()

    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = mock_response

    with patch("fai.voice.synthesize.OpenAI", return_value=mock_client):
        chunks = list(synthesize_stream("Hello", backend="openai"))

    # OpenAI returns single chunk with is_final=True
    assert len(chunks) == 1
    assert chunks[0].is_final is True


def test_synthesize_stream_elevenlabs_with_voice(
    mock_elevenlabs_stream: MagicMock,
) -> None:
    """Verify ElevenLabs streaming with voice parameter."""
    with patch("fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_stream):
        list(synthesize_stream("Hello", backend="elevenlabs", voice="adam"))

    # Verify voice ID was passed (Adam's voice ID)
    call_kwargs = mock_elevenlabs_stream.text_to_speech.convert.call_args[1]
    assert call_kwargs.get("voice_id") == "pNInz6obpgDQGcFmaJgB"


def test_synthesize_stream_invalid_openai_voice_raises() -> None:
    """Verify invalid OpenAI voice raises ValueError."""
    with pytest.raises(ValueError, match="Invalid OpenAI voice"):
        list(synthesize_stream("Hello", backend="openai", voice="invalid"))


def test_synthesize_stream_invalid_elevenlabs_voice_raises() -> None:
    """Verify invalid ElevenLabs voice raises ValueError."""
    with pytest.raises(ValueError, match="Invalid ElevenLabs voice"):
        list(synthesize_stream("Hello", backend="elevenlabs", voice="invalid"))


# =============================================================================
# Streaming audio playback tests
# =============================================================================


def test_play_audio_stream_returns_combined_audio(
    sample_audio_chunk: AudioChunk,
) -> None:
    """Verify play_audio_stream returns combined AudioData."""
    chunk2 = AudioChunk(
        samples=np.ones(4000, dtype=np.float32),
        sample_rate=16000,
        is_final=True,
    )

    with patch("fai.voice.playback.sd"):
        result = play_audio_stream(iter([sample_audio_chunk, chunk2]), blocking=False)

    assert isinstance(result, AudioData)
    assert len(result.samples) == 8000  # 4000 + 4000
    assert result.sample_rate == 16000


def test_play_audio_stream_calls_sounddevice(
    sample_audio_chunk: AudioChunk,
) -> None:
    """Verify play_audio_stream calls sd.play."""
    with patch("fai.voice.playback.sd") as mock_sd:
        play_audio_stream(iter([sample_audio_chunk]), blocking=False)

    mock_sd.play.assert_called_once()


def test_play_audio_stream_blocking_waits(
    sample_audio_chunk: AudioChunk,
) -> None:
    """Verify play_audio_stream with blocking=True calls sd.wait."""
    with patch("fai.voice.playback.sd") as mock_sd:
        play_audio_stream(iter([sample_audio_chunk]), blocking=True)

    mock_sd.wait.assert_called_once()


def test_play_audio_stream_non_blocking_does_not_wait(
    sample_audio_chunk: AudioChunk,
) -> None:
    """Verify play_audio_stream with blocking=False does not wait."""
    with patch("fai.voice.playback.sd") as mock_sd:
        play_audio_stream(iter([sample_audio_chunk]), blocking=False)

    mock_sd.wait.assert_not_called()


def test_play_audio_stream_empty_chunks_raises() -> None:
    """Verify play_audio_stream raises for empty chunks."""
    empty_chunks: list[AudioChunk] = []

    with pytest.raises(ValueError, match="No audio chunks provided"):
        play_audio_stream(iter(empty_chunks))


def test_play_audio_stream_skips_empty_samples() -> None:
    """Verify play_audio_stream skips chunks with empty samples."""
    empty_chunk = AudioChunk(
        samples=np.array([], dtype=np.float32),
        sample_rate=16000,
        is_final=False,
    )
    valid_chunk = AudioChunk(
        samples=np.zeros(4000, dtype=np.float32),
        sample_rate=16000,
        is_final=True,
    )

    with patch("fai.voice.playback.sd"):
        result = play_audio_stream(iter([empty_chunk, valid_chunk]), blocking=False)

    # Only valid chunk should be included
    assert len(result.samples) == 4000


# =============================================================================
# Streaming animation tests
# =============================================================================


def test_animate_stream_yields_video_frames(
    mock_face_path: Path, sample_image: np.ndarray
) -> None:
    """Verify animate_stream yields VideoFrame objects."""
    chunk = AudioChunk(
        samples=np.zeros(16000, dtype=np.float32),  # 1 second
        sample_rate=16000,
        is_final=True,
    )

    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate_stream(mock_face_path, iter([chunk])))

    assert len(frames) > 0
    assert all(isinstance(f, VideoFrame) for f in frames)


def test_animate_stream_frame_count_matches_audio(
    mock_face_path: Path, sample_image: np.ndarray
) -> None:
    """Verify animate_stream generates frames matching audio duration."""
    # 1 second of audio at 16kHz
    chunk = AudioChunk(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        is_final=True,
    )

    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate_stream(mock_face_path, iter([chunk])))

    # 1 second at 30 FPS = 30 frames
    assert len(frames) == 30


def test_animate_stream_missing_file_raises(sample_audio_chunk: AudioChunk) -> None:
    """Verify animate_stream raises FileNotFoundError for missing image."""
    fake_path = Path("/nonexistent/face.jpg")

    with pytest.raises(FileNotFoundError, match="Face image not found"):
        list(animate_stream(fake_path, iter([sample_audio_chunk])))


def test_animate_stream_unreadable_image_raises(
    mock_face_path: Path, sample_audio_chunk: AudioChunk
) -> None:
    """Verify animate_stream raises ValueError when cv2.imread returns None."""
    with (
        patch("fai.motion.animate.cv2.imread", return_value=None),
        pytest.raises(ValueError, match="Failed to read image"),
    ):
        list(animate_stream(mock_face_path, iter([sample_audio_chunk])))


def test_animate_stream_handles_multiple_chunks(
    mock_face_path: Path, sample_image: np.ndarray
) -> None:
    """Verify animate_stream handles multiple audio chunks."""
    chunks = [
        AudioChunk(
            samples=np.zeros(8000, dtype=np.float32),  # 0.5 second
            sample_rate=16000,
            is_final=False,
        ),
        AudioChunk(
            samples=np.zeros(8000, dtype=np.float32),  # 0.5 second
            sample_rate=16000,
            is_final=True,
        ),
    ]

    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate_stream(mock_face_path, iter(chunks)))

    # Total 1 second at 30 FPS = 30 frames
    assert len(frames) == 30


def test_animate_stream_empty_chunks_yields_one_frame(
    mock_face_path: Path, sample_image: np.ndarray
) -> None:
    """Verify animate_stream yields at least one frame for empty audio."""
    empty_chunk = AudioChunk(
        samples=np.array([], dtype=np.float32),
        sample_rate=16000,
        is_final=True,
    )

    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate_stream(mock_face_path, iter([empty_chunk])))

    # Should yield at least 1 frame even for empty audio
    assert len(frames) >= 1


def test_animate_stream_timestamps_are_sequential(
    mock_face_path: Path, sample_image: np.ndarray
) -> None:
    """Verify animate_stream generates sequential timestamps."""
    chunk = AudioChunk(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        is_final=True,
    )

    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate_stream(mock_face_path, iter([chunk])))

    timestamps = [f.timestamp_ms for f in frames]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


# =============================================================================
# Streaming orchestrator tests
# =============================================================================


def test_run_conversation_stream_missing_image_raises() -> None:
    """Verify run_conversation_stream raises FileNotFoundError."""
    fake_path = Path("/nonexistent/face.jpg")

    with pytest.raises(FileNotFoundError, match="Face image not found"):
        run_conversation_stream(fake_path)


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate_stream")
@patch("fai.orchestrator.loop.play_audio_stream")
@patch("fai.orchestrator.loop.synthesize_stream")
@patch("fai.orchestrator.loop.generate_response_stream")
@patch("builtins.input")
def test_run_conversation_stream_single_turn(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    sample_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify streaming conversation flow for single turn."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = iter(
        [
            TextChunk(text="Hi there!", is_final=False),
            TextChunk(text="", is_final=True),
        ]
    )
    mock_synthesize.return_value = iter(
        [
            AudioChunk(
                samples=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                is_final=True,
            )
        ]
    )
    mock_play.return_value = sample_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation_stream(mock_face_path, text_mode=True)

    # Verify streaming functions were called
    mock_generate.assert_called_once()
    mock_synthesize.assert_called_once()
    mock_play.assert_called_once()
    mock_animate.assert_called_once()
    mock_display.assert_called_once()


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate_stream")
@patch("fai.orchestrator.loop.play_audio_stream")
@patch("fai.orchestrator.loop.synthesize_stream")
@patch("fai.orchestrator.loop.generate_response_stream")
@patch("builtins.input")
def test_run_conversation_stream_passes_dialogue_backend(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    sample_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify dialogue backend is passed to generate_response_stream."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = iter([TextChunk(text="", is_final=True)])
    mock_synthesize.return_value = iter(
        [
            AudioChunk(
                samples=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                is_final=True,
            )
        ]
    )
    mock_play.return_value = sample_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation_stream(mock_face_path, text_mode=True, dialogue_backend="claude")

    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs.get("backend") == "claude"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate_stream")
@patch("fai.orchestrator.loop.play_audio_stream")
@patch("fai.orchestrator.loop.synthesize_stream")
@patch("fai.orchestrator.loop.generate_response_stream")
@patch("builtins.input")
def test_run_conversation_stream_passes_tts_backend(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    sample_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify TTS backend is passed to synthesize_stream."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = iter([TextChunk(text="", is_final=True)])
    mock_synthesize.return_value = iter(
        [
            AudioChunk(
                samples=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                is_final=True,
            )
        ]
    )
    mock_play.return_value = sample_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation_stream(mock_face_path, text_mode=True, tts_backend="elevenlabs")

    call_kwargs = mock_synthesize.call_args[1]
    assert call_kwargs.get("backend") == "elevenlabs"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate_stream")
@patch("fai.orchestrator.loop.play_audio_stream")
@patch("fai.orchestrator.loop.synthesize_stream")
@patch("fai.orchestrator.loop.generate_response_stream")
@patch("builtins.input")
def test_run_conversation_stream_passes_voice(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    sample_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify voice is passed to synthesize_stream."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = iter([TextChunk(text="", is_final=True)])
    mock_synthesize.return_value = iter(
        [
            AudioChunk(
                samples=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                is_final=True,
            )
        ]
    )
    mock_play.return_value = sample_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation_stream(mock_face_path, text_mode=True, voice="echo")

    call_kwargs = mock_synthesize.call_args[1]
    assert call_kwargs.get("voice") == "echo"


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate_stream")
@patch("fai.orchestrator.loop.play_audio_stream")
@patch("fai.orchestrator.loop.synthesize_stream")
@patch("fai.orchestrator.loop.generate_response_stream")
@patch("builtins.input")
def test_run_conversation_stream_skips_empty_input(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    sample_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify empty input is skipped in streaming mode."""
    mock_input.side_effect = ["", "   ", "Hello", KeyboardInterrupt]
    mock_generate.return_value = iter([TextChunk(text="", is_final=True)])
    mock_synthesize.return_value = iter(
        [
            AudioChunk(
                samples=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                is_final=True,
            )
        ]
    )
    mock_play.return_value = sample_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation_stream(mock_face_path, text_mode=True)

    # Should only process "Hello"
    assert mock_generate.call_count == 1


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate_stream")
@patch("fai.orchestrator.loop.play_audio_stream")
@patch("fai.orchestrator.loop.synthesize_stream")
@patch("fai.orchestrator.loop.generate_response_stream")
@patch("builtins.input")
def test_run_conversation_stream_handles_keyboard_interrupt(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
) -> None:
    """Verify KeyboardInterrupt exits gracefully in streaming mode."""
    mock_input.side_effect = KeyboardInterrupt

    # Should not raise
    run_conversation_stream(mock_face_path, text_mode=True)

    mock_generate.assert_not_called()


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate_stream")
@patch("fai.orchestrator.loop.play_audio_stream")
@patch("fai.orchestrator.loop.synthesize_stream")
@patch("fai.orchestrator.loop.generate_response_stream")
@patch("builtins.input")
def test_run_conversation_stream_maintains_history(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    sample_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify conversation history is maintained in streaming mode."""
    mock_input.side_effect = ["First", "Second", KeyboardInterrupt]

    captured_histories: list[list[dict[str, str]]] = []

    def capture_history(
        user_text: str, history: list[dict[str, str]], **kwargs: str
    ) -> Iterator[TextChunk]:
        captured_histories.append(list(history))
        return iter(
            [
                TextChunk(text=f"Response to {user_text}", is_final=False),
                TextChunk(text="", is_final=True),
            ]
        )

    mock_generate.side_effect = capture_history
    mock_synthesize.return_value = iter(
        [
            AudioChunk(
                samples=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                is_final=True,
            )
        ]
    )
    mock_play.return_value = sample_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation_stream(mock_face_path, text_mode=True)

    # First call should have empty history
    assert captured_histories[0] == []

    # Second call should have first turn in history
    assert len(captured_histories[1]) == 2
    assert captured_histories[1][0] == {"role": "user", "content": "First"}
    assert captured_histories[1][1] == {
        "role": "assistant",
        "content": "Response to First",
    }


# =============================================================================
# Streaming timeout tests
# =============================================================================


def test_generate_response_stream_passes_timeout_to_openai_client(
    mock_openai_client_stream: MagicMock,
) -> None:
    """Verify generate_response_stream passes timeout to OpenAI client."""
    with patch(
        "fai.dialogue.generate.OpenAI", return_value=mock_openai_client_stream
    ) as mock_cls:
        list(generate_response_stream("Hello", [], timeout=30.0))

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs.get("timeout") == 30.0


def test_generate_response_stream_no_timeout_omits_from_client(
    mock_openai_client_stream: MagicMock,
) -> None:
    """Verify generate_response_stream omits timeout when None."""
    with patch(
        "fai.dialogue.generate.OpenAI", return_value=mock_openai_client_stream
    ) as mock_cls:
        list(generate_response_stream("Hello", [], timeout=None))

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert "timeout" not in call_kwargs


def test_generate_response_stream_claude_passes_timeout() -> None:
    """Verify Claude streaming passes timeout to client constructor."""
    mock_stream = MagicMock()
    mock_stream.text_stream = iter(["Hello"])

    mock_client = MagicMock()
    mock_client.messages.stream.return_value.__enter__ = MagicMock(
        return_value=mock_stream
    )
    mock_client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)

    with patch("fai.dialogue.generate.Anthropic", return_value=mock_client) as mock_cls:
        list(generate_response_stream("Hello", [], backend="claude", timeout=45.0))

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs.get("timeout") == 45.0


def test_synthesize_stream_passes_timeout_to_elevenlabs_client(
    mock_elevenlabs_stream: MagicMock,
) -> None:
    """Verify synthesize_stream passes timeout to ElevenLabs client."""
    with patch(
        "fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_stream
    ) as mock_cls:
        list(synthesize_stream("Hello", backend="elevenlabs", timeout=30.0))

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs.get("timeout") == 30.0


def test_synthesize_stream_no_timeout_omits_from_elevenlabs_client(
    mock_elevenlabs_stream: MagicMock,
) -> None:
    """Verify synthesize_stream omits timeout when None for ElevenLabs."""
    with patch(
        "fai.voice.synthesize.ElevenLabs", return_value=mock_elevenlabs_stream
    ) as mock_cls:
        list(synthesize_stream("Hello", backend="elevenlabs", timeout=None))

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert "timeout" not in call_kwargs


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate_stream")
@patch("fai.orchestrator.loop.play_audio_stream")
@patch("fai.orchestrator.loop.synthesize_stream")
@patch("fai.orchestrator.loop.generate_response_stream")
@patch("builtins.input")
def test_run_conversation_stream_passes_timeout(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    mock_face_path: Path,
    sample_audio: AudioData,
    mock_frame: VideoFrame,
) -> None:
    """Verify timeout is passed through in streaming conversation."""
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = iter([TextChunk(text="Hi", is_final=True)])
    mock_synthesize.return_value = iter(
        [
            AudioChunk(
                samples=np.zeros(16000, dtype=np.float32),
                sample_rate=16000,
                is_final=True,
            )
        ]
    )
    mock_play.return_value = sample_audio
    mock_animate.return_value = iter([mock_frame])

    run_conversation_stream(mock_face_path, text_mode=True, timeout=30.0)

    # Verify timeout was passed to generate_response_stream
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs.get("timeout") == 30.0

    # Verify timeout was passed to synthesize_stream
    call_kwargs = mock_synthesize.call_args[1]
    assert call_kwargs.get("timeout") == 30.0
