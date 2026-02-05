"""Tests for the recording component."""

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fai.recording import SessionRecorder
from fai.recording.record import (
    load_audio_wav,
    load_session_metadata,
    save_audio_wav,
    save_video_frames,
)
from fai.types import AudioData, VideoFrame


@pytest.fixture  # type: ignore[misc]
def sample_audio() -> AudioData:
    """Create sample audio data for testing."""
    # Create a simple sine wave
    sample_rate = 16000
    duration = 0.5  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return AudioData(samples=samples, sample_rate=sample_rate)


@pytest.fixture  # type: ignore[misc]
def sample_frame() -> VideoFrame:
    """Create a sample video frame for testing."""
    # Create a simple colored frame
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # Blue channel
    return VideoFrame(image=image, timestamp_ms=0)


@pytest.fixture  # type: ignore[misc]
def sample_frames() -> list[VideoFrame]:
    """Create multiple sample video frames for testing."""
    frames = []
    for i in range(10):
        image = np.full((100, 100, 3), i * 25, dtype=np.uint8)
        frames.append(VideoFrame(image=image, timestamp_ms=i * 33))
    return frames


# === save_audio_wav tests ===


def test_save_audio_wav_creates_file(tmp_path: Path, sample_audio: AudioData) -> None:
    """Verify save_audio_wav creates a valid WAV file."""
    output_path = tmp_path / "test.wav"

    save_audio_wav(sample_audio, output_path)

    assert output_path.exists()


def test_save_audio_wav_file_is_valid_wav(
    tmp_path: Path, sample_audio: AudioData
) -> None:
    """Verify saved file is a valid WAV with correct parameters."""
    output_path = tmp_path / "test.wav"

    save_audio_wav(sample_audio, output_path)

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1  # Mono
        assert wav_file.getsampwidth() == 2  # 16-bit
        assert wav_file.getframerate() == sample_audio.sample_rate


def test_save_audio_wav_empty_samples_raises(tmp_path: Path) -> None:
    """Verify save_audio_wav raises ValueError for empty samples."""
    audio = AudioData(samples=np.array([], dtype=np.float32), sample_rate=16000)
    output_path = tmp_path / "test.wav"

    with pytest.raises(ValueError, match="audio samples cannot be empty"):
        save_audio_wav(audio, output_path)


def test_save_audio_wav_preserves_sample_count(
    tmp_path: Path, sample_audio: AudioData
) -> None:
    """Verify sample count is preserved in saved file."""
    output_path = tmp_path / "test.wav"

    save_audio_wav(sample_audio, output_path)

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnframes() == len(sample_audio.samples)


# === load_audio_wav tests ===


def test_load_audio_wav_returns_audio_data(
    tmp_path: Path, sample_audio: AudioData
) -> None:
    """Verify load_audio_wav returns correct AudioData."""
    output_path = tmp_path / "test.wav"
    save_audio_wav(sample_audio, output_path)

    loaded = load_audio_wav(output_path)

    assert isinstance(loaded, AudioData)
    assert loaded.sample_rate == sample_audio.sample_rate
    assert len(loaded.samples) == len(sample_audio.samples)


def test_load_audio_wav_file_not_found(tmp_path: Path) -> None:
    """Verify load_audio_wav raises FileNotFoundError for missing file."""
    nonexistent = tmp_path / "nonexistent.wav"

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_audio_wav(nonexistent)


def test_save_load_roundtrip_preserves_values(
    tmp_path: Path, sample_audio: AudioData
) -> None:
    """Verify save/load roundtrip preserves audio values within tolerance."""
    output_path = tmp_path / "test.wav"
    save_audio_wav(sample_audio, output_path)

    loaded = load_audio_wav(output_path)

    # Allow for int16 quantization error
    np.testing.assert_allclose(
        loaded.samples, sample_audio.samples, rtol=0, atol=1 / 32767
    )


# === save_video_frames tests ===


def test_save_video_frames_creates_file(
    tmp_path: Path, sample_frames: list[VideoFrame]
) -> None:
    """Verify save_video_frames creates a video file."""
    output_path = tmp_path / "test.mp4"

    save_video_frames(sample_frames, output_path)

    assert output_path.exists()


def test_save_video_frames_empty_raises(tmp_path: Path) -> None:
    """Verify save_video_frames raises ValueError for empty frames."""
    output_path = tmp_path / "test.mp4"

    with pytest.raises(ValueError, match="no frames to save"):
        save_video_frames([], output_path)


def test_save_video_frames_with_iterator(
    tmp_path: Path, sample_frames: list[VideoFrame]
) -> None:
    """Verify save_video_frames works with iterator input."""
    output_path = tmp_path / "test.mp4"

    save_video_frames(iter(sample_frames), output_path)

    assert output_path.exists()


def test_save_video_frames_uses_correct_codec(
    tmp_path: Path, sample_frame: VideoFrame
) -> None:
    """Verify save_video_frames uses specified codec."""
    output_path = tmp_path / "test.mp4"

    # Should not raise - default codec "mp4v" should work
    save_video_frames([sample_frame], output_path)

    assert output_path.exists()


# === SessionRecorder tests ===


def test_session_recorder_creates_session_dir(tmp_path: Path) -> None:
    """Verify SessionRecorder.start() creates session directory."""
    recorder = SessionRecorder(tmp_path)

    recorder.start()

    assert recorder.session_dir.exists()
    assert recorder.session_dir.parent == tmp_path


def test_session_recorder_session_id_format(tmp_path: Path) -> None:
    """Verify session ID has correct timestamp format."""
    recorder = SessionRecorder(tmp_path)

    # Should match YYYY-MM-DD_HHMMSS format
    assert len(recorder.session_id) == 17
    assert recorder.session_id[4] == "-"
    assert recorder.session_id[7] == "-"
    assert recorder.session_id[10] == "_"


def test_session_recorder_session_dir_has_session_prefix(
    tmp_path: Path,
) -> None:
    """Verify session directory has 'session_' prefix."""
    recorder = SessionRecorder(tmp_path)

    assert recorder.session_dir.name.startswith("session_")


def test_record_turn_creates_turn_dir(tmp_path: Path, sample_audio: AudioData) -> None:
    """Verify record_turn creates turn directory."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()

    turn_dir = recorder.record_turn(
        user_text="Hello",
        response_text="Hi there!",
    )

    assert turn_dir.exists()
    assert turn_dir.name == "turn_001"


def test_record_turn_increments_turn_number(tmp_path: Path) -> None:
    """Verify record_turn increments turn number."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()

    turn1 = recorder.record_turn(user_text="First", response_text="Response 1")
    turn2 = recorder.record_turn(user_text="Second", response_text="Response 2")

    assert turn1.name == "turn_001"
    assert turn2.name == "turn_002"


def test_record_turn_saves_user_audio(tmp_path: Path, sample_audio: AudioData) -> None:
    """Verify record_turn saves user audio when provided."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()

    turn_dir = recorder.record_turn(
        user_text="Hello",
        response_text="Hi!",
        user_audio=sample_audio,
    )

    user_audio_path = turn_dir / "user_audio.wav"
    assert user_audio_path.exists()


def test_record_turn_saves_response_audio(
    tmp_path: Path, sample_audio: AudioData
) -> None:
    """Verify record_turn saves response audio when provided."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()

    turn_dir = recorder.record_turn(
        user_text="Hello",
        response_text="Hi!",
        response_audio=sample_audio,
    )

    response_audio_path = turn_dir / "response_audio.wav"
    assert response_audio_path.exists()


def test_record_turn_saves_video(
    tmp_path: Path, sample_frames: list[VideoFrame]
) -> None:
    """Verify record_turn saves video when provided."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()

    turn_dir = recorder.record_turn(
        user_text="Hello",
        response_text="Hi!",
        video_frames=sample_frames,
    )

    video_path = turn_dir / "response_video.mp4"
    assert video_path.exists()


def test_finalize_creates_metadata_file(tmp_path: Path) -> None:
    """Verify finalize creates session.json."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()
    recorder.record_turn(user_text="Test", response_text="Response")

    metadata_path = recorder.finalize()

    assert metadata_path.exists()
    assert metadata_path.name == "session.json"


def test_finalize_metadata_contains_session_info(tmp_path: Path) -> None:
    """Verify metadata contains session information."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()
    recorder.record_turn(user_text="Hello", response_text="Hi!")
    recorder.record_turn(user_text="Bye", response_text="Goodbye!")

    metadata_path = recorder.finalize()

    with open(metadata_path) as f:
        data = json.load(f)

    assert data["session_id"] == recorder.session_id
    assert data["total_turns"] == 2
    assert len(data["turns"]) == 2
    assert data["turns"][0]["user_text"] == "Hello"
    assert data["turns"][0]["response_text"] == "Hi!"
    assert data["turns"][1]["user_text"] == "Bye"


def test_finalize_includes_custom_metadata(tmp_path: Path) -> None:
    """Verify finalize includes custom metadata."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()
    recorder.record_turn(user_text="Test", response_text="Response")

    custom_metadata = {"backend": "wav2lip", "text_mode": True}
    recorder.finalize(metadata=custom_metadata)

    data = load_session_metadata(recorder.session_dir)
    assert data["metadata"]["backend"] == "wav2lip"
    assert data["metadata"]["text_mode"] is True


def test_finalize_is_idempotent(tmp_path: Path) -> None:
    """Verify calling finalize multiple times is safe."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()
    recorder.record_turn(user_text="Test", response_text="Response")

    path1 = recorder.finalize()
    path2 = recorder.finalize()

    assert path1 == path2


def test_turn_data_includes_files_info(
    tmp_path: Path, sample_audio: AudioData, sample_frames: list[VideoFrame]
) -> None:
    """Verify turn data includes file references."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()
    recorder.record_turn(
        user_text="Test",
        response_text="Response",
        user_audio=sample_audio,
        response_audio=sample_audio,
        video_frames=sample_frames,
    )

    recorder.finalize()
    data = load_session_metadata(recorder.session_dir)

    files = data["turns"][0]["files"]
    assert files["user_audio"] == "user_audio.wav"
    assert files["response_audio"] == "response_audio.wav"
    assert files["response_video"] == "response_video.mp4"


# === load_session_metadata tests ===


def test_load_session_metadata_returns_dict(tmp_path: Path) -> None:
    """Verify load_session_metadata returns correct data."""
    recorder = SessionRecorder(tmp_path)
    recorder.start()
    recorder.record_turn(user_text="Test", response_text="Response")
    recorder.finalize()

    data = load_session_metadata(recorder.session_dir)

    assert isinstance(data, dict)
    assert "session_id" in data
    assert "turns" in data


def test_load_session_metadata_file_not_found(tmp_path: Path) -> None:
    """Verify load_session_metadata raises for missing file."""
    nonexistent = tmp_path / "nonexistent_session"
    nonexistent.mkdir()

    with pytest.raises(FileNotFoundError, match="Session metadata not found"):
        load_session_metadata(nonexistent)


# === Recording integration tests ===


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_with_recording(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    tmp_path: Path,
    sample_audio: AudioData,
    sample_frame: VideoFrame,
) -> None:
    """Verify recording works in orchestrator loop."""
    from fai.orchestrator import run_conversation
    from fai.types import DialogueResponse

    # Create face image
    face_path = tmp_path / "face.jpg"
    face_path.touch()

    # Setup mocks
    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi there!")
    mock_synthesize.return_value = sample_audio
    mock_animate.return_value = iter([sample_frame])

    output_dir = tmp_path / "recordings"

    run_conversation(
        face_path,
        text_mode=True,
        record=True,
        output_dir=output_dir,
    )

    # Verify recording was created
    assert output_dir.exists()

    # Find the session directory
    session_dirs = list(output_dir.glob("session_*"))
    assert len(session_dirs) == 1

    session_dir = session_dirs[0]

    # Verify session.json was created
    metadata_path = session_dir / "session.json"
    assert metadata_path.exists()

    # Verify metadata content
    with open(metadata_path) as f:
        data = json.load(f)

    assert data["total_turns"] == 1
    assert data["turns"][0]["user_text"] == "Hello"
    assert data["turns"][0]["response_text"] == "Hi there!"
    assert data["metadata"]["text_mode"] is True


@patch("fai.orchestrator.loop.display")
@patch("fai.orchestrator.loop.animate")
@patch("fai.orchestrator.loop.play_audio")
@patch("fai.orchestrator.loop.synthesize")
@patch("fai.orchestrator.loop.generate_response")
@patch("builtins.input")
def test_run_conversation_without_recording(
    mock_input: MagicMock,
    mock_generate: MagicMock,
    mock_synthesize: MagicMock,
    mock_play_audio: MagicMock,
    mock_animate: MagicMock,
    mock_display: MagicMock,
    tmp_path: Path,
    sample_audio: AudioData,
    sample_frame: VideoFrame,
) -> None:
    """Verify no recording when record=False."""
    from fai.orchestrator import run_conversation
    from fai.types import DialogueResponse

    face_path = tmp_path / "face.jpg"
    face_path.touch()

    mock_input.side_effect = ["Hello", KeyboardInterrupt]
    mock_generate.return_value = DialogueResponse(text="Hi!")
    mock_synthesize.return_value = sample_audio
    mock_animate.return_value = iter([sample_frame])

    output_dir = tmp_path / "recordings"

    run_conversation(
        face_path,
        text_mode=True,
        record=False,
        output_dir=output_dir,
    )

    # Verify no recording directory was created
    assert not output_dir.exists()
