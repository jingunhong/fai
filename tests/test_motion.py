"""Tests for the motion component."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fai.motion import animate, get_available_backends
from fai.motion.animate import (
    _animate_with_auto_backend,
    _animate_with_specific_backend,
    _apply_breathing_effect,
    _generate_breathing_frames,
)
from fai.motion.backend import (
    calculate_audio_duration_ms,
    calculate_frame_count,
    read_video_frames,
)
from fai.types import AudioData, VideoFrame


@pytest.fixture  # type: ignore[misc]
def sample_audio() -> AudioData:
    """Create sample audio data for testing (1 second at 16kHz)."""
    samples = np.zeros(16000, dtype=np.float32)
    return AudioData(samples=samples, sample_rate=16000)


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


# === Backend utility tests ===


def test_calculate_audio_duration_ms_basic(sample_audio: AudioData) -> None:
    """Verify audio duration calculation for 1 second audio."""
    duration = calculate_audio_duration_ms(sample_audio)
    assert duration == 1000


def test_calculate_audio_duration_ms_zero_sample_rate() -> None:
    """Verify duration is 0 for zero sample rate."""
    audio = AudioData(samples=np.zeros(1000, dtype=np.float32), sample_rate=0)
    assert calculate_audio_duration_ms(audio) == 0


def test_calculate_frame_count_basic() -> None:
    """Verify frame count for 1 second at 30 FPS."""
    assert calculate_frame_count(1000) == 30


def test_calculate_frame_count_minimum() -> None:
    """Verify minimum frame count is 1."""
    assert calculate_frame_count(0) == 1
    assert calculate_frame_count(10) == 1


def test_calculate_frame_count_custom_fps() -> None:
    """Verify frame count with custom FPS."""
    assert calculate_frame_count(1000, fps=60) == 60


def test_read_video_frames_nonexistent_file(tmp_path: Path) -> None:
    """Verify read_video_frames yields nothing for nonexistent file."""
    video_path = tmp_path / "nonexistent.mp4"
    frames = list(read_video_frames(video_path))
    assert frames == []


def test_read_video_frames_yields_frames(
    tmp_path: Path, sample_image: np.ndarray
) -> None:
    """Verify read_video_frames yields VideoFrame objects."""
    from unittest.mock import MagicMock

    video_path = tmp_path / "test.mp4"
    video_path.touch()

    with patch("fai.motion.backend.cv2.VideoCapture") as mock_cap:
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [
            (True, sample_image.copy()),
            (True, sample_image.copy()),
            (False, None),
        ]
        mock_cap.return_value = mock_cap_instance

        frames = list(read_video_frames(video_path, fps=30))

        assert len(frames) == 2
        assert all(isinstance(f, VideoFrame) for f in frames)


def test_read_video_frames_timestamps(tmp_path: Path, sample_image: np.ndarray) -> None:
    """Verify read_video_frames generates correct timestamps."""
    from unittest.mock import MagicMock

    video_path = tmp_path / "test.mp4"
    video_path.touch()

    with patch("fai.motion.backend.cv2.VideoCapture") as mock_cap:
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [
            (True, sample_image.copy()),
            (True, sample_image.copy()),
            (True, sample_image.copy()),
            (False, None),
        ]
        mock_cap.return_value = mock_cap_instance

        frames = list(read_video_frames(video_path, fps=30))

        assert len(frames) == 3
        assert frames[0].timestamp_ms == 0
        assert frames[1].timestamp_ms == 33  # 1000/30 = 33.33
        assert frames[2].timestamp_ms == 66  # 2000/30 = 66.66


# === Breathing animation tests ===


def test_apply_breathing_effect_returns_same_shape(sample_image: np.ndarray) -> None:
    """Verify breathing effect preserves image shape."""
    result = _apply_breathing_effect(sample_image, 0)
    assert result.shape == sample_image.shape


def test_apply_breathing_effect_returns_uint8(sample_image: np.ndarray) -> None:
    """Verify breathing effect returns uint8 dtype."""
    result = _apply_breathing_effect(sample_image, 1000)
    assert result.dtype == np.uint8


def test_generate_breathing_frames_count(
    sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify breathing frames generates correct number of frames."""
    frames = list(_generate_breathing_frames(sample_image, sample_audio))
    # 1 second at 30 FPS = 30 frames
    assert len(frames) == 30


def test_generate_breathing_frames_timestamps(
    sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify breathing frames have sequential timestamps."""
    frames = list(_generate_breathing_frames(sample_image, sample_audio))
    timestamps = [f.timestamp_ms for f in frames]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


# === Main animate function tests ===


def test_animate_yields_video_frames(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify animate yields VideoFrame objects."""
    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, sample_audio))

    assert len(frames) > 0
    assert all(isinstance(f, VideoFrame) for f in frames)


def test_animate_frame_has_correct_shape(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify each frame has the same shape as the input image."""
    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, sample_audio))

    for frame in frames:
        assert frame.image.shape == sample_image.shape


def test_animate_timestamps_are_sequential(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify frame timestamps are sequential and increasing."""
    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, sample_audio))

    timestamps = [f.timestamp_ms for f in frames]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


def test_animate_frame_count_matches_audio_duration(
    mock_face_path: Path, sample_image: np.ndarray
) -> None:
    """Verify frame count approximately matches audio duration at 30fps."""
    # 2 seconds of audio at 16kHz
    samples = np.zeros(32000, dtype=np.float32)
    audio = AudioData(samples=samples, sample_rate=16000)

    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, audio))

    # 2 seconds at 30fps = 60 frames
    assert len(frames) == 60


def test_animate_missing_file_raises(sample_audio: AudioData) -> None:
    """Verify animate raises FileNotFoundError for missing image."""
    fake_path = Path("/nonexistent/face.jpg")

    with pytest.raises(FileNotFoundError, match="Face image not found"):
        list(animate(fake_path, sample_audio))


def test_animate_frame_dtype_is_uint8(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify frame images are uint8 dtype."""
    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, sample_audio))

    for frame in frames:
        assert frame.image.dtype == np.uint8


def test_animate_first_frame_starts_at_zero(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify the first frame has timestamp 0."""
    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, sample_audio))

    assert frames[0].timestamp_ms == 0


def test_animate_short_audio(mock_face_path: Path, sample_image: np.ndarray) -> None:
    """Verify animate handles very short audio (generates at least 1 frame)."""
    # Very short audio: 100 samples at 16kHz = 6.25ms
    short_audio = AudioData(
        samples=np.zeros(100, dtype=np.float32),
        sample_rate=16000,
    )

    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, short_audio))

    assert len(frames) >= 1


def test_animate_unreadable_image_raises(
    mock_face_path: Path, sample_audio: AudioData
) -> None:
    """Verify animate raises ValueError when cv2.imread returns None."""
    with (
        patch("fai.motion.animate.cv2.imread", return_value=None),
        pytest.raises(ValueError, match="Failed to read image"),
    ):
        list(animate(mock_face_path, sample_audio))


def test_animate_is_iterator(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify animate returns an iterator (yields frames lazily)."""
    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        result = animate(mock_face_path, sample_audio)

    # Should be an iterator, not a list
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


# === Backend selection tests ===


def test_animate_with_backend_none(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify backend='none' uses breathing animation."""
    with patch("fai.motion.animate.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, sample_audio, backend="none"))

    assert len(frames) == 30  # 1 second at 30 FPS


def test_animate_with_backend_auto_fallback(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify backend='auto' falls back to breathing when no backends available."""
    with (
        patch("fai.motion.animate.cv2.imread", return_value=sample_image),
        patch("fai.motion.animate.Wav2LipBackend.is_available", return_value=False),
        patch("fai.motion.animate.SadTalkerBackend.is_available", return_value=False),
    ):
        frames = list(animate(mock_face_path, sample_audio, backend="auto"))

    assert len(frames) == 30


def test_animate_with_auto_backend_uses_wav2lip_when_available(
    sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify auto backend prefers wav2lip when available."""
    mock_frame = VideoFrame(image=sample_image, timestamp_ms=0)

    with (
        patch("fai.motion.animate.Wav2LipBackend.is_available", return_value=True),
        patch(
            "fai.motion.animate.Wav2LipBackend.generate_frames",
            return_value=iter([mock_frame]),
        ) as mock_gen,
    ):
        frames = list(_animate_with_auto_backend(sample_image, sample_audio))

    mock_gen.assert_called_once()
    assert len(frames) == 1


def test_animate_with_specific_backend_raises_for_unknown(
    sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify unknown backend name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        list(_animate_with_specific_backend(sample_image, sample_audio, "unknown"))


def test_animate_with_specific_backend_raises_when_unavailable(
    sample_image: np.ndarray, sample_audio: AudioData
) -> None:
    """Verify unavailable backend raises RuntimeError."""
    with (
        patch("fai.motion.animate.Wav2LipBackend.is_available", return_value=False),
        pytest.raises(RuntimeError, match="not available"),
    ):
        list(_animate_with_specific_backend(sample_image, sample_audio, "wav2lip"))


def test_get_available_backends_none() -> None:
    """Verify get_available_backends returns empty when none available."""
    with (
        patch("fai.motion.animate.Wav2LipBackend.is_available", return_value=False),
        patch("fai.motion.animate.SadTalkerBackend.is_available", return_value=False),
    ):
        backends = get_available_backends()

    assert backends == []


def test_get_available_backends_all() -> None:
    """Verify get_available_backends returns all when available."""
    with (
        patch("fai.motion.animate.Wav2LipBackend.is_available", return_value=True),
        patch("fai.motion.animate.SadTalkerBackend.is_available", return_value=True),
    ):
        backends = get_available_backends()

    assert "wav2lip" in backends
    assert "sadtalker" in backends
