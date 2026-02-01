"""Tests for the motion component."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fai.motion import animate
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


def test_animate_yields_video_frames(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify animate yields VideoFrame objects."""
    with patch("fai.motion.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, sample_audio))

    assert len(frames) > 0
    assert all(isinstance(f, VideoFrame) for f in frames)


def test_animate_frame_has_correct_shape(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify each frame has the same shape as the input image."""
    with patch("fai.motion.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, sample_audio))

    for frame in frames:
        assert frame.image.shape == sample_image.shape


def test_animate_timestamps_are_sequential(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify frame timestamps are sequential and increasing."""
    with patch("fai.motion.cv2.imread", return_value=sample_image):
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

    with patch("fai.motion.cv2.imread", return_value=sample_image):
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
    with patch("fai.motion.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, sample_audio))

    for frame in frames:
        assert frame.image.dtype == np.uint8


def test_animate_first_frame_starts_at_zero(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify the first frame has timestamp 0."""
    with patch("fai.motion.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, sample_audio))

    assert frames[0].timestamp_ms == 0


def test_animate_short_audio(mock_face_path: Path, sample_image: np.ndarray) -> None:
    """Verify animate handles very short audio (generates at least 1 frame)."""
    # Very short audio: 100 samples at 16kHz = 6.25ms
    short_audio = AudioData(
        samples=np.zeros(100, dtype=np.float32),
        sample_rate=16000,
    )

    with patch("fai.motion.cv2.imread", return_value=sample_image):
        frames = list(animate(mock_face_path, short_audio))

    assert len(frames) >= 1


def test_animate_unreadable_image_raises(
    mock_face_path: Path, sample_audio: AudioData
) -> None:
    """Verify animate raises ValueError when cv2.imread returns None."""
    with (
        patch("fai.motion.cv2.imread", return_value=None),
        pytest.raises(ValueError, match="Failed to read image"),
    ):
        list(animate(mock_face_path, sample_audio))


def test_animate_is_iterator(
    mock_face_path: Path, sample_audio: AudioData, sample_image: np.ndarray
) -> None:
    """Verify animate returns an iterator (yields frames lazily)."""
    with patch("fai.motion.cv2.imread", return_value=sample_image):
        result = animate(mock_face_path, sample_audio)

    # Should be an iterator, not a list
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")
